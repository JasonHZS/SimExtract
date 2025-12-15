"""ColBERT late interaction attribution method using BGE-M3 multi-vectors."""

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..base import AttributionMethod, AttributionResult, AttributionSpan
from .sparse import _get_cached_bge_m3_model
from .utils import normalize_score

logger = logging.getLogger(__name__)


@dataclass
class WindowScoreResult:
    """Result from window score computation.
    
    Attributes:
        segment_scores: Tensor of shape [N_windows] with scores for each window
        window_size: Actual window size used (may be clamped to doc length)
        stride: Stride between windows
        n_windows: Number of windows computed
    """
    segment_scores: torch.Tensor
    window_size: int
    stride: int
    n_windows: int


class ColBERTAttribution(AttributionMethod):
    """Similarity attribution via ColBERT-style late interaction.

    This method generates token-level embeddings for both texts using BGE-M3
    and computes MaxSim scores within sliding windows to identify which spans
    in text_b are most similar to text_a.

    ColBERT uses multi-vector representations where each token gets its own
    embedding vector. The similarity between two texts is computed as:

        s_mul = (1/N) * sum_{i=1}^{N} max_{j=1}^{M} E_q[i] · E_p[j]

    where E_q and E_p are the token embeddings of query and passage.

    For span extraction, we use sliding windows with max pooling:
    1. Compute the full interaction matrix: Q × D^T
    2. For each query token, find max similarity within each window
    3. Sum scores across query tokens to get window scores
    4. Return top-k windows as attribution spans
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize ColBERTAttribution method.

        Args:
            config: Configuration dictionary containing:
                - model_name: str, BGE-M3 model path (default: "BAAI/bge-m3")
                - use_fp16: bool, whether to use half precision (default: True)
                - device: str, device to use (default: auto-detect)
                - window_size: int, sliding window token count (default: 50)
                - window_overlap: int, token overlap between windows (default: 10)
                - top_k_spans: int, number of top spans to return (default: 3)

        Raises:
            RuntimeError: If model initialization fails
        """
        super().__init__(config)

        self.model_name = config.get("model_name", "BAAI/bge-m3")
        self.use_fp16 = config.get("use_fp16", True)
        self.device = config.get("device", None)
        self.query_max_length = config.get("query_max_length", 1024)
        self.passage_max_length = config.get("passage_max_length", 1024)
        self.window_size = config.get("window_size", 50)
        self.window_overlap = config.get("window_overlap", 10)
        self.top_k_spans = config.get("top_k_spans", 3)
        self.top_k_topic_keywords = config.get("top_k_topic_keywords", 5)

        # Validate window parameters
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.window_overlap < 0:
            raise ValueError("window_overlap must be non-negative")
        if self.window_overlap >= self.window_size:
            raise ValueError("window_overlap must be less than window_size")
        if self.top_k_topic_keywords <= 0:
            raise ValueError("top_k_topic_keywords must be positive")

        # Initialize the model
        self._model = None
        self._initialize_model()

        logger.info(
            f"ColBERTAttribution initialized: model={self.model_name}, "
            f"window_size={self.window_size}, window_overlap={self.window_overlap}"
        )

    def _initialize_model(self) -> None:
        """Initialize the BGE-M3 model.

        Raises:
            RuntimeError: If model initialization fails
        """
        try:
            if not isinstance(self.model_name, str) or not self.model_name.strip():
                raise ValueError("model_name must be a non-empty string")

            # NOTE: model instances are cached per-process to avoid repeated loads
            self._model = _get_cached_bge_m3_model(
                self.model_name,
                self.use_fp16,
                self.device,
                query_max_length=self.query_max_length,
                passage_max_length=self.passage_max_length
            )
            logger.info(f"BGE-M3 model loaded for ColBERT: {self.model_name}")

        except ImportError as e:
            raise RuntimeError(
                "FlagEmbedding is required for ColBERTAttribution. "
                "Install it with: pip install FlagEmbedding"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize BGE-M3 model: {e}") from e

    @property
    def model(self):
        """Get the BGE-M3 model instance."""
        if self._model is None:
            raise RuntimeError("Model not initialized")
        return self._model

    @property
    def tokenizer(self):
        """Get the tokenizer from the model."""
        return self.model.tokenizer

    def _encode_colbert(self, texts: List[str]) -> List[torch.Tensor]:
        """Encode texts to ColBERT multi-vector embeddings.

        Args:
            texts: List of texts to encode

        Returns:
            List of tensors, each with shape [num_tokens, embedding_dim]
        """
        output = self.model.encode(
            texts,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True
        )
        return output['colbert_vecs']

    def _compute_interaction_matrix(
        self,
        query_vecs: torch.Tensor,
        doc_vecs: torch.Tensor
    ) -> torch.Tensor:
        """Compute the cosine similarity interaction matrix between query and doc vectors.

        This normalizes both sets of vectors and computes the dot product matrix,
        resulting in cosine similarities between all query-doc token pairs.

        Args:
            query_vecs: Query token embeddings, shape [N_q, D]
            doc_vecs: Document token embeddings, shape [N_d, D]

        Returns:
            Interaction matrix of shape [N_q, N_d] where entry [i, j] is the
            cosine similarity between query token i and document token j.
        """
        # Normalize vectors for cosine similarity
        query_normed = F.normalize(query_vecs.float(), p=2, dim=-1)
        doc_normed = F.normalize(doc_vecs.float(), p=2, dim=-1)

        # Compute interaction matrix: [N_q, N_d]
        sim_matrix = torch.matmul(query_normed, doc_normed.transpose(-2, -1))

        return sim_matrix

    def _compute_window_scores(
        self,
        sim_matrix: torch.Tensor,
        doc_length: int
    ) -> Optional[WindowScoreResult]:
        """Compute scores for sliding windows over the document.

        Uses 1D max pooling to find the best matching query token within each window,
        then sums across all query tokens to get the final window score.

        Args:
            sim_matrix: Interaction matrix of shape [N_q, N_d]
            doc_length: Number of document tokens (N_d)

        Returns:
            WindowScoreResult containing segment scores and window parameters,
            or None if the document is too short for windowing.
        """
        # Clamp window size to doc length
        window_size = min(self.window_size, doc_length)
        stride = max(1, self.window_size - self.window_overlap)

        if window_size < 1:
            return None

        # Add batch dimension for pooling: [1, N_q, N_d]
        sim_batched = sim_matrix.unsqueeze(0)

        # max_pool1d operates on last dimension
        # Input: [1, N_q, N_d], treats N_q as channels
        # Output: [1, N_q, N_windows]
        window_max_scores = F.max_pool1d(
            sim_batched,
            kernel_size=window_size,
            stride=stride
        )

        # Sum over query dimension to get window scores: [1, N_windows]
        segment_scores = window_max_scores.sum(dim=1)

        # Remove batch dimension: [N_windows]
        segment_scores = segment_scores.squeeze(0)
        n_windows = segment_scores.shape[0]

        return WindowScoreResult(
            segment_scores=segment_scores,
            window_size=window_size,
            stride=stride,
            n_windows=n_windows
        )

    def _resolve_spans(
        self,
        text_b: str,
        top_indices: torch.Tensor,
        top_scores: torch.Tensor,
        window_size: int,
        stride: int,
        n_doc_tokens: int
    ) -> Tuple[List[AttributionSpan], List[int], List[tuple], List[int]]:
        """Map window indices to character-level AttributionSpan objects.

        Handles tokenization alignment, special token filtering, and character
        position mapping.

        Args:
            text_b: Target text (document)
            top_indices: Tensor of top-k window indices
            top_scores: Tensor of top-k window scores
            window_size: Size of each window in tokens
            stride: Stride between windows
            n_doc_tokens: Total number of document tokens from ColBERT encoding

        Returns:
            Tuple of (spans, input_ids, offsets, special_mask) where:
            - spans: List of AttributionSpan objects
            - input_ids: Token IDs for text_b
            - offsets: Character offset tuples for each token
            - special_mask: Binary mask (1 for special tokens)
        """
        # Tokenize text_b with same config as model.encode() for alignment
        encoding_b = self.tokenizer(
            text_b,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.passage_max_length
        )
        input_ids_b = encoding_b["input_ids"]
        offsets_b = encoding_b["offset_mapping"]

        # Build special token mask
        special_ids = set(self.tokenizer.all_special_ids)
        special_mask_b = [1 if tid in special_ids else 0 for tid in input_ids_b]

        # Align lengths: model.encode may drop padding/SEP, take min
        min_len = min(n_doc_tokens, len(input_ids_b))
        offsets_b = offsets_b[:min_len]
        special_mask_b = special_mask_b[:min_len]

        if min_len == 0:
            return [], input_ids_b, offsets_b, special_mask_b

        # Build spans
        spans = []
        k = len(top_scores)
        max_score = top_scores[0].item() if k > 0 else 1.0

        for rank, (score, window_idx) in enumerate(
            zip(top_scores.tolist(), top_indices.tolist())
        ):
            # Calculate token range for this window
            token_start = window_idx * stride
            token_end = min(token_start + window_size, min_len)

            # Get character positions, skipping leading special tokens
            if token_start >= min_len:
                continue

            # Skip leading special tokens in window
            valid_start = token_start
            while valid_start < token_end and special_mask_b[valid_start] == 1:
                valid_start += 1
            if valid_start >= token_end:
                continue  # Window is all special tokens

            # Skip trailing special tokens in window
            valid_end = token_end - 1
            while valid_end > valid_start and special_mask_b[valid_end] == 1:
                valid_end -= 1

            char_start = offsets_b[valid_start][0]
            char_end = offsets_b[valid_end][1]

            if char_start >= char_end:
                continue

            # Extract span text
            span_text = text_b[char_start:char_end]

            # Normalize score
            normalized = normalize_score(score, max_score)

            span = AttributionSpan(
                text=span_text,
                start_idx=char_start,
                end_idx=char_end,
                score=normalized,
                metadata={
                    "raw_score": score,
                    "window_idx": window_idx,
                    "token_start": token_start,
                    "token_end": token_end,
                    "window_rank": rank + 1,
                }
            )
            spans.append(span)

        return spans, input_ids_b[:min_len], offsets_b, special_mask_b

    def extract(self, text_a: str, text_b: str) -> AttributionResult:
        """Extract attribution spans from text_b based on similarity to text_a.

        Uses ColBERT-style late interaction with sliding windows to find
        contiguous text spans with highest MaxSim scores.

        Args:
            text_a: Source text (query/reference)
            text_b: Target text (document to be analyzed)

        Returns:
            AttributionResult with scored spans from text_b

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If attribution extraction fails
        """
        # Validate inputs
        if not text_a or not text_a.strip():
            raise ValueError("text_a cannot be empty")
        if not text_b or not text_b.strip():
            raise ValueError("text_b cannot be empty")

        try:
            # Step 1: Encode both texts to get ColBERT vectors
            colbert_vecs = self._encode_colbert([text_a, text_b])
            query_vecs = colbert_vecs[0]  # [N_q, D]
            doc_vecs = colbert_vecs[1]    # [N_d, D]

            # Convert to tensors if they are numpy arrays
            if not isinstance(query_vecs, torch.Tensor):
                query_vecs = torch.tensor(query_vecs)
            if not isinstance(doc_vecs, torch.Tensor):
                doc_vecs = torch.tensor(doc_vecs)

            n_q, dim = query_vecs.shape
            n_d, _ = doc_vecs.shape

            logger.debug(f"Query vectors: {n_q} tokens, Doc vectors: {n_d} tokens, dim={dim}")

            # Step 2: Compute interaction matrix (includes normalization)
            sim_matrix = self._compute_interaction_matrix(query_vecs, doc_vecs)

            # Step 3: Compute window scores
            window_result = self._compute_window_scores(sim_matrix, n_d)

            if window_result is None:
                logger.warning("Document too short for windowing")
                return self._empty_result(text_a, text_b)

            logger.debug(f"Computed {window_result.n_windows} window scores")

            # Step 4: Find top-k windows
            k = min(self.top_k_spans, window_result.n_windows)
            top_scores, top_indices = torch.topk(window_result.segment_scores, k)

            # Step 5: Resolve spans (map window indices to character positions)
            spans, input_ids_b, offsets_b, special_mask_b = self._resolve_spans(
                text_b,
                top_indices,
                top_scores,
                window_result.window_size,
                window_result.stride,
                n_d
            )

            if not input_ids_b:
                logger.warning("No tokens found in text_b")
                return self._empty_result(text_a, text_b)

            # Step 6: Compute overall MaxSim score for metadata
            # Standard ColBERT score: mean of max similarities per query token
            max_sims_per_query = sim_matrix.max(dim=-1).values  # [N_q]
            colbert_score = max_sims_per_query.mean().item()

            # Step 7: Extract topic keywords (global summary-style tokens)
            topic_keywords = self._extract_topic_keywords(
                sim_matrix,
                input_ids_b,
                offsets_b,
                special_mask_b,
                text_b,
                top_k=self.top_k_topic_keywords
            )

            # Build result
            result = AttributionResult(
                text_a=text_a,
                text_b=text_b,
                method_name=self.name,
                spans=spans,
                metadata={
                    "colbert_score": colbert_score,
                    "num_query_tokens": n_q,
                    "num_doc_tokens": n_d,
                    "window_size": window_result.window_size,
                    "window_overlap": self.window_overlap,
                    "stride": window_result.stride,
                    "total_windows": window_result.n_windows,
                    "topic_keywords": topic_keywords,
                }
            )

            logger.info(
                f"ColBERT attribution complete: {len(spans)} spans, "
                f"colbert_score={colbert_score:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"ColBERT attribution extraction failed: {e}")
            raise RuntimeError(f"Failed to extract attribution: {e}") from e

    def _extract_topic_keywords(
        self,
        sim_matrix: torch.Tensor,
        input_ids: List[int],
        offsets: List[tuple],
        special_mask: List[int],
        text_b: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Extract global topic keywords from the interaction matrix.

        Args:
            sim_matrix: Interaction matrix [N_q, N_d] or [1, N_q, N_d]
            input_ids: Token IDs for text_b
            offsets: Character offset tuples (start, end) for each token
            special_mask: 1 for special tokens, 0 for normal tokens
            text_b: Original target text
            top_k: Number of keywords to return

        Returns:
            List of keyword dictionaries (text, score, start_idx, end_idx)
        """
        if sim_matrix.dim() == 3:
            sim_matrix = sim_matrix.squeeze(0)
        if sim_matrix.dim() != 2 or sim_matrix.size(1) == 0:
            return []

        # Aggregate across query tokens to capture tokens broadly related to the query.
        aggregated = torch.relu(sim_matrix).mean(dim=0)  # [N_d]

        if torch.all(aggregated == 0):
            # Fallback to max if everything is zero (e.g., all negatives)
            aggregated = torch.relu(sim_matrix).max(dim=0).values

        num_tokens = aggregated.shape[0]
        if num_tokens == 0:
            return []

        # Filter out special tokens, punctuation and short tokens before selecting top-k
        import string
        valid_indices = []
        for idx in range(min(num_tokens, len(input_ids))):
            # Skip special tokens
            if special_mask[idx] == 1:
                continue
            # Get token text via offset
            start, end = offsets[idx]
            if start >= end:
                continue
            token_text = text_b[start:end].strip()
            # Filter if empty, just punctuation, or length < 2 (unless it's CJK)
            if not token_text:
                continue
            if all(char in string.punctuation for char in token_text):
                continue
            if len(token_text) < 2 and not any(ord(c) > 127 for c in token_text):
                continue
            valid_indices.append(idx)
        
        if not valid_indices:
            return []
            
        # Create a mask to zero out invalid tokens
        mask = torch.zeros_like(aggregated, dtype=torch.bool)
        mask[valid_indices] = True
        aggregated = aggregated * mask

        k = min(max(top_k, 1), len(valid_indices))
        top_scores, top_indices = torch.topk(aggregated, k=k)
        
        # Filter out zero scores that might come from masking if k > valid_count
        nonzero_mask = top_scores > 0
        top_scores = top_scores[nonzero_mask]
        top_indices = top_indices[nonzero_mask]
        
        if top_indices.numel() == 0:
            return []

        # Prepare mapping for quick score lookup
        score_map = {idx: score for idx, score in zip(top_indices.tolist(), top_scores.tolist())}

        # Group contiguous token indices to produce readable phrases
        sorted_indices = sorted(score_map.keys())
        groups: List[List[int]] = []
        if sorted_indices:
            current_group = [sorted_indices[0]]
            for idx in sorted_indices[1:]:
                if idx == current_group[-1] + 1:
                    current_group.append(idx)
                else:
                    groups.append(current_group)
                    current_group = [idx]
            groups.append(current_group)

        # Score each group using the sum of token scores
        keyword_entries = []
        for group in groups:
            start_token = group[0]
            end_token = group[-1]
            if start_token >= len(offsets) or end_token >= len(offsets):
                continue

            start_idx = offsets[start_token][0]
            end_idx = offsets[end_token][1]
            if start_idx >= end_idx:
                continue

            # Double check final text isn't just punctuation
            text_span = text_b[start_idx:end_idx].strip()
            if not text_span or all(char in string.punctuation for char in text_span):
                continue

            raw_score = sum(score_map.get(idx, 0.0) for idx in group)
            
            keyword_entries.append({
                "text": text_span,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "score": raw_score,
            })

        if not keyword_entries:
            return []

        # Keep only top_k phrases by raw score
        keyword_entries.sort(key=lambda x: x["score"], reverse=True)
        keyword_entries = keyword_entries[:top_k]

        # Normalize scores for readability
        max_score = keyword_entries[0]["score"] if keyword_entries else 1.0
        for entry in keyword_entries:
            entry["score"] = normalize_score(entry["score"], max_score)

        return keyword_entries

    def _empty_result(self, text_a: str, text_b: str) -> AttributionResult:
        """Create an empty attribution result.

        Args:
            text_a: Source text
            text_b: Target text

        Returns:
            Empty AttributionResult
        """
        return AttributionResult(
            text_a=text_a,
            text_b=text_b,
            method_name=self.name,
            spans=[],
            metadata={
                "colbert_score": 0.0,
                "num_query_tokens": 0,
                "num_doc_tokens": 0,
                "topic_keywords": [],
            }
        )

    def compute_colbert_score(self, text_a: str, text_b: str) -> float:
        """Compute the ColBERT MaxSim score between two texts.

        This computes the standard ColBERT similarity:
            score = (1/N_q) * sum_{i=1}^{N_q} max_{j=1}^{N_d} E_q[i] · E_d[j]

        Args:
            text_a: First text (query)
            text_b: Second text (document)

        Returns:
            ColBERT MaxSim score
        """
        colbert_vecs = self._encode_colbert([text_a, text_b])
        query_vecs = colbert_vecs[0]
        doc_vecs = colbert_vecs[1]

        if not isinstance(query_vecs, torch.Tensor):
            query_vecs = torch.tensor(query_vecs)
        if not isinstance(doc_vecs, torch.Tensor):
            doc_vecs = torch.tensor(doc_vecs)

        # Reuse the interaction matrix computation
        sim_matrix = self._compute_interaction_matrix(query_vecs, doc_vecs)
        max_sims = sim_matrix.max(dim=-1).values
        return max_sims.mean().item()
