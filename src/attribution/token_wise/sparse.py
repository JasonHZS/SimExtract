"""Sparse embedding attribution method using BGE-M3 lexical weights.
参考：https://bge-model.com/API/inference/embedder/encoder_only/M3Embedder.html"""

import logging
import re
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Dict, Any, List, Optional, Tuple

from ..base import AttributionMethod, AttributionResult, AttributionSpan
from .utils import normalize_score

logger = logging.getLogger(__name__)

_BGE_M3_MODEL_INIT_LOCK = Lock()


@lru_cache(maxsize=8)
def _get_cached_bge_m3_model(
    model_name: str,
    use_fp16: bool,
    device: Optional[str],
    query_max_length: int = 512,
    passage_max_length: int = 512
):
    """Process-local singleton cache for BGEM3FlagModel.

    Important:
    - This cache only deduplicates model loads **within the same Python process**.
    - If you run a script multiple times, each run is a new process → it will still reload.
    - In production (e.g., FastAPI/uvicorn), keep a long-lived process and reuse the cached model.
    """
    from FlagEmbedding import BGEM3FlagModel

    kwargs = {
        "use_fp16": use_fp16,
        "query_max_length": query_max_length,
        "passage_max_length": passage_max_length
    }
    if device is not None:
        kwargs["device"] = device

    # Guard initialization because model loading touches GPU / files and can be expensive.
    with _BGE_M3_MODEL_INIT_LOCK:
        return BGEM3FlagModel(model_name, **kwargs)


class SparseAttribution(AttributionMethod):
    """Similarity attribution via BGE-M3 sparse (lexical) embeddings.

    This method uses the lexical_weights from BGE-M3 to compute token-level
    contribution scores. The sparse representation assigns a weight to each
    token based on its importance in the text.

    The relevance score between two texts is computed as:
        s_lex = sum_{t in A ∩ B} (w_at * w_bt)

    where w_at and w_bt are the lexical weights of token t in text A and B.

    Two modes of attribution:
    1. Top-N tokens: Extract the tokens with highest contribution scores
    2. Sliding window: Find contiguous spans with highest average contribution
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize SparseAttribution method.

        Args:
            config: Configuration dictionary containing:
                - model_name: str, BGE-M3 model name or path (default: "BAAI/bge-m3")
                  Can be either a HuggingFace model id or a local directory path
                - use_fp16: bool, whether to use half precision (default: True)
                - device: str, device to use (default: auto-detect)
                - window_size: int, sliding window token count (default: 30)
                - window_overlap: int, token overlap between windows (default: 20)
                - top_k_spans: int, number of top spans to return (default: 5)

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
        self.window_overlap = config.get("window_overlap", 40)
        self.top_k_spans = config.get("top_k_spans", 3)
        self.top_k_tokens = config.get("top_k_tokens", 5)

        # Validate window parameters
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.window_overlap < 0:
            raise ValueError("window_overlap must be non-negative")
        if self.window_overlap >= self.window_size:
            raise ValueError("window_overlap must be less than window_size")

        # Initialize the model
        self._model = None
        self._initialize_model()

        logger.info(
            f"SparseAttribution initialized: model={self.model_name}, "
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
            # when SparseAttribution is constructed multiple times (e.g., in a web service).
            self._model = _get_cached_bge_m3_model(
                self.model_name,
                self.use_fp16,
                self.device,
                query_max_length=self.query_max_length,
                passage_max_length=self.passage_max_length
            )
            logger.info(f"BGE-M3 model loaded: {self.model_name}")

        except ImportError as e:
            raise RuntimeError(
                "FlagEmbedding is required for SparseAttribution. "
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

    def _encode_sparse(self, texts: List[str]) -> List[Dict[int, float]]:
        """Encode texts to sparse lexical weights.

        Args:
            texts: List of texts to encode

        Returns:
            List of dictionaries mapping token_id -> weight
        """
        output = self.model.encode(
            texts,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False
        )
        return output['lexical_weights']

    def _get_token_contributions(
        self,
        text_a: str,
        text_b: str
    ) -> Tuple[Dict[str, float], float]:
        """Compute per-token contribution scores.

        Args:
            text_a: Source text (reference)
            text_b: Target text (to be analyzed)

        Returns:
            Tuple of (token_contributions, total_score):
                - token_contributions: dict mapping token -> contribution score
                - total_score: sum of all contributions (lexical matching score)
        """
        # Encode both texts
        weights_a, weights_b = self._encode_sparse([text_a, text_b])

        # NOTE:
        # FlagEmbedding(BGE-M3) returns `lexical_weights` as dict[token_id -> weight].

        def _token_key(token_id: int) -> str:
            # clean_up_tokenization_spaces=False: Disable automatic formatting (such as merging punctuation spaces), 
            # ensure the original token is obtained for precise matching.
            token_str = self.tokenizer.decode(
                [int(token_id)],
                clean_up_tokenization_spaces=False,
            )
            return token_str.strip()

        # Compute per-token contributions (intersection), aggregated by token string
        contributions: Dict[str, float] = {}
        for token_id_b, weight_b in weights_b.items():
            if token_id_b not in weights_a:
                continue
            token = _token_key(token_id_b)
            if not token:
                continue
            contributions[token] = contributions.get(token, 0.0) + (weights_a[token_id_b] * weight_b)

        # Compute total score
        total_score = sum(contributions.values()) if contributions else 0.0

        logger.debug(
            f"Token contributions: {len(contributions)} tokens, "
            f"total_score={total_score:.4f}"
        )

        return contributions, total_score

    def get_top_contributing_tokens(
        self,
        text_a: str,
        text_b: str,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get the top-N tokens with highest contribution scores.

        This method identifies which tokens in text_b contribute most to
        its similarity with text_a, based on lexical weight intersection.

        Args:
            text_a: Source text (reference)
            text_b: Target text (to be analyzed)
            top_n: Number of top tokens to return (default: 10)

        Returns:
            List of dictionaries with keys:
                - token: str, the token text
                - score: float, the contribution score (w_a * w_b)
                - normalized_score: float, score / total_score (0-1)

        Example:
            >>> attr = SparseAttribution({"model_name": "BAAI/bge-m3"})
            >>> tokens = attr.get_top_contributing_tokens(
            ...     "AI is transforming healthcare",
            ...     "Machine learning for medical diagnosis"
            ... )
            >>> print(tokens[0])
            {'token': 'medical', 'score': 0.85, 'normalized_score': 0.32}
        """
        contributions, total_score = self._get_token_contributions(text_a, text_b)

        if not contributions:
            logger.warning("No common tokens found between text_a and text_b")
            return []

        # Sort by score descending
        sorted_tokens = sorted(
            contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        # Build result list
        results = []
        for token, score in sorted_tokens:
            normalized = score / total_score if total_score > 0 else 0.0
            results.append({
                "token": token,
                "score": score,
                "normalized_score": normalized
            })

        return results

    def _tokenize_with_positions(self, text: str) -> List[Dict[str, Any]]:
        """Tokenize text and get character positions for each token.

        Args:
            text: Text to tokenize

        Returns:
            List of token info dictionaries with keys:
                - token_id: int, the token ID
                - token: str, the token string
                - start: int, character start position
                - end: int, character end position
                - index: int, position in the token sequence
        """
        # Encode with offset mapping
        # For Chinese text, each character typically gets its own token
        # with (start, end) representing single-character spans, e.g.:
        # "机器学习" -> [(0,1), (1,2), (2,3), (3,4)]
        # For English, tokens may span multiple characters:
        # "learning" -> [(0,8)]
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )

        tokens = []
        for idx, (token_id, (start, end)) in enumerate(
            zip(encoding['input_ids'], encoding['offset_mapping'])
        ):
            # Skip special tokens (e.g., [CLS], [SEP], [PAD]) or empty spans.
            # Empty spans occur when start == end, indicating no actual text content.
            # Special tokens are typically added by tokenizers for model input formatting
            # but don't correspond to meaningful text positions in the original string.
            if start == end:
                continue

            token_str = self.tokenizer.decode([token_id])
            tokens.append({
                "token_id": token_id,
                "token": token_str,
                "start": start,
                "end": end,
                "index": idx
            })

        return tokens

    def _compute_window_scores(
        self,
        text_b: str,
        contributions: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Compute scores for sliding windows over text_b.

        Args:
            text_b: Target text to analyze
            contributions: Token contribution scores from _get_token_contributions

        Returns:
            List of window info dictionaries with keys:
                - start_idx: int, character start position
                - end_idx: int, character end position
                - text: str, window text content
                - score: float, average contribution score
                - token_count: int, number of tokens in window
                - contributing_tokens: int, tokens with non-zero contribution
        """
        # Tokenize text_b with positions
        tokens = self._tokenize_with_positions(text_b)

        if not tokens:
            logger.warning("No tokens found in text_b")
            return []

        # Build token-to-contribution mapping
        # Note: BGE tokenizer may produce subwords, we need to match them
        # Map each token to its contribution score from the sparse lexical weights.
        # This creates a parallel list where token_scores[i] corresponds to tokens[i].
        token_scores = []
        for t in tokens:
            token_text = t["token"].strip()
            # Try exact match first
            score = contributions.get(token_text, 0.0)
            # Also try lowercase to handle case mismatches
            if score == 0.0:
                score = contributions.get(token_text.lower(), 0.0)
            token_scores.append(score)

        # Sliding window computation
        windows = []
        num_tokens = len(tokens)
        step_size = self.window_size - self.window_overlap
        if step_size < 1:
            step_size = 1

        window_start = 0
        while window_start < num_tokens:
            window_end = min(window_start + self.window_size, num_tokens)

            # Get window tokens and scores
            window_tokens = tokens[window_start:window_end]
            window_scores = token_scores[window_start:window_end]

            # Compute statistics
            contributing = [s for s in window_scores if s > 0]
            if contributing:
                avg_score = sum(contributing) / len(contributing)
            else:
                avg_score = 0.0

            # Get character positions
            char_start = window_tokens[0]["start"]
            char_end = window_tokens[-1]["end"]

            windows.append({
                "start_idx": char_start,
                "end_idx": char_end,
                "text": text_b[char_start:char_end],
                "score": avg_score,
                "token_count": len(window_tokens),
                "contributing_tokens": len(contributing),
                "window_token_start": window_start,
                "window_token_end": window_end
            })

            # Move window
            window_start += step_size

            # If we've covered all tokens, break
            if window_end >= num_tokens:
                break

        return windows

    def _normalize_score(self, score: float, max_score: float) -> float:
        """Normalize score to [0, 1] range.

        Args:
            score: Raw score to normalize
            max_score: Maximum score for normalization

        Returns:
            Normalized score in [0, 1]
        """
        return normalize_score(score, max_score)

    def extract(self, text_a: str, text_b: str) -> AttributionResult:
        """Extract attribution spans from text_b based on similarity to text_a.

        Uses sliding window approach to find contiguous text spans with
        highest average token contribution scores.

        Args:
            text_a: Source text (reference)
            text_b: Target text (to be analyzed)

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
            # Step 1: Compute token contributions
            contributions, total_score = self._get_token_contributions(text_a, text_b)

            logger.info(
                f"Computing attribution: {len(contributions)} contributing tokens, "
                f"total_score={total_score:.4f}"
            )

            # Step 2: Compute window scores
            windows = self._compute_window_scores(text_b, contributions)

            if not windows:
                logger.warning("No windows generated, returning empty result")
                return AttributionResult(
                    text_a=text_a,
                    text_b=text_b,
                    method_name=self.name,
                    spans=[],
                    metadata={
                        "total_lexical_score": total_score,
                        "num_contributing_tokens": len(contributions),
                    }
                )

            # Step 3: Sort windows by score and take top-k
            windows.sort(key=lambda x: x["score"], reverse=True)
            top_windows = windows[:self.top_k_spans]

            # Find max score for normalization
            max_score = top_windows[0]["score"] if top_windows else 1.0

            # Step 4: Create AttributionSpans
            spans = []
            for idx, window in enumerate(top_windows):
                normalized_score = self._normalize_score(window["score"], max_score)

                span = AttributionSpan(
                    text=window["text"],
                    start_idx=window["start_idx"],
                    end_idx=window["end_idx"],
                    score=normalized_score,
                    metadata={
                        "raw_score": window["score"],
                        "token_count": window["token_count"],
                        "contributing_tokens": window["contributing_tokens"],
                        "window_rank": idx + 1,
                    }
                )
                spans.append(span)

            # Step 5: Get top contributing tokens for metadata
            top_tokens = self.get_top_contributing_tokens(text_a, text_b, top_n=10)

            # Step 6: Create result
            result = AttributionResult(
                text_a=text_a,
                text_b=text_b,
                method_name=self.name,
                spans=spans,
                metadata={
                    "total_lexical_score": total_score,
                    "num_contributing_tokens": len(contributions),
                    "top_contributing_tokens": top_tokens,
                    "window_size": self.window_size,
                    "window_overlap": self.window_overlap,
                    "total_windows_analyzed": len(windows),
                }
            )

            logger.info(
                f"Attribution complete: {len(spans)} spans, "
                f"best_score={spans[0].score:.4f}" if spans else "no spans"
            )

            return result

        except Exception as e:
            logger.error(f"Attribution extraction failed: {e}")
            raise RuntimeError(f"Failed to extract attribution: {e}") from e

    def compute_lexical_score(self, text_a: str, text_b: str) -> float:
        """Compute the total lexical matching score between two texts.

        This is a convenience method that returns the same score as
        model.compute_lexical_matching_score().

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            Lexical matching score (sum of w_a * w_b for common tokens)
        """
        _, total_score = self._get_token_contributions(text_a, text_b)
        return total_score
