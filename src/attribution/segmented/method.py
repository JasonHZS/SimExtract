"""Segmented vectorization attribution method implementation."""

import re
import math
import logging
from typing import Dict, Any, List, Tuple, Optional

from ..base import AttributionMethod, AttributionResult, AttributionSpan
from ...data_pipeline.vectorizers.tei_vectorizer import TEIVectorizer


logger = logging.getLogger(__name__)


class SegmentedAttribution(AttributionMethod):
    """Similarity attribution via segmented vectorization.

    This method splits text_b into segments, vectorizes each segment,
    and computes similarity with text_a to identify contributing spans.

    Supports two segmentation strategies:
    - fixed_length: Chunks by token count (智能识别中英文token)
    - fixed_sentences: Groups consecutive sentences together

    Token definition:
    - Chinese characters (CJK): Each character is a token
    - English words: Each word is a token
    - Numbers: Each number sequence is a token
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize SegmentedAttribution method.

        Args:
            config: Configuration dictionary containing:
                - segmentation_method: "fixed_length" or "fixed_sentences"
                - chunk_size: int (default 30) - number of tokens per chunk
                - chunk_overlap: int (default 10) - overlap tokens
                - num_sentences: int (default 3) - sentences per segment
                - vectorizer: Pre-initialized TEIVectorizer instance, OR
                - vectorizer_config: dict for TEIVectorizer initialization

        Raises:
            ValueError: If config is invalid
            RuntimeError: If vectorizer cannot be initialized
        """
        super().__init__(config)

        # Extract configuration parameters
        self.segmentation_method = config.get("segmentation_method", "fixed_sentences")
        self.chunk_size = config.get("chunk_size", 30)
        self.chunk_overlap = config.get("chunk_overlap", 10)
        self.num_sentences = config.get("num_sentences", 3)

        # Validate configuration
        self._validate_config()

        # Initialize vectorizer
        self.vectorizer = self._initialize_vectorizer(config)

        logger.info(
            f"SegmentedAttribution initialized: "
            f"method={self.segmentation_method}, "
            f"chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, "
            f"num_sentences={self.num_sentences}"
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Validate segmentation method
        valid_methods = ["fixed_length", "fixed_sentences"]
        if self.segmentation_method not in valid_methods:
            raise ValueError(
                f"Invalid segmentation_method: {self.segmentation_method}. "
                f"Must be one of {valid_methods}"
            )

        # Validate fixed_length parameters
        if self.segmentation_method == "fixed_length":
            if self.chunk_size <= 0:
                raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
            if self.chunk_overlap < 0:
                raise ValueError(
                    f"chunk_overlap must be non-negative, got {self.chunk_overlap}"
                )
            if self.chunk_overlap >= self.chunk_size:
                raise ValueError(
                    f"chunk_overlap ({self.chunk_overlap}) must be less than "
                    f"chunk_size ({self.chunk_size})"
                )

        # Validate fixed_sentences parameters
        if self.segmentation_method == "fixed_sentences":
            if not 1 <= self.num_sentences <= 10:
                raise ValueError(
                    f"num_sentences must be between 1 and 10, got {self.num_sentences}"
                )
            if self.num_sentences < 3:
                logger.warning(
                    f"num_sentences={self.num_sentences} is low, may result in "
                    f"very granular segments"
                )

    def _initialize_vectorizer(self, config: Dict[str, Any]):
        """Initialize vectorizer from config.

        Args:
            config: Configuration dictionary

        Returns:
            Vectorizer instance with embed(), get_dimension(), and health_check() methods

        Raises:
            RuntimeError: If vectorizer initialization fails
        """
        # Check if vectorizer is already provided
        if "vectorizer" in config:
            vectorizer = config["vectorizer"]
            # Duck typing: check if it has the required methods
            required_methods = ["embed", "get_dimension", "health_check"]
            for method in required_methods:
                if not hasattr(vectorizer, method):
                    raise ValueError(
                        f"Provided vectorizer must have {method}() method"
                    )
            logger.info("Using pre-initialized vectorizer")
            return vectorizer

        # Initialize from vectorizer_config
        vectorizer_config = config.get("vectorizer_config", {})
        try:
            vectorizer = TEIVectorizer(
                api_url=vectorizer_config.get("api_url", "http://localhost:8080/embed"),
                batch_size=vectorizer_config.get("batch_size", 64),
                max_retries=vectorizer_config.get("max_retries", 3),
                timeout=vectorizer_config.get("timeout", 60),
                dimension=vectorizer_config.get("dimension", 1024)
            )

            # Perform health check
            if not vectorizer.health_check():
                raise RuntimeError("TEI vectorizer health check failed")

            logger.info("TEI vectorizer initialized successfully")
            return vectorizer

        except Exception as e:
            raise RuntimeError(f"Failed to initialize vectorizer: {e}")

    def extract(self, text_a: str, text_b: str) -> AttributionResult:
        """Extract attribution from text_b to text_a.

        Args:
            text_a: Source text (reference)
            text_b: Target text (to be analyzed and segmented)

        Returns:
            AttributionResult with scored spans from text_b

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If attribution extraction fails
        """
        # Validate inputs
        self._validate_inputs(text_a, text_b)

        try:
            # Step 1: Vectorize source text
            logger.debug(f"Vectorizing source text ({len(text_a)} chars)")
            source_embedding = self.vectorizer.embed([text_a])[0]

            # Step 2: Segment target text
            logger.debug(
                f"Segmenting target text ({len(text_b)} chars) "
                f"using {self.segmentation_method}"
            )
            segments = self._segment_text(text_b)

            if not segments:
                raise ValueError("Text segmentation resulted in no segments")

            logger.info(f"Created {len(segments)} segments from target text")

            # Step 3: Vectorize all segments in batch
            segment_texts = [seg["text"] for seg in segments]
            logger.debug(f"Vectorizing {len(segment_texts)} segments")
            segment_embeddings = self.vectorizer.embed(segment_texts)

            # Step 4: Compute similarity scores
            logger.debug("Computing similarity scores")
            similarity_scores = [
                self._cosine_similarity(source_embedding, seg_emb)
                for seg_emb in segment_embeddings
            ]

            # Step 5: Create attribution spans
            spans = []
            for idx, (segment, score) in enumerate(zip(segments, similarity_scores)):
                span = AttributionSpan(
                    text=segment["text"],
                    start_idx=segment["start_idx"],
                    end_idx=segment["end_idx"],
                    score=score,
                    metadata={
                        "segment_index": idx,
                        "total_segments": len(segments),
                        "segmentation_method": self.segmentation_method,
                        "segment_length": len(segment["text"]),
                    }
                )
                spans.append(span)

            # Step 6: Sort spans by score descending
            spans.sort(key=lambda x: x.score, reverse=True)

            # Step 7: Log best segment info
            best_score = max(similarity_scores) if similarity_scores else 0.0
            logger.info(
                f"Best segment score: {best_score:.4f}, "
                f"segment text: '{spans[0].text[:50]}...'"
            )

            # Step 8: Create and return result
            result = AttributionResult(
                text_a=text_a,
                text_b=text_b,
                method_name=self.name,
                spans=spans,
                metadata={
                    "segmentation_method": self.segmentation_method,
                    "num_segments": len(segments),
                    "chunk_size": self.chunk_size if self.segmentation_method == "fixed_length" else None,
                    "chunk_overlap": self.chunk_overlap if self.segmentation_method == "fixed_length" else None,
                    "num_sentences": self.num_sentences if self.segmentation_method == "fixed_sentences" else None,
                    "all_scores": similarity_scores,
                    "vectorizer_dimension": self.vectorizer.get_dimension(),
                }
            )

            return result

        except Exception as e:
            logger.error(f"Attribution extraction failed: {e}")
            raise RuntimeError(f"Failed to extract attribution: {e}")

    def batch_extract(
        self,
        pairs: List[Tuple[str, str]]
    ) -> List[AttributionResult]:
        """Extract attribution for multiple text pairs efficiently.

        This optimized implementation vectorizes all source texts in one batch.

        Args:
            pairs: List of (text_a, text_b) tuples

        Returns:
            List of AttributionResults

        Raises:
            RuntimeError: If batch extraction fails
        """
        if not pairs:
            return []

        logger.info(f"Batch processing {len(pairs)} text pairs")

        try:
            # Vectorize all source texts in one batch
            source_texts = [pair[0] for pair in pairs]
            logger.debug(f"Batch vectorizing {len(source_texts)} source texts")
            source_embeddings = self.vectorizer.embed(source_texts)

            # Process each pair with pre-computed embedding
            results = []
            for idx, ((text_a, text_b), source_emb) in enumerate(
                zip(pairs, source_embeddings)
            ):
                # Validate inputs
                self._validate_inputs(text_a, text_b)

                # Segment target text
                segments = self._segment_text(text_b)
                if not segments:
                    raise ValueError(f"No segments for pair {idx}")

                # Vectorize segments
                segment_texts = [seg["text"] for seg in segments]
                segment_embeddings = self.vectorizer.embed(segment_texts)

                # Compute similarities
                similarity_scores = [
                    self._cosine_similarity(source_emb, seg_emb)
                    for seg_emb in segment_embeddings
                ]

                # Create spans
                spans = []
                for seg_idx, (segment, score) in enumerate(zip(segments, similarity_scores)):
                    span = AttributionSpan(
                        text=segment["text"],
                        start_idx=segment["start_idx"],
                        end_idx=segment["end_idx"],
                        score=score,
                        metadata={
                            "segment_index": seg_idx,
                            "total_segments": len(segments),
                            "segmentation_method": self.segmentation_method,
                            "segment_length": len(segment["text"]),
                        }
                    )
                    spans.append(span)

                # Sort spans
                spans.sort(key=lambda x: x.score, reverse=True)

                # Create result
                result = AttributionResult(
                    text_a=text_a,
                    text_b=text_b,
                    method_name=self.name,
                    spans=spans,
                    metadata={
                        "segmentation_method": self.segmentation_method,
                        "num_segments": len(segments),
                        "all_scores": similarity_scores,
                    }
                )
                results.append(result)

                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(pairs)} pairs")

            logger.info(f"Batch processing complete: {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
            raise RuntimeError(f"Failed to batch extract attribution: {e}")

    def _validate_inputs(self, text_a: str, text_b: str) -> None:
        """Validate input parameters.

        Args:
            text_a: Source text
            text_b: Target text

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate text_a
        if not isinstance(text_a, str):
            raise ValueError("text_a must be a string")
        if not text_a or not text_a.strip():
            raise ValueError("text_a cannot be empty")

        # Validate text_b
        if not isinstance(text_b, str):
            raise ValueError("text_b must be a string")
        if not text_b or not text_b.strip():
            raise ValueError("text_b cannot be empty")

        # Warn for very short text
        if len(text_b) < 10:
            logger.warning(
                f"text_b is very short ({len(text_b)} chars), "
                "segmentation may not be meaningful"
            )

    def _segment_text(self, text: str) -> List[Dict[str, Any]]:
        """Segment text using configured strategy.

        Args:
            text: Text to segment

        Returns:
            List of segment dictionaries with keys:
            - text: segment text content
            - start_idx: character start index
            - end_idx: character end index

        Raises:
            ValueError: If segmentation method is unknown
        """
        if self.segmentation_method == "fixed_length":
            return self._segment_fixed_length(text)
        elif self.segmentation_method == "fixed_sentences":
            return self._segment_fixed_sentences(text)
        else:
            raise ValueError(
                f"Unknown segmentation method: {self.segmentation_method}"
            )

    def _segment_fixed_length(self, text: str) -> List[Dict[str, Any]]:
        """Segment text into fixed-length chunks with overlap.

        Uses smart tokenization that automatically handles Chinese and English:
        - Chinese characters (CJK): Each character is a token
        - English words: Each word is a token
        - Numbers: Each number sequence is a token

        Args:
            text: Text to segment

        Returns:
            List of segment dictionaries
        """
        # Tokenize text
        tokens = self._smart_tokenize(text)

        if not tokens:
            # No tokens found, return entire text
            return [{
                "text": text,
                "start_idx": 0,
                "end_idx": len(text),
            }]

        # If fewer tokens than chunk_size, return entire text
        if len(tokens) <= self.chunk_size:
            return [{
                "text": text,
                "start_idx": 0,
                "end_idx": len(text),
            }]

        # Create segments by token count
        segments = []
        step_size = self.chunk_size - self.chunk_overlap

        token_idx = 0
        while token_idx < len(tokens):
            # Get chunk of tokens
            end_token_idx = min(token_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[token_idx:end_token_idx]

            # Get character positions
            start_idx = chunk_tokens[0]["start"]
            end_idx = chunk_tokens[-1]["end"]

            # Extract text
            segment_text = text[start_idx:end_idx]

            if segment_text.strip():
                segments.append({
                    "text": segment_text,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                })

            # If we've processed all tokens, break
            if end_token_idx >= len(tokens):
                break

            # Move to next position
            token_idx += step_size

        logger.debug(
            f"Token-based segmentation: {len(segments)} segments "
            f"({len(tokens)} tokens, chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )

        return segments

    def _smart_tokenize(self, text: str) -> List[Dict[str, Any]]:
        """Smart tokenization for mixed Chinese/English text.

        Token definition:
        - CJK characters (Chinese/Japanese/Korean): Each character is one token
        - English letters: Consecutive letters form one token (word)
        - Numbers: Consecutive digits form one token
        - Other characters are skipped (whitespace, punctuation treated as separators)

        Args:
            text: Text to tokenize

        Returns:
            List of token dictionaries with keys:
            - text: token text
            - start: character start position
            - end: character end position
        """
        # Pattern matches:
        # 1. Single CJK character (Chinese: \u4e00-\u9fff, Japanese: \u3040-\u309f, \u30a0-\u30ff, Korean: \uac00-\ud7af)
        # 2. Consecutive English letters
        # 3. Consecutive digits
        pattern = r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]|[a-zA-Z]+|\d+'

        tokens = []
        for match in re.finditer(pattern, text):
            tokens.append({
                "text": match.group(),
                "start": match.start(),
                "end": match.end()
            })

        logger.debug(f"Tokenized into {len(tokens)} tokens")
        return tokens

    def _segment_fixed_sentences(self, text: str) -> List[Dict[str, Any]]:
        """Segment text into groups of sentences.

        Args:
            text: Text to segment

        Returns:
            List of segment dictionaries
        """
        # Split into sentences
        sentences = self._split_sentences(text)

        if not sentences:
            # No sentences found, return entire text
            return [{
                "text": text,
                "start_idx": 0,
                "end_idx": len(text),
            }]

        # If fewer sentences than num_sentences, return all as one segment
        if len(sentences) <= self.num_sentences:
            return [{
                "text": text,
                "start_idx": 0,
                "end_idx": len(text),
            }]

        # Group sentences into segments
        segments = []
        for i in range(0, len(sentences), self.num_sentences):
            # Get group of sentences
            sentence_group = sentences[i:i + self.num_sentences]

            # Calculate indices
            start_idx = sentence_group[0]["start_idx"]
            end_idx = sentence_group[-1]["end_idx"]

            # Extract text
            segment_text = text[start_idx:end_idx]

            if segment_text.strip():
                segments.append({
                    "text": segment_text,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                })

        logger.debug(
            f"Sentence-based segmentation: {len(segments)} segments "
            f"({len(sentences)} sentences, {self.num_sentences} sentences/segment)"
        )

        return segments

    def _split_sentences(self, text: str) -> List[Dict[str, Any]]:
        """Split text into sentences with position tracking.

        Handles both Chinese (。！？；) and English (. ! ? ;) sentence delimiters.

        Args:
            text: Text to split

        Returns:
            List of sentence dictionaries with keys:
            - text: sentence text
            - start_idx: character start index
            - end_idx: character end index
        """
        # Regex pattern for sentence boundaries
        # Matches one or more punctuation marks followed by whitespace or end of string
        sentence_pattern = r'[.!?;。!?;]+(?:\s|$)'

        sentences = []
        last_end = 0

        # Find all sentence delimiters
        for match in re.finditer(sentence_pattern, text):
            end_pos = match.end()
            sentence_text = text[last_end:end_pos].strip()

            if sentence_text:  # Only add non-empty sentences
                sentences.append({
                    "text": sentence_text,
                    "start_idx": last_end,
                    "end_idx": end_pos,
                })

            last_end = end_pos

        # Handle remaining text (if no delimiter at end)
        if last_end < len(text):
            remaining_text = text[last_end:].strip()
            if remaining_text:
                sentences.append({
                    "text": remaining_text,
                    "start_idx": last_end,
                    "end_idx": len(text),
                })

        # Fallback: if no sentences found, treat entire text as one sentence
        if not sentences:
            sentences.append({
                "text": text.strip(),
                "start_idx": 0,
                "end_idx": len(text),
            })

        logger.debug(f"Split text into {len(sentences)} sentences")
        return sentences

    def _cosine_similarity(
        self,
        embedding_a: List[float],
        embedding_b: List[float]
    ) -> float:
        """Compute cosine similarity between two embedding vectors.

        Formula: cos(θ) = (A · B) / (||A|| × ||B||)

        Args:
            embedding_a: First embedding vector
            embedding_b: Second embedding vector

        Returns:
            Cosine similarity score (0-1 range, clamped)

        Raises:
            ValueError: If embeddings have different dimensions
        """
        if len(embedding_a) != len(embedding_b):
            raise ValueError(
                f"Embedding dimension mismatch: {len(embedding_a)} vs {len(embedding_b)}"
            )

        if len(embedding_a) == 0 or len(embedding_b) == 0:
            raise ValueError("Embeddings cannot be empty")

        # Compute dot product
        dot_product = sum(a * b for a, b in zip(embedding_a, embedding_b))

        # Compute magnitudes
        magnitude_a = math.sqrt(sum(a * a for a in embedding_a))
        magnitude_b = math.sqrt(sum(b * b for b in embedding_b))

        # Handle zero vectors
        if magnitude_a == 0 or magnitude_b == 0:
            logger.warning("Zero magnitude vector encountered, returning 0 similarity")
            return 0.0

        # Compute cosine similarity
        similarity = dot_product / (magnitude_a * magnitude_b)

        # Clamp to [0, 1] range
        # Cosine similarity is naturally in [-1, 1], but for embeddings it's usually [0, 1]
        similarity = max(0.0, min(1.0, similarity))

        return similarity
