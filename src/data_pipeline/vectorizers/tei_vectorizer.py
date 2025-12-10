"""TEI (Text Embeddings Inference) vectorizer implementation."""

import time
import logging
from typing import List, cast
import requests
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction

from .base import BaseVectorizer


logger = logging.getLogger(__name__)


class TEIVectorizer(BaseVectorizer, EmbeddingFunction[Documents]):
    """TEI API vectorizer with batch support and retry logic.

    Features:
    - Batch processing for efficiency
    - Exponential backoff retry
    - Timeout control
    - Health check support
    - ChromaDB EmbeddingFunction compatibility
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8080/embed",
        batch_size: int = 64,
        max_retries: int = 3,
        timeout: int = 60,
        dimension: int = 1024
    ):
        """Initialize TEI vectorizer.

        Args:
            api_url: TEI API endpoint URL
            batch_size: Number of texts to process per batch
            max_retries: Maximum retry attempts on failure
            timeout: Request timeout in seconds
            dimension: Expected embedding dimension
        """
        self.api_url = api_url
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.dimension = dimension

        logger.info(
            f"TEI Vectorizer initialized: {api_url} "
            f"(batch_size={batch_size}, timeout={timeout}s)"
        )

    def __call__(self, input: Documents) -> Embeddings:
        """ChromaDB EmbeddingFunction interface.

        This method is called by ChromaDB when vectorizing documents.

        Args:
            input: List of text strings (Documents type from ChromaDB)

        Returns:
            List of embedding vectors (Embeddings type from ChromaDB)
        """
        return self.embed(input)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Convert texts to embeddings.

        This method calls embed_batch internally with the configured batch size.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If texts is empty
            RuntimeError: If vectorization fails after all retries
        """
        if not texts:
            raise ValueError("Cannot embed empty text list")

        return self.embed_batch(texts, self.batch_size)

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int
    ) -> List[List[float]]:
        """Convert texts to embeddings in batches.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process per batch

        Returns:
            List of embedding vectors in same order as input

        Raises:
            ValueError: If texts is empty or batch_size is invalid
            RuntimeError: If vectorization fails
        """
        if not texts:
            raise ValueError("Cannot embed empty text list")
        if batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {batch_size}")

        logger.debug(
            f"Embedding {len(texts)} texts in batches of {batch_size}"
        )

        all_embeddings: List[List[float]] = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

            logger.debug(
                f"Batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} "
                f"completed ({len(batch)} texts)"
            )

        if len(all_embeddings) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(texts)}, "
                f"got {len(all_embeddings)}"
            )

        return cast(Embeddings, all_embeddings)

    def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch with exponential backoff retry.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If all retry attempts fail
        """
        for attempt in range(self.max_retries):
            try:
                return self._embed_batch_request(texts)

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        f"TEI request timeout (attempt {attempt + 1}/{self.max_retries}). "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("TEI request failed after all retry attempts")
                    raise RuntimeError(
                        f"TEI vectorization failed after {self.max_retries} attempts"
                    )

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"TEI request error: {e} (attempt {attempt + 1}/{self.max_retries}). "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"TEI request failed after all retries: {e}")
                    raise RuntimeError(f"TEI vectorization failed: {e}")

            except Exception as e:
                logger.error(f"Unexpected error in TEI vectorization: {e}")
                raise RuntimeError(f"TEI vectorization failed: {e}")

        raise RuntimeError("Should not reach here")

    def _embed_batch_request(self, texts: List[str]) -> List[List[float]]:
        """Make a single TEI API request.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            requests.exceptions.Timeout: If request times out
            requests.exceptions.RequestException: If request fails
            ValueError: If response format is invalid
        """
        response = requests.post(
            self.api_url,
            json={"inputs": texts},
            timeout=self.timeout,
            headers={"Content-Type": "application/json"}
        )

        # Check HTTP status
        response.raise_for_status()

        # Parse response
        embeddings = response.json()

        # Validate response format
        if not isinstance(embeddings, list):
            raise ValueError("TEI response must be a list")

        if len(embeddings) != len(texts):
            raise ValueError(
                f"TEI returned {len(embeddings)} embeddings "
                f"for {len(texts)} texts"
            )

        return embeddings

    def get_dimension(self) -> int:
        """Get the dimension of output embeddings.

        Returns:
            Embedding dimension size
        """
        return self.dimension

    def health_check(self) -> bool:
        """Check if the TEI service is healthy.

        Returns:
            True if service is available, False otherwise
        """
        try:
            # Try to embed a simple test text
            test_text = ["test"]
            self._embed_batch_request(test_text)
            logger.info("TEI service health check passed")
            return True

        except Exception as e:
            logger.warning(f"TEI service health check failed: {e}")
            return False


def check_tei_service(api_url: str, timeout: int = 10) -> bool:
    """Check if TEI service is available.

    This is a standalone function for easy service checking.

    Args:
        api_url: TEI API endpoint URL
        timeout: Request timeout in seconds

    Returns:
        True if service is available, False otherwise
    """
    try:
        response = requests.post(
            api_url,
            json={"inputs": ["test"]},
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        logger.info(f"TEI service is available at {api_url}")
        return True

    except Exception as e:
        logger.error(f"TEI service check failed: {e}")
        return False
