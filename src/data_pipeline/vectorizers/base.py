"""Base class for vectorizers."""

from abc import ABC, abstractmethod
from typing import List


class BaseVectorizer(ABC):
    """Abstract base class for text vectorizers.

    All vectorizers should inherit from this class and implement
    the required methods for converting text to embeddings.
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Convert texts to embeddings.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors, where each vector is a list of floats

        Raises:
            ValueError: If texts is empty or contains invalid entries
            RuntimeError: If vectorization fails
        """
        pass

    @abstractmethod
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
            List of embedding vectors

        Raises:
            ValueError: If texts is empty or batch_size is invalid
            RuntimeError: If vectorization fails

        Note:
            This method should handle batching internally and return
            all embeddings in the same order as input texts.
        """
        pass

    def get_dimension(self) -> int:
        """Get the dimension of output embeddings.

        Returns:
            Embedding dimension size

        Note:
            Subclasses should override this if they know the dimension
        """
        raise NotImplementedError(
            "Subclass should implement get_dimension() to return embedding size"
        )

    def health_check(self) -> bool:
        """Check if the vectorizer service is healthy.

        Returns:
            True if service is available and working, False otherwise

        Note:
            Subclasses can override this for custom health checks
        """
        return True  # Default: assume healthy
