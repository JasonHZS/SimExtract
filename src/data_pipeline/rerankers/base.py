"""Base class for rerankers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RerankResult:
    """Result from a reranking operation.

    Contains the reranked documents with their scores.
    """

    index: int  # Original index in the input list
    text: str  # The document text
    score: float  # Relevance score from the cross-encoder
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseReranker(ABC):
    """Abstract base class for rerankers.

    All rerankers should inherit from this class and implement
    the required methods for reranking documents against a query.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None,
    ) -> List[RerankResult]:
        """Rerank documents based on relevance to query.

        Args:
            query: The query text to rank against
            documents: List of document texts to rerank
            top_k: Optional limit on number of results to return.
                   If None, returns all documents ranked.

        Returns:
            List of RerankResult sorted by score (descending)

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If reranking fails
        """
        pass

    def rerank_single(self, query: str, document: str) -> float:
        """Get relevance score for a single query-document pair.

        Convenience method for scoring a single document.

        Args:
            query: The query text
            document: The document text to score

        Returns:
            Relevance score (typically 0-1 for sigmoid output)
        """
        results = self.rerank(query, [document], top_k=1)
        if results:
            return results[0].score
        return 0.0

    def health_check(self) -> bool:
        """Check if the reranker service is healthy.

        Returns:
            True if service is available and working, False otherwise

        Note:
            Subclasses can override this for custom health checks
        """
        return True  # Default: assume healthy

