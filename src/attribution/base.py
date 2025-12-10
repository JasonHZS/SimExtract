"""Base class for similarity attribution methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class AttributionSpan:
    """A text span with attribution score.

    Represents a segment of text from text_b that contributes
    to similarity with text_a.
    """
    text: str  # The actual text content
    start_idx: int  # Character start index in text_b
    end_idx: int  # Character end index in text_b
    score: float  # Attribution score (0-1, higher = more similar)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional info

    def __post_init__(self):
        """Validate span data."""
        if self.start_idx < 0:
            raise ValueError("start_idx must be non-negative")
        if self.end_idx < self.start_idx:
            raise ValueError("end_idx must be >= start_idx")
        if not 0 <= self.score <= 1:
            raise ValueError("score must be between 0 and 1")


@dataclass
class AttributionResult:
    """Result from a similarity attribution analysis.

    Contains the attribution spans from text_b that are most
    similar to text_a, ranked by score.
    """
    text_a: str  # Source text (reference)
    text_b: str  # Target text (to be analyzed)
    method_name: str  # Attribution method used
    overall_similarity: float  # Overall similarity score (0-1)
    spans: List[AttributionSpan]  # Attributed spans, sorted by score
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional info

    def __post_init__(self):
        """Validate result data."""
        if not 0 <= self.overall_similarity <= 1:
            raise ValueError("overall_similarity must be between 0 and 1")

    def top_k_spans(self, k: int = 5) -> List[AttributionSpan]:
        """Get top-k highest scoring spans.

        Args:
            k: Number of top spans to return

        Returns:
            List of top-k spans sorted by score (descending)
        """
        return sorted(
            self.spans,
            key=lambda x: x.score,
            reverse=True
        )[:k]

    def get_coverage(self, threshold: float = 0.5) -> float:
        """Calculate percentage of text_b covered by high-scoring spans.

        Args:
            threshold: Minimum score for a span to be counted

        Returns:
            Coverage ratio (0-1)
        """
        if not self.text_b:
            return 0.0

        total_chars = len(self.text_b)
        covered_chars = sum(
            span.end_idx - span.start_idx
            for span in self.spans
            if span.score >= threshold
        )

        return min(covered_chars / total_chars, 1.0)


class AttributionMethod(ABC):
    """Abstract base class for similarity attribution methods.

    All attribution methods should inherit from this class and
    implement the required methods for extracting attribution
    from text pairs.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize attribution method.

        Args:
            config: Configuration dictionary for the method
        """
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def extract(
        self,
        text_a: str,
        text_b: str
    ) -> AttributionResult:
        """Extract attribution from text_b to text_a.

        This is the core method that identifies which parts of text_b
        contribute most to its similarity with text_a.

        Args:
            text_a: Source text (the reference)
            text_b: Target text (to be analyzed for attribution)

        Returns:
            AttributionResult with scored spans

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If attribution fails
        """
        pass

    def batch_extract(
        self,
        pairs: List[tuple[str, str]]
    ) -> List[AttributionResult]:
        """Extract attribution for multiple text pairs.

        Default implementation calls extract() for each pair.
        Subclasses can override for optimized batch processing.

        Args:
            pairs: List of (text_a, text_b) tuples

        Returns:
            List of AttributionResults
        """
        results = []
        for text_a, text_b in pairs:
            result = self.extract(text_a, text_b)
            results.append(result)
        return results

    def get_config(self) -> Dict[str, Any]:
        """Get method configuration.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def set_config(self, config: Dict[str, Any]) -> None:
        """Update method configuration.

        Args:
            config: New configuration to merge with existing
        """
        self.config.update(config)
