"""Late chunking attribution method (skeleton)."""

from ..base import AttributionMethod, AttributionResult


class LateChunkingAttribution(AttributionMethod):
    """Similarity attribution via late chunking strategy.

    This method generates full-context embeddings first, then applies
    intelligent chunking at the embedding level to preserve context
    information while computing attribution scores.

    TODO: Implement the late chunking and embedding aggregation logic.
    """

    def extract(self, text_a: str, text_b: str) -> AttributionResult:
        """Extract attribution from text_b to text_a.

        Args:
            text_a: Source text (reference)
            text_b: Target text (to be analyzed)

        Returns:
            AttributionResult with scored spans

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError(
            "LateChunkingAttribution.extract() is not yet implemented. "
            "This is a skeleton implementation for future development."
        )
