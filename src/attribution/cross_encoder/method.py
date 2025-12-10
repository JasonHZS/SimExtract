"""Cross-Encoder attention analysis attribution method (skeleton)."""

from ..base import AttributionMethod, AttributionResult


class CrossEncoderAttribution(AttributionMethod):
    """Similarity attribution via Cross-Encoder attention analysis.

    This method uses BERT Cross-Encoder to process text pairs and
    analyzes the attention matrix to identify which tokens in text_b
    receive attention from text_a.

    TODO: Implement the Cross-Encoder and attention extraction logic.
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
            "CrossEncoderAttribution.extract() is not yet implemented. "
            "This is a skeleton implementation for future development."
        )
