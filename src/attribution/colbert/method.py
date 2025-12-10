"""ColBERT late interaction attribution method (skeleton)."""

from ..base import AttributionMethod, AttributionResult


class ColBERTAttribution(AttributionMethod):
    """Similarity attribution via ColBERT late interaction.

    This method generates token-level embeddings for both texts and
    computes MaxSim scores to identify which tokens in text_b are
    most similar to tokens in text_a.

    TODO: Implement the ColBERT model and MaxSim computation logic.
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
            "ColBERTAttribution.extract() is not yet implemented. "
            "This is a skeleton implementation for future development."
        )
