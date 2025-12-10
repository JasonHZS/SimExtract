"""Segmented vectorization attribution method (skeleton)."""

from ..base import AttributionMethod, AttributionResult


class SegmentedAttribution(AttributionMethod):
    """Similarity attribution via segmented vectorization.

    This method splits text_b into segments, vectorizes each segment,
    and computes similarity with text_a to identify contributing spans.

    TODO: Implement the segmentation and attribution logic.
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
            "SegmentedAttribution.extract() is not yet implemented. "
            "This is a skeleton implementation for future development."
        )
