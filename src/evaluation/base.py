"""Base class for evaluation methods.

This module defines the abstract interface for all evaluation methods
that assess the quality of similarity attribution results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Union

from src.attribution.base import AttributionSpan


@dataclass
class EvaluationResult:
    """Result from an evaluation metric.

    Contains the evaluation score and metadata about the assessment.
    """

    score: float  # The evaluation metric value
    metric_name: str  # Name of the metric (e.g., "drop_one_contribution")
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional info

    def __post_init__(self):
        """Validate result data."""
        if not self.metric_name:
            raise ValueError("metric_name cannot be empty")


class BaseEvaluator(ABC):
    """Abstract base class for attribution evaluation methods.

    All evaluation methods should inherit from this class and implement
    the required methods for assessing attribution quality.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize evaluator.

        Args:
            config: Optional configuration dictionary for the evaluator
        """
        self.config = config or {}
        self.name = self.__class__.__name__

    @abstractmethod
    def evaluate(
        self,
        source_text: str,
        target_text: str,
        span: Union[str, AttributionSpan],
    ) -> EvaluationResult:
        """Evaluate the quality of an attributed span.

        This is the core method that assesses how well a span from target_text
        explains the similarity to source_text.

        Args:
            source_text: The reference text (text A)
            target_text: The full target text (text B)
            span: The extracted span to evaluate. Can be:
                - str: The span text (will match first occurrence in target_text)
                - AttributionSpan: Span with precise character positions

        Returns:
            EvaluationResult with the evaluation score and metadata

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If evaluation fails
        """
        pass

    def _validate_inputs(
        self,
        source_text: str,
        target_text: str,
        span: Union[str, AttributionSpan],
    ) -> None:
        """Validate evaluation inputs.

        Args:
            source_text: The reference text
            target_text: The target text
            span: The span to evaluate

        Raises:
            ValueError: If any input is invalid
        """
        if not source_text or not source_text.strip():
            raise ValueError("source_text cannot be empty")
        if not target_text or not target_text.strip():
            raise ValueError("target_text cannot be empty")

        if isinstance(span, str):
            if not span or not span.strip():
                raise ValueError("span string cannot be empty")
            if span not in target_text:
                raise ValueError("span string not found in target_text")
        elif isinstance(span, AttributionSpan):
            if span.start_idx < 0 or span.end_idx > len(target_text):
                raise ValueError("span indices out of bounds for target_text")
            # Verify span text matches
            extracted = target_text[span.start_idx : span.end_idx]
            if extracted != span.text:
                raise ValueError(
                    f"span.text '{span.text}' does not match target_text "
                    f"at indices [{span.start_idx}:{span.end_idx}]: '{extracted}'"
                )
        else:
            raise TypeError(
                f"span must be str or AttributionSpan, got {type(span).__name__}"
            )

    def _extract_span_text(
        self,
        target_text: str,
        span: Union[str, AttributionSpan],
    ) -> tuple[str, int, int]:
        """Extract span text and character positions.

        Args:
            target_text: The target text containing the span
            span: The span (string or AttributionSpan)

        Returns:
            Tuple of (span_text, start_idx, end_idx)
        """
        if isinstance(span, AttributionSpan):
            return span.text, span.start_idx, span.end_idx
        else:
            # For string span, find first occurrence
            start_idx = target_text.find(span)
            end_idx = start_idx + len(span)
            return span, start_idx, end_idx

    def get_config(self) -> Dict[str, Any]:
        """Get evaluator configuration.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def set_config(self, config: Dict[str, Any]) -> None:
        """Update evaluator configuration.

        Args:
            config: New configuration to merge with existing
        """
        self.config.update(config)

