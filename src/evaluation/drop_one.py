"""Drop-one ablation evaluation method.

This module implements contribution-based evaluation by measuring
how much the similarity drops when a span is removed from the target text.

Example:
    >>> from src.data_pipeline.vectorizers.tei_vectorizer import TEIVectorizer
    >>> from src.evaluation.drop_one import DropOneEvaluator
    >>>
    >>> vectorizer = TEIVectorizer(endpoint="http://localhost:8080")
    >>> evaluator = DropOneEvaluator(vectorizer)
    >>>
    >>> source = "AI is transforming the healthcare industry."
    >>> target = "Machine learning for medical diagnosis is revolutionizing patient care."
    >>> span = "Machine learning for medical diagnosis"
    >>>
    >>> result = evaluator.evaluate(source, target, span)
    >>> print(f"Contribution score: {result.score:.4f}")
"""

import numpy as np
from typing import Any, Dict, Union

from src.attribution.base import AttributionSpan
from src.data_pipeline.vectorizers.base import BaseVectorizer
from src.evaluation.base import BaseEvaluator, EvaluationResult
from src.utils.similarity import cosine_similarity


class DropOneEvaluator(BaseEvaluator):
    """Evaluator using drop-one ablation to measure span contribution.

    This evaluator measures how much a span contributes to the similarity
    between source and target texts by computing:

        contribution = baseline_similarity - ablated_similarity

    Where:
        - baseline_similarity = sim(source, target)
        - ablated_similarity = sim(source, target_with_span_removed)

    A higher contribution score means the span is more important for
    explaining the similarity between the texts.
    """

    def __init__(
        self,
        vectorizer: BaseVectorizer,
        config: Dict[str, Any] = None,
    ):
        """Initialize the drop-one evaluator.

        Args:
            vectorizer: Vectorizer instance for computing embeddings
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.vectorizer = vectorizer

    def evaluate(
        self,
        source_text: str,
        target_text: str,
        span: Union[str, AttributionSpan],
    ) -> EvaluationResult:
        """Evaluate span contribution using drop-one ablation.

        Measures how much the similarity drops when the span is removed
        from the target text.

        Args:
            source_text: The reference text (text A)
            target_text: The full target text (text B)
            span: The extracted span to evaluate

        Returns:
            EvaluationResult with contribution score:
                - score > 0: Removing span decreases similarity (good attribution)
                - score ≈ 0: Span has little impact on similarity
                - score < 0: Removing span increases similarity (poor attribution)
        """
        # Validate inputs
        self._validate_inputs(source_text, target_text, span)

        # Extract span text and positions
        span_text, start_idx, end_idx = self._extract_span_text(target_text, span)

        # Create ablated text by removing the span
        ablated_text = self._remove_span(target_text, start_idx, end_idx)

        # Compute embeddings
        embeddings = self.vectorizer.embed([source_text, target_text, ablated_text])
        vec_source = np.array(embeddings[0])
        vec_target = np.array(embeddings[1])
        vec_ablated = np.array(embeddings[2])

        # Compute similarities
        baseline_sim = cosine_similarity(vec_source, vec_target)
        ablated_sim = cosine_similarity(vec_source, vec_ablated)

        # Contribution = how much similarity drops when span is removed
        contribution = baseline_sim - ablated_sim

        return EvaluationResult(
            score=contribution,
            metric_name="drop_one_contribution",
            metadata={
                "baseline_similarity": baseline_sim,
                "ablated_similarity": ablated_sim,
                "span_text": span_text,
                "span_start": start_idx,
                "span_end": end_idx,
                "ablated_text": ablated_text,
                "span_length": len(span_text),
                "target_length": len(target_text),
                "span_ratio": len(span_text) / len(target_text) if target_text else 0,
            },
        )

    def _remove_span(self, text: str, start_idx: int, end_idx: int) -> str:
        """Remove a span from text and clean up whitespace.

        Args:
            text: Original text
            start_idx: Start character index of span
            end_idx: End character index of span

        Returns:
            Text with span removed and whitespace cleaned
        """
        # Remove the span
        before = text[:start_idx]
        after = text[end_idx:]

        # Join with appropriate whitespace
        # Avoid double spaces or leading/trailing spaces
        result = before.rstrip() + " " + after.lstrip()
        return result.strip()

    def evaluate_multiple_spans(
        self,
        source_text: str,
        target_text: str,
        spans: list[Union[str, AttributionSpan]],
    ) -> list[EvaluationResult]:
        """Evaluate multiple spans and rank by contribution.

        Args:
            source_text: The reference text
            target_text: The target text
            spans: List of spans to evaluate

        Returns:
            List of EvaluationResults, one per span
        """
        results = []
        for span in spans:
            result = self.evaluate(source_text, target_text, span)
            results.append(result)
        return results

