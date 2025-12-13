"""Cross-Encoder evaluation method.

This module implements evaluation using cross-encoder models (rerankers)
to score the relevance between source text and extracted spans.

Cross-encoders are more accurate than bi-encoders because they process
both texts together, capturing complex interaction patterns.

Example:
    >>> from src.data_pipeline.rerankers import TEIReranker
    >>> from src.evaluation import CrossEncoderEvaluator
    >>>
    >>> reranker = TEIReranker(api_url="http://localhost:8080/rerank")
    >>> evaluator = CrossEncoderEvaluator(reranker)
    >>>
    >>> source = "AI is transforming the healthcare industry."
    >>> target = "Machine learning for medical diagnosis is revolutionizing patient care."
    >>> span = "Machine learning for medical diagnosis"
    >>>
    >>> result = evaluator.evaluate(source, target, span)
    >>> print(f"Cross-encoder score: {result.score:.4f}")
"""

from typing import Any, Dict, Union

from src.attribution.base import AttributionSpan
from src.data_pipeline.rerankers.base import BaseReranker
from src.evaluation.base import BaseEvaluator, EvaluationResult


class CrossEncoderEvaluator(BaseEvaluator):
    """Evaluator using cross-encoder model to score span relevance.

    This evaluator uses a cross-encoder (reranker) model to directly
    score the semantic relevance between the source text and the
    extracted span.

    Unlike bi-encoder approaches that separately encode texts and
    compute cosine similarity, cross-encoders process both texts
    together through attention layers, capturing deeper interactions.

    Score interpretation:
        - Higher scores (closer to 1.0): Strong semantic relevance
        - Lower scores (closer to 0.0): Weak semantic relevance
    """

    def __init__(
        self,
        reranker: BaseReranker,
        config: Dict[str, Any] = None,
    ):
        """Initialize the cross-encoder evaluator.

        Args:
            reranker: Reranker instance for scoring query-document pairs
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.reranker = reranker

    def evaluate(
        self,
        source_text: str,
        target_text: str,
        span: Union[str, AttributionSpan],
    ) -> EvaluationResult:
        """Evaluate span relevance using cross-encoder scoring.

        Scores the extracted span against the source text using
        a cross-encoder model.

        Args:
            source_text: The reference text (text A / query)
            target_text: The full target text (text B)
            span: The extracted span to evaluate

        Returns:
            EvaluationResult with cross-encoder score (0-1)
        """
        # Validate inputs
        self._validate_inputs(source_text, target_text, span)

        # Extract span text
        span_text, start_idx, end_idx = self._extract_span_text(target_text, span)

        # Score using cross-encoder: (source_text, span_text)
        # The reranker treats source_text as query and span_text as document
        score = self.reranker.rerank_single(
            query=source_text,
            document=span_text,
        )

        return EvaluationResult(
            score=score,
            metric_name="cross_encoder_score",
            metadata={
                "span_text": span_text,
                "span_start": start_idx,
                "span_end": end_idx,
                "span_length": len(span_text),
                "source_length": len(source_text),
                "target_length": len(target_text),
            },
        )

    def evaluate_multiple_spans(
        self,
        source_text: str,
        target_text: str,
        spans: list[Union[str, AttributionSpan]],
    ) -> list[EvaluationResult]:
        """Evaluate multiple spans efficiently using batch reranking.

        Args:
            source_text: The reference text
            target_text: The target text
            spans: List of spans to evaluate

        Returns:
            List of EvaluationResults, one per span
        """
        if not spans:
            return []

        # Extract all span texts
        span_infos = []
        span_texts = []
        for span in spans:
            self._validate_inputs(source_text, target_text, span)
            span_text, start_idx, end_idx = self._extract_span_text(target_text, span)
            span_infos.append((span_text, start_idx, end_idx))
            span_texts.append(span_text)

        # Batch rerank all spans
        rerank_results = self.reranker.rerank(
            query=source_text,
            documents=span_texts,
        )

        # Map scores back to original order
        # rerank_results are sorted by score, but we need original order
        score_by_index = {r.index: r.score for r in rerank_results}

        results = []
        for i, (span_text, start_idx, end_idx) in enumerate(span_infos):
            score = score_by_index.get(i, 0.0)
            results.append(
                EvaluationResult(
                    score=score,
                    metric_name="cross_encoder_score",
                    metadata={
                        "span_text": span_text,
                        "span_start": start_idx,
                        "span_end": end_idx,
                        "span_length": len(span_text),
                        "source_length": len(source_text),
                        "target_length": len(target_text),
                    },
                )
            )

        return results

