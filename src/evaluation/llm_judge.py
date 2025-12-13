"""LLM-as-a-Judge evaluation method.

This module implements evaluation using LLMs to score span relevance
on a 1-5 scale.

Example:
    >>> from src.utils.llm_client import LLMClient
    >>> from src.evaluation import LLMJudgeEvaluator
    >>>
    >>> client = LLMClient()  # Uses OPENAI_API_KEY from env
    >>> evaluator = LLMJudgeEvaluator(client)
    >>>
    >>> source = "AI is transforming the healthcare industry."
    >>> target = "Machine learning for medical diagnosis is revolutionizing patient care."
    >>> span = "Machine learning for medical diagnosis"
    >>>
    >>> result = evaluator.evaluate(source, target, span)
    >>> print(f"LLM Judge score: {result.score}/5")
"""

import re
import logging
from typing import Any, Dict, Union, Optional

from src.attribution.base import AttributionSpan
from src.evaluation.base import BaseEvaluator, EvaluationResult
from src.utils.llm_client import LLMClient


logger = logging.getLogger(__name__)


DEFAULT_PROMPT_TEMPLATE = """Given Source Text A and a Span S extracted from Target Text B, evaluate how much Span S contributes to understanding the semantic similarity or relevance to Text A.

**Source Text A:**
{source_text}

**Target Text B:**
{target_text}

**Extracted Span S:**
{span_text}

**Task:**
Rate how relevant Span S is to Text A on a scale of 1 to 5:
- 1: Completely irrelevant, no connection to Text A
- 2: Slightly relevant, weak or tangential connection
- 3: Moderately relevant, some meaningful connection
- 4: Highly relevant, strong semantic connection
- 5: Perfectly relevant, captures the core similarity to Text A

**Output ONLY a single number (1-5).**"""


class LLMJudgeEvaluator(BaseEvaluator):
    """Evaluator using LLM to judge span relevance.

    This evaluator uses a large language model to assess how well
    an extracted span captures the semantic relevance to the source text.

    The LLM provides a human-like judgment on a 1-5 scale, which can
    capture nuanced relevance that automatic metrics might miss.

    Score interpretation:
        - 1: Completely irrelevant
        - 2: Slightly relevant
        - 3: Moderately relevant
        - 4: Highly relevant
        - 5: Perfectly relevant
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = None,
        prompt_template: str = None,
        temperature: float = 0.0,
        config: Dict[str, Any] = None,
    ):
        """Initialize the LLM judge evaluator.

        Args:
            llm_client: LLM client instance for API calls.
            model: Model to use for evaluation.
                   Defaults to client's default model.
            prompt_template: Custom prompt template with placeholders:
                             {source_text}, {target_text}, {span_text}
            temperature: Sampling temperature (0.0 for deterministic).
            config: Optional configuration dictionary.
        """
        super().__init__(config)
        self.llm_client = llm_client
        self.model = model
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.temperature = temperature

    def evaluate(
        self,
        source_text: str,
        target_text: str,
        span: Union[str, AttributionSpan],
    ) -> EvaluationResult:
        """Evaluate span relevance using LLM judgment.

        Args:
            source_text: The reference text (text A)
            target_text: The full target text (text B)
            span: The extracted span to evaluate

        Returns:
            EvaluationResult with LLM score (1-5)
        """
        # Validate inputs
        self._validate_inputs(source_text, target_text, span)

        # Extract span text
        span_text, start_idx, end_idx = self._extract_span_text(target_text, span)

        # Construct prompt
        prompt = self.prompt_template.format(
            source_text=source_text,
            target_text=target_text,
            span_text=span_text,
        )

        # Call LLM
        response = self.llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=self.temperature,
            max_tokens=10,  # We only need a single number
        )

        # Parse score from response
        score = self._parse_score(response)

        return EvaluationResult(
            score=score,
            metric_name="llm_judge_score",
            metadata={
                "raw_response": response,
                "span_text": span_text,
                "span_start": start_idx,
                "span_end": end_idx,
                "model": self.model or self.llm_client.default_model,
                "normalized_score": score / 5.0,  # Normalized to 0-1
            },
        )

    def _parse_score(self, response: str) -> float:
        """Parse score from LLM response.

        Handles various formats:
        - "3"
        - "Score: 3"
        - "3/5"
        - "3 out of 5"

        Args:
            response: Raw LLM response text

        Returns:
            Score as float (1.0-5.0)

        Raises:
            ValueError: If score cannot be parsed
        """
        response = response.strip()

        # Try to find a number 1-5 in the response
        # Pattern: standalone number, or number before /5, or "Score: N"
        patterns = [
            r"^(\d)$",  # Just a single digit
            r"^(\d)\.0*$",  # Digit with .0
            r"[Ss]core[:\s]*(\d)",  # "Score: N" or "score N"
            r"(\d)\s*/\s*5",  # "N/5"
            r"(\d)\s+out\s+of\s+5",  # "N out of 5"
            r"^[^\d]*(\d)[^\d]*$",  # Any single digit in the string
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                score = int(match.group(1))
                if 1 <= score <= 5:
                    return float(score)

        # If no valid score found, log warning and return middle score
        logger.warning(
            f"Could not parse score from LLM response: '{response}'. "
            f"Defaulting to 3."
        )
        return 3.0

    def evaluate_multiple_spans(
        self,
        source_text: str,
        target_text: str,
        spans: list[Union[str, AttributionSpan]],
    ) -> list[EvaluationResult]:
        """Evaluate multiple spans.

        Note: This makes separate LLM calls for each span.
        For large numbers of spans, consider batching strategies.

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

    def set_prompt_template(self, template: str) -> None:
        """Update the prompt template.

        Args:
            template: New prompt template with placeholders:
                      {source_text}, {target_text}, {span_text}
        """
        # Validate template has required placeholders
        required = ["{source_text}", "{target_text}", "{span_text}"]
        missing = [p for p in required if p not in template]
        if missing:
            raise ValueError(f"Prompt template missing placeholders: {missing}")
        self.prompt_template = template

