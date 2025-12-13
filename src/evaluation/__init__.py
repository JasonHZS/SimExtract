"""Evaluation module for attribution methods.

This module provides tools to assess the quality of similarity
attribution results using various metrics.

Available evaluators:
    - DropOneEvaluator: Measures span contribution via ablation
    - CrossEncoderEvaluator: Scores span relevance using cross-encoder models
    - LLMJudgeEvaluator: Uses LLM to judge span relevance (1-5 scale)

Example (Drop-one ablation):
    >>> from src.data_pipeline.vectorizers import TEIVectorizer
    >>> from src.evaluation import DropOneEvaluator
    >>>
    >>> vectorizer = TEIVectorizer(api_url="http://localhost:8080/embed")
    >>> evaluator = DropOneEvaluator(vectorizer)
    >>> result = evaluator.evaluate(source, target, span)
    >>> print(f"Contribution: {result.score:.4f}")

Example (Cross-encoder):
    >>> from src.data_pipeline.rerankers import TEIReranker
    >>> from src.evaluation import CrossEncoderEvaluator
    >>>
    >>> reranker = TEIReranker(api_url="http://localhost:8080/rerank")
    >>> evaluator = CrossEncoderEvaluator(reranker)
    >>> result = evaluator.evaluate(source, target, span)
    >>> print(f"Relevance: {result.score:.4f}")

Example (LLM-as-a-Judge):
    >>> from src.utils.llm_client import LLMClient
    >>> from src.evaluation import LLMJudgeEvaluator
    >>>
    >>> client = LLMClient()  # Uses OPENAI_API_KEY from .env
    >>> evaluator = LLMJudgeEvaluator(client)
    >>> result = evaluator.evaluate(source, target, span)
    >>> print(f"LLM score: {result.score}/5")
"""

from src.evaluation.base import BaseEvaluator, EvaluationResult
from src.evaluation.cross_encoder import CrossEncoderEvaluator
from src.evaluation.drop_one import DropOneEvaluator
from src.evaluation.llm_judge import LLMJudgeEvaluator

__all__ = [
    "BaseEvaluator",
    "EvaluationResult",
    "CrossEncoderEvaluator",
    "DropOneEvaluator",
    "LLMJudgeEvaluator",
]
