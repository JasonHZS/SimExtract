"""TEI (Text Embeddings Inference) reranker implementation.

Uses the TEI rerank endpoint for cross-encoder based document reranking.

Example:
    >>> from src.data_pipeline.rerankers import TEIReranker
    >>>
    >>> reranker = TEIReranker(api_url="http://localhost:8080/rerank")
    >>> results = reranker.rerank(
    ...     query="What is machine learning?",
    ...     documents=["ML is a subset of AI.", "The weather is nice."]
    ... )
    >>> for r in results:
    ...     print(f"{r.score:.4f}: {r.text}")
"""

import time
import logging
from typing import List

import requests

from .base import BaseReranker, RerankResult


logger = logging.getLogger(__name__)


class TEIReranker(BaseReranker):
    """TEI API reranker with retry logic.

    Uses the TEI /rerank endpoint which runs a cross-encoder model
    to score query-document pairs.

    Features:
    - Exponential backoff retry
    - Timeout control
    - Health check support
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8080/rerank",
        max_retries: int = 3,
        timeout: int = 60,
    ):
        """Initialize TEI reranker.

        Args:
            api_url: TEI rerank API endpoint URL
            max_retries: Maximum retry attempts on failure
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.max_retries = max_retries
        self.timeout = timeout

        logger.info(
            f"TEI Reranker initialized: {api_url} (timeout={timeout}s)"
        )

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None,
    ) -> List[RerankResult]:
        """Rerank documents based on relevance to query.

        Args:
            query: The query text to rank against
            documents: List of document texts to rerank
            top_k: Optional limit on number of results to return

        Returns:
            List of RerankResult sorted by score (descending)

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If reranking fails after all retries
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if not documents:
            raise ValueError("documents list cannot be empty")

        # Call TEI API with retry
        raw_results = self._rerank_with_retry(query, documents)

        # Convert to RerankResult objects
        results = []
        for item in raw_results:
            idx = item["index"]
            results.append(
                RerankResult(
                    index=idx,
                    text=documents[idx],
                    score=item["score"],
                )
            )

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        # Apply top_k limit if specified
        if top_k is not None and top_k > 0:
            results = results[:top_k]

        return results

    def _rerank_with_retry(
        self,
        query: str,
        documents: List[str],
    ) -> List[dict]:
        """Rerank with exponential backoff retry.

        Args:
            query: The query text
            documents: List of document texts

        Returns:
            Raw response from TEI API

        Raises:
            RuntimeError: If all retry attempts fail
        """
        for attempt in range(self.max_retries):
            try:
                return self._rerank_request(query, documents)

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"TEI rerank timeout (attempt {attempt + 1}/{self.max_retries}). "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("TEI rerank failed after all retry attempts")
                    raise RuntimeError(
                        f"TEI rerank failed after {self.max_retries} attempts"
                    )

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"TEI rerank error: {e} (attempt {attempt + 1}/{self.max_retries}). "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"TEI rerank failed after all retries: {e}")
                    raise RuntimeError(f"TEI rerank failed: {e}")

            except Exception as e:
                logger.error(f"Unexpected error in TEI rerank: {e}")
                raise RuntimeError(f"TEI rerank failed: {e}")

        raise RuntimeError("Should not reach here")

    def _rerank_request(
        self,
        query: str,
        documents: List[str],
    ) -> List[dict]:
        """Make a single TEI rerank API request.

        Args:
            query: The query text
            documents: List of document texts

        Returns:
            List of dicts with 'index' and 'score' keys

        Raises:
            requests.exceptions.Timeout: If request times out
            requests.exceptions.RequestException: If request fails
            ValueError: If response format is invalid
        """
        response = requests.post(
            self.api_url,
            json={
                "query": query,
                "texts": documents,
            },
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
        )

        # Check HTTP status
        response.raise_for_status()

        # Parse response
        results = response.json()

        # Validate response format
        if not isinstance(results, list):
            raise ValueError("TEI rerank response must be a list")

        for item in results:
            if "index" not in item or "score" not in item:
                raise ValueError(
                    "TEI rerank response items must have 'index' and 'score'"
                )

        return results

    def health_check(self) -> bool:
        """Check if the TEI rerank service is healthy.

        Returns:
            True if service is available, False otherwise
        """
        try:
            # Try to rerank a simple test
            self._rerank_request("test query", ["test document"])
            logger.info("TEI rerank service health check passed")
            return True

        except Exception as e:
            logger.warning(f"TEI rerank service health check failed: {e}")
            return False

