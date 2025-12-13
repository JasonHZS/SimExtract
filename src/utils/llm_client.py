"""LLM client for OpenAI-compatible API interactions.

Supports OpenAI, Azure OpenAI, Aliyun Dashscope, and other
OpenAI-compatible API endpoints.

Environment Variables:
    OPENAI_API_KEY: API key for OpenAI
    DASHSCOPE_API_KEY: API key for Aliyun Dashscope

The client auto-detects the provider based on which API key is set
and uses the appropriate base URL automatically.

Aliyun Dashscope Example:
    >>> # .env file:
    >>> # DASHSCOPE_API_KEY=sk-xxx
    >>>
    >>> from src.utils.llm_client import LLMClient
    >>> client = LLMClient()  # Auto-detects Dashscope, uses qwen-plus
    >>> response = client.chat_completion(
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )

OpenAI Example:
    >>> # .env file:
    >>> # OPENAI_API_KEY=sk-xxx
    >>>
    >>> from src.utils.llm_client import LLMClient
    >>> client = LLMClient()  # Auto-detects OpenAI, uses gpt-4o-mini
    >>> response = client.chat_completion(
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )
"""

import os
import logging
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Provider base URLs (hardcoded, no need for env vars)
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_BASE_URL_INTL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class LLMClient:
    """Client for OpenAI-compatible LLM APIs.

    Supports multiple providers through the OpenAI client interface:
    - OpenAI (default)
    - Aliyun Dashscope (auto-detected via DASHSCOPE_API_KEY)
    - Other OpenAI-compatible APIs (via base_url parameter)

    Provider auto-detection:
        - If DASHSCOPE_API_KEY is set -> uses Dashscope with qwen-plus
        - If OPENAI_API_KEY is set -> uses OpenAI with gpt-4o-mini

    Recommended Aliyun models:
        - qwen-plus: Balanced performance (recommended)
        - qwen-max: Best quality for complex tasks
        - qwen-turbo: Fastest and cheapest
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        default_model: str = None,
        timeout: int = 60,
        region: str = "cn",
    ):
        """Initialize the LLM client.

        Args:
            api_key: API key for authentication.
                     Falls back to DASHSCOPE_API_KEY or OPENAI_API_KEY env var.
            base_url: Custom base URL for the API.
                      If None, auto-detects based on API key source.
            default_model: Default model to use for completions.
                           Auto-detected based on provider if not specified.
            timeout: Request timeout in seconds.
            region: Region for Dashscope ("cn" for China, "intl" for Singapore).
                    Only used when auto-detecting Dashscope.
        """
        # Determine API key and provider
        self._provider = None

        if api_key:
            self.api_key = api_key
        elif os.getenv("DASHSCOPE_API_KEY"):
            self.api_key = os.getenv("DASHSCOPE_API_KEY")
            self._provider = "dashscope"
        elif os.getenv("OPENAI_API_KEY"):
            self.api_key = os.getenv("OPENAI_API_KEY")
            self._provider = "openai"
        else:
            self.api_key = None

        if not self.api_key:
            raise ValueError(
                "API key is required. Set DASHSCOPE_API_KEY or OPENAI_API_KEY "
                "environment variable, or pass api_key parameter."
            )

        # Determine base URL
        if base_url:
            self.base_url = base_url
        elif self._provider == "dashscope":
            # Auto-set Dashscope base URL based on region
            self.base_url = (
                DASHSCOPE_BASE_URL if region == "cn" else DASHSCOPE_BASE_URL_INTL
            )
        else:
            # OpenAI uses default (None means use openai library default)
            self.base_url = None

        # Auto-detect default model based on provider
        if default_model:
            self.default_model = default_model
        elif self._provider == "dashscope":
            self.default_model = "qwen-plus"
        else:
            self.default_model = "gpt-4o-mini"

        self.timeout = timeout

        # Initialize OpenAI client
        client_kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = OpenAI(**client_kwargs)

        provider_name = self._provider or "custom"
        logger.info(
            f"LLM Client initialized (provider={provider_name}, "
            f"model={self.default_model})"
        )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.0,
        max_tokens: int = None,
        **kwargs,
    ) -> str:
        """Generate a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                      Example: [{"role": "user", "content": "Hello!"}]
            model: Model to use. Defaults to self.default_model.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in response.
            **kwargs: Additional arguments passed to the API.

        Returns:
            The assistant's response content as a string.

        Raises:
            RuntimeError: If the API call fails.
        """
        model = model or self.default_model

        try:
            completion_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens:
                completion_kwargs["max_tokens"] = max_tokens
            completion_kwargs.update(kwargs)

            response = self.client.chat.completions.create(**completion_kwargs)

            content = response.choices[0].message.content
            logger.debug(f"LLM response: {content[:100]}...")
            return content

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise RuntimeError(f"LLM API call failed: {e}")

    def health_check(self) -> bool:
        """Check if the LLM service is accessible.

        Returns:
            True if service is available, False otherwise.
        """
        try:
            self.chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            logger.info("LLM service health check passed")
            return True
        except Exception as e:
            logger.warning(f"LLM service health check failed: {e}")
            return False

    @property
    def provider(self) -> str:
        """Get the detected provider name."""
        return self._provider or "custom"
