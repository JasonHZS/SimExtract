"""OpenAI vectorizer implementation (placeholder).

TODO: Implement OpenAI embedding API integration.

This module will support vectorization using OpenAI's embedding models
like text-embedding-3-large.
"""

from .base import BaseVectorizer


class OpenAIVectorizer(BaseVectorizer):
    """OpenAI embedding API vectorizer (not yet implemented).

    TODO: Implement OpenAI API integration for:
    - text-embedding-3-small
    - text-embedding-3-large
    - text-embedding-ada-002
    """

    def embed(self, texts):
        raise NotImplementedError(
            "OpenAIVectorizer is not yet implemented. "
            "This is a placeholder for future development."
        )

    def embed_batch(self, texts, batch_size):
        raise NotImplementedError(
            "OpenAIVectorizer is not yet implemented. "
            "This is a placeholder for future development."
        )
