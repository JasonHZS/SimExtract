"""Local vectorizer implementation (placeholder).

TODO: Implement local sentence-transformers based vectorization.

This module will support vectorization using locally-run models
via the sentence-transformers library.
"""

from .base import BaseVectorizer


class LocalVectorizer(BaseVectorizer):
    """Local sentence-transformers vectorizer (not yet implemented).

    TODO: Implement local vectorization using:
    - sentence-transformers library
    - HuggingFace transformers
    - Support for custom models
    """

    def embed(self, texts):
        raise NotImplementedError(
            "LocalVectorizer is not yet implemented. "
            "This is a placeholder for future development."
        )

    def embed_batch(self, texts, batch_size):
        raise NotImplementedError(
            "LocalVectorizer is not yet implemented. "
            "This is a placeholder for future development."
        )
