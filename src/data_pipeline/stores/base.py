"""Base class for vector stores."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseStore(ABC):
    """Abstract base class for vector storage backends.

    All vector stores should inherit from this class and implement
    the required methods for storing and managing vector collections.
    """

    @abstractmethod
    def get_or_create_collection(
        self,
        name: str,
        **kwargs
    ):
        """Get existing collection or create a new one.

        Args:
            name: Collection name
            **kwargs: Store-specific configuration options

        Returns:
            Collection object (store-specific type)

        Raises:
            ValueError: If collection name is invalid
            RuntimeError: If collection creation fails
        """
        pass

    @abstractmethod
    def batch_add(
        self,
        collection,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        batch_size: int = 100
    ) -> Dict[str, int]:
        """Add documents to collection in batches.

        Args:
            collection: Collection object to add to
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of unique document IDs
            batch_size: Number of documents per batch

        Returns:
            Dictionary with statistics:
                - "added": Number of successfully added documents
                - "failed": Number of failed documents

        Raises:
            ValueError: If input lists have different lengths
            RuntimeError: If batch addition fails
        """
        pass

    def get_collection(self, name: str) -> Optional[Any]:
        """Get existing collection.

        Args:
            name: Collection name

        Returns:
            Collection object if exists, None otherwise

        Note:
            Subclasses should override this if they support
            getting collections without creating them
        """
        raise NotImplementedError(
            "Subclass should implement get_collection() to retrieve collections"
        )

    def delete_collection(self, name: str) -> bool:
        """Delete a collection.

        Args:
            name: Collection name

        Returns:
            True if deleted successfully, False otherwise

        Note:
            Subclasses should override this if they support deletion
        """
        raise NotImplementedError(
            "Subclass should implement delete_collection() to remove collections"
        )

    def list_collections(self) -> List[str]:
        """List all collection names.

        Returns:
            List of collection names

        Note:
            Subclasses should override this if they support listing
        """
        raise NotImplementedError(
            "Subclass should implement list_collections() to list all collections"
        )
