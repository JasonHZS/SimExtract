"""ChromaDB vector store implementation."""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.api.models.Collection import Collection

from .base import BaseStore


logger = logging.getLogger(__name__)


class ChromaStore(BaseStore):
    """ChromaDB vector store with collection management and batch operations.

    Features:
    - Persistent storage
    - Collection management with custom embedding functions
    - Batch insertion with error handling
    - Distance metric configuration
    """

    def __init__(self, persist_directory: str):
        """Initialize ChromaDB client.

        Args:
            persist_directory: Directory for persistent storage
        """
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        logger.info(f"ChromaDB client initialized: {persist_directory}")

    def get_or_create_collection(
        self,
        name: str,
        embedding_function=None,
        distance_metric: str = "cosine",
        **kwargs
    ) -> Collection:
        """Get existing collection or create a new one.

        Args:
            name: Collection name
            embedding_function: Custom embedding function for the collection
            distance_metric: Distance metric ("cosine", "l2", "ip")
            **kwargs: Additional ChromaDB configuration

        Returns:
            Collection object

        Raises:
            ValueError: If distance metric is invalid
            RuntimeError: If collection creation fails
        """
        # Validate distance metric
        valid_metrics = ["cosine", "l2", "ip"]
        if distance_metric not in valid_metrics:
            raise ValueError(
                f"Invalid distance metric: {distance_metric}. "
                f"Must be one of {valid_metrics}"
            )

        try:
            # Prepare metadata
            metadata = {"hnsw:space": distance_metric}
            metadata.update(kwargs.get("metadata", {}))

            collection = self.client.get_or_create_collection(
                name=name,
                embedding_function=embedding_function,
                metadata=metadata
            )

            logger.info(
                f"Collection '{name}' ready "
                f"(distance={distance_metric}, count={collection.count()})"
            )
            return collection

        except Exception as e:
            logger.error(f"Failed to get/create collection '{name}': {e}")
            raise RuntimeError(f"Collection operation failed: {e}")

    def batch_add(
        self,
        collection: Collection,
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
        """
        # Validate inputs
        if not (len(documents) == len(metadatas) == len(ids)):
            raise ValueError(
                f"Length mismatch: documents={len(documents)}, "
                f"metadatas={len(metadatas)}, ids={len(ids)}"
            )

        if not documents:
            logger.warning("No documents to add")
            return {"added": 0, "failed": 0}

        stats = {"added": 0, "failed": 0}
        total = len(documents)

        logger.debug(f"Adding {total} documents in batches of {batch_size}")

        # Process in batches
        for i in range(0, total, batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            try:
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                stats["added"] += len(batch_docs)
                logger.debug(
                    f"Batch {i // batch_size + 1}: "
                    f"Added {len(batch_docs)} documents"
                )

            except Exception as e:
                stats["failed"] += len(batch_docs)
                logger.error(
                    f"Batch {i // batch_size + 1} failed: {e}. "
                    f"Skipping {len(batch_docs)} documents"
                )

                # Try adding individually to identify problematic documents
                for j, (doc, meta, doc_id) in enumerate(
                    zip(batch_docs, batch_metas, batch_ids)
                ):
                    try:
                        collection.add(
                            documents=[doc],
                            metadatas=[meta],
                            ids=[doc_id]
                        )
                        stats["added"] += 1
                        stats["failed"] -= 1
                    except Exception as e2:
                        logger.warning(
                            f"Failed to add document {doc_id}: {e2}"
                        )

        logger.info(
            f"Batch add completed: {stats['added']} added, "
            f"{stats['failed']} failed"
        )
        return stats

    def get_collection(self, name: str) -> Optional[Collection]:
        """Get existing collection.

        Args:
            name: Collection name

        Returns:
            Collection object or None if not found
        """
        try:
            collection = self.client.get_collection(name)
            logger.info(f"Collection '{name}' retrieved (count={collection.count()})")
            return collection
        except Exception:
            logger.warning(f"Collection '{name}' not found")
            return None

    def list_collections(self) -> List[str]:
        """List all collection names.

        Returns:
            List of collection names
        """
        try:
            collections = self.client.list_collections()
            names = [c.name for c in collections]
            logger.info(f"Found {len(names)} collections: {names}")
            return names
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def delete_collection(self, name: str) -> bool:
        """Delete a collection.

        Args:
            name: Collection name

        Returns:
            True if deleted, False if not found or error
        """
        try:
            self.client.delete_collection(name)
            logger.info(f"Collection '{name}' deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{name}': {e}")
            return False

    def query_collection(
        self,
        collection: Collection,
        query_texts: List[str] = None,
        query_embeddings: List[List[float]] = None,
        n_results: int = 10,
        where: Dict[str, Any] = None,
        where_document: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Query collection for similar documents.

        Args:
            collection: Collection to query
            query_texts: List of query texts (if using embedding function)
            query_embeddings: List of query embeddings
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document content filter

        Returns:
            Query results dictionary

        Raises:
            ValueError: If neither query_texts nor query_embeddings provided
        """
        if not query_texts and not query_embeddings:
            raise ValueError("Must provide either query_texts or query_embeddings")

        try:
            results = collection.query(
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            logger.debug(f"Query returned {len(results['ids'][0])} results")
            return results

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise RuntimeError(f"Failed to query collection: {e}")
