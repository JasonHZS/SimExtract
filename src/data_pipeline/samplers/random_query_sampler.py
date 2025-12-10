"""Random query sampler for ChromaDB collections.

This module provides a utility to randomly sample documents from ChromaDB
collections and query for similar documents using vector similarity search.
"""

import logging
import random
from typing import Dict, Any, Optional, List

from chromadb.api.models.Collection import Collection

from src.data_pipeline.stores.chroma_store import ChromaStore


logger = logging.getLogger(__name__)


class RandomQuerySampler:
    """Randomly sample a document and query for similar documents.

    This class provides functionality to randomly select a document from a
    ChromaDB collection and query for its n most similar documents based on
    vector similarity. Useful for experimental evaluation and analysis.

    The returned results include the original sampled document plus n similar
    documents (n+1 total), in ChromaDB's native dictionary format.

    Example:
        >>> from src.data_pipeline.stores.chroma_store import ChromaStore
        >>> from src.data_pipeline.samplers import RandomQuerySampler
        >>>
        >>> store = ChromaStore(persist_directory="./chroma_db")
        >>> sampler = RandomQuerySampler(store, random_seed=42)
        >>> results = sampler.sample_and_query("my_collection", n_results=5)
        >>> print(f"Total documents: {len(results['ids'][0])}")  # 6 (1 + 5)
        >>> print(f"Query doc distance: {results['distances'][0][0]}")  # ~0.0

    Attributes:
        chroma_store: ChromaStore instance for database access
        random_seed: Optional seed for reproducibility
    """

    def __init__(self, chroma_store: ChromaStore, random_seed: Optional[int] = None):
        """Initialize the random query sampler.

        Args:
            chroma_store: ChromaStore instance for accessing ChromaDB
            random_seed: Optional seed for reproducible random sampling

        Raises:
            TypeError: If chroma_store is not a ChromaStore instance
        """
        if not isinstance(chroma_store, ChromaStore):
            raise TypeError("chroma_store must be a ChromaStore instance")

        self.chroma_store = chroma_store
        self.random_seed = random_seed

        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
            logger.info(f"Random seed set to {random_seed}")

        logger.debug("RandomQuerySampler initialized")

    def sample_and_query(
        self,
        collection_name: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """Sample a random document and query for similar documents.

        This method randomly selects a document from the specified collection,
        extracts its embedding vector, and queries for the n most similar documents.
        The returned results include the original document (first result with
        distance ~0) plus n similar documents, totaling n+1 documents.

        Args:
            collection_name: Name of the ChromaDB collection to sample from
            n_results: Number of similar documents to retrieve (default: 5)
                      Note: Total returned documents will be n_results + 1

        Returns:
            Dictionary in ChromaDB native format:
                {
                    'ids': [[doc_id1, doc_id2, ...]],
                    'documents': [[doc1_text, doc2_text, ...]],
                    'metadatas': [[meta1, meta2, ...]],
                    'distances': [[0.0, dist1, dist2, ...]]
                }
            The first element (index 0) is the query document with distance ~0.0

        Raises:
            ValueError: If collection doesn't exist or is empty, or if n_results < 0
            RuntimeError: If document fetching or querying fails

        Example:
            >>> results = sampler.sample_and_query("my_collection", n_results=3)
            >>> print(f"Documents returned: {len(results['ids'][0])}")  # 4
            >>> print(f"Query doc ID: {results['ids'][0][0]}")
            >>> print(f"Most similar doc: {results['ids'][0][1]}")
        """
        if n_results < 0:
            raise ValueError(f"n_results must be non-negative, got {n_results}")

        logger.info(f"Starting sample_and_query for collection '{collection_name}' "
                   f"with n_results={n_results}")

        # Get collection
        collection = self._get_collection(collection_name)

        # Validate collection has documents
        collection_count = collection.count()
        if collection_count == 0:
            raise ValueError(f"Collection '{collection_name}' is empty, cannot sample documents")

        logger.info(f"Collection '{collection_name}' has {collection_count} documents")

        # Adjust n_results if necessary
        n_results = self._validate_n_results(n_results, collection_count)

        # Get random document ID
        doc_id = self._get_random_document_id(collection)
        logger.info(f"Randomly selected document: {doc_id}")

        # Fetch document with embedding
        doc_data = self._fetch_document_with_embedding(collection, doc_id)
        embedding = doc_data['embeddings'][0]
        logger.debug(f"Extracted embedding vector of shape {len(embedding)}")

        # Query for similar documents (n+1 to include query doc)
        results = self._query_similar_documents(collection, embedding, n_results + 1)

        # Verify query document is first result
        if results['ids'][0][0] == doc_id:
            logger.debug(f"Verified query document is first result (distance: {results['distances'][0][0]:.6f})")
        else:
            logger.warning(f"Query document {doc_id} is not the first result. "
                         f"First result: {results['ids'][0][0]}")

        logger.info(f"Query completed successfully. Returned {len(results['ids'][0])} documents "
                   f"(1 query doc + {len(results['ids'][0])-1} similar docs)")

        return results

    def _get_collection(self, collection_name: str) -> Collection:
        """Get collection from ChromaStore.

        Args:
            collection_name: Name of the collection

        Returns:
            Collection object

        Raises:
            ValueError: If collection doesn't exist
        """
        collection = self.chroma_store.get_collection(collection_name)
        if collection is None:
            logger.error(f"Collection '{collection_name}' not found")
            raise ValueError(f"Collection '{collection_name}' not found in ChromaDB")
        return collection

    def _validate_n_results(self, n_results: int, collection_count: int) -> int:
        """Validate and adjust n_results based on collection size.

        Args:
            n_results: Requested number of results
            collection_count: Total documents in collection

        Returns:
            Adjusted n_results value
        """
        # Maximum n_results is collection_count - 1 (excluding query doc)
        max_n_results = collection_count - 1

        if n_results > max_n_results:
            logger.warning(
                f"Requested n_results={n_results} exceeds available documents. "
                f"Collection has {collection_count} documents. "
                f"Adjusting n_results to {max_n_results}"
            )
            return max_n_results

        return n_results

    def _get_random_document_id(self, collection: Collection) -> str:
        """Randomly select a document ID from the collection.

        Args:
            collection: ChromaDB Collection object

        Returns:
            Randomly selected document ID

        Raises:
            RuntimeError: If fetching document IDs fails
        """
        try:
            # Get all document IDs
            all_data = collection.get()
            all_ids = all_data['ids']

            if not all_ids:
                raise ValueError("No document IDs found in collection")

            # Randomly select one ID
            selected_id = random.choice(all_ids)
            logger.debug(f"Selected document ID: {selected_id} from {len(all_ids)} documents")

            return selected_id

        except Exception as e:
            logger.error(f"Failed to get random document ID: {e}")
            raise RuntimeError(f"Failed to get random document ID: {e}") from e

    def _fetch_document_with_embedding(
        self,
        collection: Collection,
        doc_id: str
    ) -> Dict[str, Any]:
        """Fetch a specific document with its embedding vector.

        Args:
            collection: ChromaDB Collection object
            doc_id: Document ID to fetch

        Returns:
            Dictionary with document data including embedding

        Raises:
            RuntimeError: If document fetching fails
        """
        try:
            doc_data = collection.get(
                ids=[doc_id],
                include=['embeddings', 'documents', 'metadatas']
            )

            # Verify embedding exists
            if (doc_data['embeddings'] is None or
                len(doc_data['embeddings']) == 0 or
                doc_data['embeddings'][0] is None):
                raise ValueError(f"Document {doc_id} has no embedding vector")

            logger.debug(f"Fetched document {doc_id} with embedding")
            return doc_data

        except Exception as e:
            logger.error(f"Failed to fetch document {doc_id}: {e}")
            raise RuntimeError(f"Failed to fetch document {doc_id}: {e}") from e

    def _query_similar_documents(
        self,
        collection: Collection,
        embedding: List[float],
        n_results: int
    ) -> Dict[str, Any]:
        """Query collection for documents similar to the given embedding.

        Args:
            collection: ChromaDB Collection object
            embedding: Query embedding vector
            n_results: Number of results to return

        Returns:
            ChromaDB query results dictionary

        Raises:
            RuntimeError: If query fails
        """
        try:
            # Query using ChromaStore's query_collection method
            results = self.chroma_store.query_collection(
                collection=collection,
                query_embeddings=[embedding],  # Must be List[List[float]]
                n_results=n_results
            )

            logger.debug(f"Query returned {len(results['ids'][0])} results")
            return results

        except Exception as e:
            logger.error(f"Failed to query collection: {e}")
            raise RuntimeError(f"Failed to query collection: {e}") from e
