"""Base class for data readers."""

from abc import ABC, abstractmethod
from typing import Iterator, Tuple, List, Dict, Any
import pandas as pd


class BaseReader(ABC):
    """Abstract base class for data readers.

    All data readers should inherit from this class and implement
    the required methods for reading and preparing documents.
    """

    @abstractmethod
    def read_in_chunks(
        self,
        file_path: str,
        chunksize: int
    ) -> Iterator[pd.DataFrame]:
        """Read data file in chunks.

        Args:
            file_path: Path to the data file
            chunksize: Number of rows per chunk

        Returns:
            Iterator of DataFrame chunks

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        pass

    @abstractmethod
    def prepare_documents(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """Prepare documents, metadata, and IDs from DataFrame.

        Args:
            df: DataFrame containing the data

        Returns:
            Tuple of (documents, metadatas, ids) where:
                - documents: List of document texts
                - metadatas: List of metadata dictionaries
                - ids: List of unique document IDs

        Raises:
            ValueError: If required columns are missing
        """
        pass

    def get_total_rows(self, file_path: str) -> int:
        """Get total number of rows in file.

        Args:
            file_path: Path to the data file

        Returns:
            Total number of rows

        Note:
            This method can be overridden for optimized counting
        """
        raise NotImplementedError(
            "Subclass must implement get_total_rows() or use default implementation"
        )
