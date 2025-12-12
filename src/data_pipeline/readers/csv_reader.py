"""CSV file reader implementation."""

import os
import logging
from typing import Iterator, Tuple, List, Dict, Any
import pandas as pd

from .base import BaseReader


logger = logging.getLogger(__name__)


class CSVReader(BaseReader):
    """CSV file reader with chunked processing and data preparation.

    Features:
    - Memory-efficient chunked reading
    - Flexible field extraction
    - Empty content filtering
    - Minimum content length filtering
    - Type conversion utilities
    """

    def __init__(
        self,
        skip_empty_content: bool = True,
        min_content_length: int = 0,
        content_field: str = "content",
        id_field: str = "id",
        metadata_fields: List[str] = None
    ):
        """Initialize CSV reader.

        Args:
            skip_empty_content: Whether to skip rows with empty content
            min_content_length: Minimum character length for content (0 = no filter)
            content_field: Name of the column containing document text
            id_field: Name of the column containing document IDs
            metadata_fields: List of metadata field names to extract
        """
        self.skip_empty_content = skip_empty_content
        self.min_content_length = min_content_length
        self.content_field = content_field
        self.id_field = id_field
        self.metadata_fields = metadata_fields or [
            "author", "date", "is_digest", "title"
        ]
        # 统计：跳过的短文本数量（每次 prepare_documents 调用后更新）
        self.skipped_too_short = 0

    def read_in_chunks(
        self,
        file_path: str,
        chunksize: int = 1000
    ) -> Iterator[pd.DataFrame]:
        """Read CSV file in chunks using pandas.

        Args:
            file_path: Path to CSV file
            chunksize: Number of rows per chunk

        Yields:
            DataFrame chunks

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid CSV
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        logger.info(f"Reading CSV file: {file_path} (chunksize={chunksize})")

        try:
            chunks = pd.read_csv(
                file_path,
                chunksize=chunksize,
                encoding='utf-8'
            )

            for i, chunk in enumerate(chunks):
                logger.debug(f"Read chunk {i+1} with {len(chunk)} rows")
                yield chunk

        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise ValueError(f"Failed to read CSV file {file_path}: {e}")

    def prepare_documents(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """Prepare documents, metadata, and IDs from DataFrame.

        Args:
            df: DataFrame containing the data

        Returns:
            Tuple of (documents, metadatas, ids)

        Raises:
            ValueError: If required columns are missing
        """
        # Validate required columns
        if self.content_field not in df.columns:
            raise ValueError(f"Required column '{self.content_field}' not found")
        if self.id_field not in df.columns:
            raise ValueError(f"Required column '{self.id_field}' not found")

        documents = []
        metadatas = []
        ids = []
        # 重置短文本计数器
        self.skipped_too_short = 0

        for idx, row in df.iterrows():
            content = row.get(self.content_field, '')

            # Skip empty content if configured
            if self.skip_empty_content:
                if pd.isna(content) or not str(content).strip():
                    logger.debug(f"Skipping row {idx}: empty content")
                    continue

            # Convert to string and strip
            content_str = str(content).strip()

            # Skip content shorter than minimum length
            if self.min_content_length > 0 and len(content_str) < self.min_content_length:
                logger.debug(
                    f"Skipping row {idx}: content too short "
                    f"({len(content_str)} < {self.min_content_length})"
                )
                self.skipped_too_short += 1
                continue

            # Extract metadata
            metadata = {}
            for field in self.metadata_fields:
                if field in df.columns:
                    value = row.get(field)
                    if field == 'is_digest':
                        metadata[field] = self._safe_bool(value)
                    else:
                        metadata[field] = self._safe_str(value)

            # Prepare document data
            documents.append(content_str)
            metadatas.append(metadata)
            ids.append(str(row.get(self.id_field, f"row_{idx}")))

        logger.debug(
            f"Prepared {len(documents)} documents "
            f"(skipped {len(df) - len(documents)}, "
            f"too short: {self.skipped_too_short})"
        )

        return documents, metadatas, ids

    def get_total_rows(self, file_path: str) -> int:
        """Get total number of rows in CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            Total number of rows (excluding header)

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        try:
            # Use efficient row counting
            with open(file_path, 'r', encoding='utf-8') as f:
                total = sum(1 for _ in f) - 1  # Subtract header row
            logger.debug(f"Total rows in {file_path}: {total}")
            return total
        except Exception as e:
            logger.warning(f"Failed to count rows, using pandas: {e}")
            # Fallback to pandas
            df = pd.read_csv(file_path)
            return len(df)

    @staticmethod
    def _safe_str(value: Any) -> str:
        """Convert value to string safely.

        Args:
            value: Value to convert

        Returns:
            String representation or empty string
        """
        if pd.isna(value):
            return ""
        return str(value).strip()

    @staticmethod
    def _safe_bool(value: Any) -> bool:
        """Convert value to boolean safely.

        Args:
            value: Value to convert

        Returns:
            Boolean value
        """
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'y')
        return bool(value)
