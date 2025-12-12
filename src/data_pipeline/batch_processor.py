"""Batch processor for coordinating data ingestion."""

import os
import time
import logging
from typing import List, Dict, Any
from tqdm import tqdm

from .stores.chroma_store import ChromaStore
from .readers.base import BaseReader
from ..utils.config import Config


logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Batch processor that coordinates data reading, vectorization, and storage.

    Features:
    - Multi-file processing
    - Progress tracking with tqdm
    - Statistics collection
    - Error handling and recovery
    """

    def __init__(
        self,
        chroma_store: ChromaStore,
        reader: BaseReader,
        config: Config
    ):
        """
        Initialize batch processor.

        Args:
            chroma_store: ChromaDB store instance
            reader: Data reader instance (any BaseReader implementation)
            config: Configuration object
        """
        self.chroma_client = chroma_store  # Keep old variable name for compatibility
        self.reader = reader
        self.config = config

    def process_single_file(
        self,
        file_path: str,
        collection_name: str,
        embedding_func
    ) -> Dict[str, Any]:
        """
        Process a single data file.

        Args:
            file_path: Path to data file
            collection_name: Name of ChromaDB collection
            embedding_func: Embedding function

        Returns:
            Statistics dictionary with processing results
        """
        logger.info(f"Processing file: {file_path} → {collection_name}")

        # Initialize statistics
        stats = {
            "file": file_path,
            "collection": collection_name,
            "total_rows": 0,
            "processed": 0,
            "skipped_empty": 0,
            "skipped_too_short": 0,
            "added": 0,
            "failed": 0,
            "cleared_count": 0,
            "start_time": time.time(),
            "end_time": None,
            "duration": None
        }

        try:
            # Clear collection if configured
            clear_before_import = getattr(
                self.config.processing, 'clear_before_import', False
            )
            if clear_before_import:
                clear_result = self.chroma_client.clear_collection(collection_name)
                if clear_result["success"]:
                    stats["cleared_count"] = clear_result["previous_count"]
                    logger.info(clear_result["message"])
                else:
                    logger.warning(clear_result["message"])

            # Get or create collection
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_func,
                distance_metric=self.config.chromadb.distance_metric
            )

            # Get total rows for progress bar
            total_rows = self.reader.get_total_rows(file_path)
            stats["total_rows"] = total_rows

            # Process file in chunks
            with tqdm(
                total=total_rows,
                desc=f"Processing {os.path.basename(file_path)}",
                unit="rows"
            ) as pbar:
                for chunk in self.reader.read_in_chunks(
                    file_path,
                    chunksize=self.config.processing.csv_chunk_size
                ):
                    # Prepare documents
                    documents, metadatas, ids = self.reader.prepare_documents(chunk)

                    # Track skipped records (including too short ones)
                    skipped_too_short = getattr(self.reader, 'skipped_too_short', 0)
                    stats["skipped_too_short"] += skipped_too_short
                    # skipped_empty = total skipped - too_short
                    skipped_empty = len(chunk) - len(documents) - skipped_too_short
                    stats["skipped_empty"] += skipped_empty

                    if documents:
                        # Add to ChromaDB
                        add_stats = self.chroma_client.batch_add(
                            collection=collection,
                            documents=documents,
                            metadatas=metadatas,
                            ids=ids,
                            batch_size=self.config.processing.chroma_batch_size
                        )

                        stats["added"] += add_stats["added"]
                        stats["failed"] += add_stats["failed"]
                        stats["processed"] += len(documents)

                    # Update progress bar
                    pbar.update(len(chunk))

            # Finalize statistics
            stats["end_time"] = time.time()
            stats["duration"] = stats["end_time"] - stats["start_time"]

            logger.info(
                f"Completed {file_path}: "
                f"{stats['processed']} processed, "
                f"{stats['added']} added, "
                f"{stats['skipped_empty']} skipped (empty), "
                f"{stats['skipped_too_short']} skipped (too short), "
                f"{stats['failed']} failed, "
                f"duration: {stats['duration']:.2f}s"
            )

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            stats["error"] = str(e)
            stats["end_time"] = time.time()
            stats["duration"] = stats["end_time"] - stats["start_time"]

        return stats

    def process_all_files(
        self,
        file_configs: List[Dict[str, str]],
        embedding_func
    ) -> Dict[str, Any]:
        """
        Process all configured files.

        Args:
            file_configs: List of file configurations with 'name' and 'collection'
            embedding_func: Embedding function

        Returns:
            Aggregated statistics for all files
        """
        logger.info(f"Starting batch processing of {len(file_configs)} files")

        # Initialize aggregated statistics
        all_stats = {
            "files": [],
            "total_files": len(file_configs),
            "total_rows": 0,
            "total_processed": 0,
            "total_skipped_empty": 0,
            "total_skipped_too_short": 0,
            "total_cleared": 0,
            "total_added": 0,
            "total_failed": 0,
            "start_time": time.time(),
            "end_time": None,
            "total_duration": None
        }

        # Process each file
        for i, file_config in enumerate(file_configs, 1):
            file_name = file_config["name"]
            collection_name = file_config["collection"]

            # Construct full file path
            input_dir = self.config.data.input_dir
            file_path = os.path.join(input_dir, file_name)

            logger.info(f"[{i}/{len(file_configs)}] Processing {file_name}")

            # Process file
            file_stats = self.process_single_file(
                file_path=file_path,
                collection_name=collection_name,
                embedding_func=embedding_func
            )

            # Aggregate statistics
            all_stats["files"].append(file_stats)
            all_stats["total_rows"] += file_stats["total_rows"]
            all_stats["total_processed"] += file_stats["processed"]
            all_stats["total_skipped_empty"] += file_stats["skipped_empty"]
            all_stats["total_skipped_too_short"] += file_stats.get("skipped_too_short", 0)
            all_stats["total_cleared"] += file_stats.get("cleared_count", 0)
            all_stats["total_added"] += file_stats["added"]
            all_stats["total_failed"] += file_stats["failed"]

        # Finalize statistics
        all_stats["end_time"] = time.time()
        all_stats["total_duration"] = all_stats["end_time"] - all_stats["start_time"]

        logger.info(
            f"Batch processing completed: "
            f"{all_stats['total_processed']} processed, "
            f"{all_stats['total_added']} added, "
            f"{all_stats['total_skipped_empty']} skipped (empty), "
            f"{all_stats['total_skipped_too_short']} skipped (too short), "
            f"{all_stats['total_failed']} failed, "
            f"total duration: {all_stats['total_duration']:.2f}s"
        )

        return all_stats


def print_summary(stats: Dict[str, Any]) -> None:
    """
    Print a formatted summary of processing statistics.

    Args:
        stats: Statistics dictionary from process_all_files
    """
    print("\n" + "=" * 70)
    print(" PROCESSING SUMMARY")
    print("=" * 70)

    # Overall statistics
    print(f"\nTotal Files Processed: {stats['total_files']}")
    print(f"Total Rows: {stats['total_rows']:,}")
    print(f"Successfully Stored: {stats['total_added']:,}")
    print(f"Skipped (empty content): {stats['total_skipped_empty']:,}")
    print(f"Skipped (too short): {stats.get('total_skipped_too_short', 0):,}")
    if stats.get('total_cleared', 0) > 0:
        print(f"Cleared (before import): {stats['total_cleared']:,}")
    print(f"Failed: {stats['total_failed']:,}")

    # Duration
    duration_mins = stats['total_duration'] / 60
    print(f"Total Time: {duration_mins:.1f} minutes ({stats['total_duration']:.1f}s)")

    # Per-file breakdown
    print("\n" + "-" * 70)
    print(" Per-File Breakdown:")
    print("-" * 70)

    for file_stats in stats['files']:
        file_name = os.path.basename(file_stats['file'])
        collection = file_stats['collection']
        processed = file_stats['processed']
        added = file_stats['added']
        skipped_empty = file_stats['skipped_empty']
        skipped_short = file_stats.get('skipped_too_short', 0)
        cleared = file_stats.get('cleared_count', 0)
        duration = file_stats['duration']

        print(f"\n  {file_name} → {collection}")
        print(f"    Processed: {processed:,} | Added: {added:,}")
        print(f"    Skipped (empty): {skipped_empty:,} | Skipped (short): {skipped_short:,}")
        if cleared > 0:
            print(f"    Cleared: {cleared:,}")
        print(f"    Duration: {duration:.1f}s")

        if "error" in file_stats:
            print(f"    ERROR: {file_stats['error']}")

    # Storage estimate
    vector_dim = 1024  # BGE model dimension
    bytes_per_float = 4
    storage_mb = (stats['total_added'] * vector_dim * bytes_per_float) / (1024 * 1024)

    print("\n" + "-" * 70)
    print(f" Estimated Storage Size: ~{storage_mb:.0f} MB")
    print("=" * 70 + "\n")
