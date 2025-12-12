#!/usr/bin/env python3
"""Data preparation script for CSV ingestion into ChromaDB."""

import sys
import os
import logging

from src.utils.config import load_config, validate_config
from src.utils.logger import setup_logging, get_logger
from src.data_pipeline.vectorizers.tei_vectorizer import TEIVectorizer, check_tei_service
from src.data_pipeline.stores.chroma_store import ChromaStore
from src.data_pipeline.readers.csv_reader import CSVReader
from src.data_pipeline.batch_processor import BatchProcessor, print_summary


def main():
    """Main entry point for data ingestion."""

    # Load configuration
    try:
        config = load_config("config/data_prep.yaml")
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        sys.exit(1)

    # Setup logging
    try:
        setup_logging(
            log_level=config.logging.level,
            log_dir=config.logging.log_dir,
            log_format=config.logging.format
        )
    except Exception as e:
        print(f"ERROR: Failed to setup logging: {e}")
        sys.exit(1)

    logger = get_logger(__name__)
    logger.info("=" * 70)
    logger.info(" CSV Data Preparation to ChromaDB")
    logger.info("=" * 70)

    # Validate configuration
    warnings = validate_config(config)
    if warnings:
        logger.warning("Configuration warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")

    # Check TEI service
    logger.info("Checking TEI service availability...")
    if not check_tei_service(config.tei.api_url, timeout=10):
        logger.error(
            f"TEI service is not available at {config.tei.api_url}. "
            "Please ensure the service is running."
        )
        sys.exit(1)

    # Initialize components
    logger.info("Initializing components...")

    try:
        # Embedding function - use TEIVectorizer
        embedding_func = TEIVectorizer(
            api_url=config.tei.api_url,
            batch_size=config.tei.batch_size,
            max_retries=config.tei.max_retries,
            timeout=config.tei.timeout
        )

        # ChromaDB store - use ChromaStore
        chroma_store = ChromaStore(
            persist_directory=config.chromadb.persist_directory
        )

        # Data reader (CSV) with optional minimum content length filter
        min_content_length = getattr(config.processing, 'min_content_length', 0)
        reader = CSVReader(
            skip_empty_content=config.processing.skip_empty_content,
            min_content_length=min_content_length
        )

        if min_content_length > 0:
            logger.info(f"Content filter: min_content_length = {min_content_length}")

        # Check clear_before_import setting
        clear_before_import = getattr(config.processing, 'clear_before_import', False)
        if clear_before_import:
            logger.info("Mode: clear_before_import = True (collections will be cleared)")

        # Batch processor
        processor = BatchProcessor(
            chroma_store=chroma_store,
            reader=reader,
            config=config
        )

        logger.info("All components initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        sys.exit(1)

    # Process files
    logger.info(f"Starting processing of {len(config.data.files)} files...")
    logger.info("")

    try:
        stats = processor.process_all_files(
            file_configs=config.data.files,
            embedding_func=embedding_func
        )

        # Print summary
        print_summary(stats)

        # Exit with appropriate code
        if stats["total_failed"] > 0:
            logger.warning(
                f"Processing completed with {stats['total_failed']} failures"
            )
            sys.exit(1)
        else:
            logger.info("Processing completed successfully!")
            sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Processing failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
