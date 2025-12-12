# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SimExtract is a similarity attribution research platform that identifies and extracts key spans from text B that contribute most to its similarity with text A. The project explores and compares four different attribution methods:

1. **Segmented Vectorization**: Splits text B into segments (sentences/fixed-length/semantic units) and computes vector similarity with text A
   - Advantages: Simple implementation, computationally efficient
   - Status: Skeleton established, pending implementation

2. **Cross-Encoder Attention Analysis**: Uses BERT Cross-Encoder to process `[CLS] A [SEP] B [SEP]` and analyzes attention matrices to extract attention weights from A's tokens to B's tokens
   - Advantages: Captures deep interactions, strong theoretical foundation
   - Status: Skeleton established, pending implementation

3. **Token Wise**: Generates token-level weights/embeddings through sparse embeddings or Late Interaction mechanisms (MaxSim) to calculate attribution scores for each token
   - Implementation approaches:
     - **Sparse embedding-based token weights**: Supported via BGE-M3, which can return "lexical weights" (learned sparse weights similar to BM25) for each term while generating dense vectors, enabling term-matching scores and per-word contributions
     - **ColBERT-based token weights**: Supports obtaining vectors for each token (multi-vector/late interaction) to calculate token alignment similarity contributions; official API provides overall ColBERT scores, but returned token vectors can be used to derive per-token weights or alignment matrices
   - Advantages: Token-level granularity, preserves context
   - Status: Skeleton established, pending implementation

4. **Late Chunking**: First generates full-text contextual embeddings, then performs intelligent chunking and aggregation at the embedding layer
   - Advantages: Preserves global context, semantic chunking
   - Status: Skeleton established, pending implementation

## Architecture

The codebase follows a plugin-based architecture with clear separation of concerns:

### Core Abstractions

- **`src/attribution/base.py`**: Defines the attribution contract with `AttributionMethod` (abstract base class), `AttributionResult`, and `AttributionSpan` dataclasses. All attribution methods must inherit from `AttributionMethod` and implement the `extract(text_a, text_b)` method.

- **`src/data_pipeline/`**: Handles the full data ingestion pipeline with three base abstractions:
  - `BaseReader` → `CSVReader`: Reads data sources
  - `BaseVectorizer` → `TEIVectorizer`: Converts text to embeddings via TEI service
  - `BaseStore` → `ChromaStore`: Persists vectors to ChromaDB
  - `BatchProcessor`: Orchestrates the pipeline with progress tracking

- **`src/data_pipeline/samplers/`**: Retrieves document pairs from ChromaDB for attribution experiments. `RandomQuerySampler` randomly selects a document and returns its top-k most similar documents.

### Configuration System

Configuration is managed through YAML files parsed by `src/utils/config.py`. The `Config` class provides dot-notation access (e.g., `config.tei.api_url`). All configs live in `config/` directory:

- `data_prep.yaml`: TEI endpoint, ChromaDB path, batch sizes, input CSV files and their target collections
- Example configurations use `.yaml.example` suffix

### Logging

Logging is centralized in `src/utils/logger.py` with colorlog support. All scripts create timestamped log files in `logs/` directory. Use `get_logger(__name__)` to obtain a logger instance.

## Key Workflows

### Data Preparation Pipeline

```bash
# Start TEI service (required for vectorization)
docker run -p 8080:80 --rm -v $PWD/data:/data \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.2 \
  --model-id sentence-transformers/all-MiniLM-L6-v2

# Prepare data (reads config/data_prep.yaml)
python scripts/prepare_data.py
```

The pipeline flow: `CSVReader` → `TEIVectorizer` (batched API calls) → `ChromaStore` (batched inserts with progress bars). Collections are named per-CSV file based on config.

### Running Attribution Methods

Attribution methods can be tested with mock or real vectorizers:

```bash
# Unit test with MockVectorizer (no TEI required)
python tests/test_segmented_attribution.py

# End-to-end test with TEI (requires running TEI container)
python -m src.experiments.test_segmented_attribution

# Sanity check random sampling
python scripts/test_random_sampler.py --collection <collection_name> --n 5
```

### Typical Attribution Flow

1. Use `RandomQuerySampler` to retrieve a query document and its top-k similar documents from ChromaDB
2. Instantiate an attribution method (e.g., `SegmentedAttribution`) with configuration dict
3. For each similar document, call `method.extract(query_text, similar_doc_text)`
4. Process the returned `AttributionResult` to get top scoring spans via `result.top_k_spans(k)`

## Development Commands

```bash
# Environment setup
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -e .

# Alternative: using uv (if available)
uv sync

# Run data preparation
python scripts/prepare_data.py

# Run attribution tests
python tests/test_segmented_attribution.py  # Unit test with MockVectorizer
python -m src.experiments.test_segmented_attribution  # E2E test with TEI

# Test random sampling
python scripts/test_random_sampler.py --collection <name> --n 5
```

## Project Structure Details

```
src/
├── attribution/          # Attribution methods
│   ├── base.py          # Abstract base classes and result dataclasses
│   ├── segmented/       # Segmented vectorization method
│   ├── cross_encoder/   # Cross-encoder attention method
│   ├── token_wise/      # Token-level attribution (sparse embeddings + ColBERT)
│   └── late_chunking/   # Late chunking method
├── data_pipeline/       # Data ingestion pipeline
│   ├── readers/         # Data source readers (CSV, etc.)
│   ├── vectorizers/     # Text-to-embedding converters (TEI)
│   ├── stores/          # Vector stores (ChromaDB)
│   ├── samplers/        # Document retrieval samplers
│   └── batch_processor.py  # Pipeline orchestration
├── experiments/         # Runnable experiments and tests
└── utils/              # Shared utilities (config, logging)

scripts/                # CLI entry points for common tasks
tests/                  # Unit tests with mock dependencies
config/                 # YAML configuration files
data/                   # Input CSV files (not in git)
chroma_db/             # ChromaDB persistence directory
logs/                  # Timestamped log files
```

## Adding a New Attribution Method

1. Create a new directory under `src/attribution/<method_name>/`
2. Create `method.py` that inherits from `AttributionMethod`
3. Implement `extract(text_a, text_b) -> AttributionResult`
4. Return `AttributionResult` containing list of `AttributionSpan` objects with scores
5. Add imports to `src/attribution/__init__.py`
6. Create a unit test in `tests/test_<method_name>.py` (with MockVectorizer) and/or an E2E test in `src/experiments/test_<method_name>.py` (with real TEI service)

## Important Implementation Notes

### TEI Service Dependency

- ChromaDB is used **only** for vector storage and retrieval, not for vectorization
- All vectorization goes through TEI service via `TEIVectorizer`
- Always call `check_tei_service(url)` before starting batch operations
- TEI calls are batched according to `config.tei.batch_size` for efficiency

### Attribution Method Contract

- Input: `text_a` (query/reference), `text_b` (target for extraction)
- Output: `AttributionResult` with spans sorted by score (0-1 range)
- Spans must have valid character indices (`start_idx`, `end_idx`) into `text_b`
- Include method-specific metadata in span/result metadata dicts for debugging

### Configuration and Secrets

- Never commit actual TEI endpoints or API tokens
- Use `config/data_prep.yaml.example` as template
- Store secrets in environment variables and reference in configs
- Sanitize logs when sharing (redact document content unless from public samples)

### Testing Strategy

- **Unit tests** in `tests/`: Mock-based tests (e.g., `MockVectorizer`) for fast, isolated testing without external dependencies
- **E2E tests** in `src/experiments/`: Integration tests with real TEI service and ChromaDB, documenting required Docker commands
- No pytest suite yet; run tests directly with `python tests/test_*.py` or `python -m src.experiments.test_*`
- Name test files consistently: `test_<area>.py`

## Coding Conventions

- Python 3.9+ with type hints
- Four-space indentation, `snake_case` functions, `CamelCase` classes
- Module-level docstrings summarizing responsibilities
- Use dataclasses from `src/utils/config.py` for structured configs
- Structured logging with `get_logger(__name__)` - avoid f-strings with secrets
- Configuration defaults in YAML examples, not hardcoded constants

## Git Workflow

- Commit format: `type：summary` (e.g., `feat：添加初始项目结构`)
- Keep subjects under 72 characters
- Mention affected modules in commit body
- PRs should describe problem, include test commands with sample output
- Flag changes requiring ChromaDB regeneration or TEI service restart
- Main branch: `main`
