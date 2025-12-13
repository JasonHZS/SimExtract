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

- **`src/data_pipeline/rerankers/`**: Cross-encoder based document reranking infrastructure:
  - `BaseReranker`: Abstract base class for reranking models
  - `TEIReranker`: Implementation using TEI rerank endpoint with retry logic
  - `RerankResult`: Dataclass containing reranked documents with scores

- **`src/evaluation/`**: Framework for evaluating attribution quality with three methods:
  - `BaseEvaluator`: Abstract base class defining the evaluation contract
  - `DropOneEvaluator`: Ablation-based evaluation measuring similarity drop when span is removed
  - `CrossEncoderEvaluator`: Uses reranker models to score span relevance
  - `LLMJudgeEvaluator`: LLM-as-a-Judge evaluation on 1-5 scale
  - `EvaluationResult`: Dataclass containing evaluation scores and metadata

- **`src/utils/`**: Shared utilities:
  - `config.py`: YAML configuration parsing with dot-notation access
  - `logger.py`: Centralized logging with colorlog support
  - `llm_client.py`: OpenAI-compatible LLM client supporting multiple providers (OpenAI, Dashscope)
  - `similarity.py`: Common similarity functions (cosine similarity, etc.)

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

### Evaluating Attribution Quality

The project provides three complementary evaluation methods to assess how well extracted spans explain similarity:

#### 1. Drop-One Ablation (`DropOneEvaluator`)

Measures span contribution by computing how much similarity drops when the span is removed

**Use case:** Quantitative measurement of span importance via ablation testing.

#### 2. Cross-Encoder Scoring (`CrossEncoderEvaluator`)

Uses cross-encoder (reranker) models to directly score span relevance to source text

**Use case:** Deep semantic relevance scoring with cross-attention between texts.

**Note:** TEI rerank endpoint requires a cross-encoder model like `cross-encoder/ms-marco-MiniLM-L-6-v2`.

#### 3. LLM-as-a-Judge (`LLMJudgeEvaluator`)

Uses LLMs to provide human-like relevance judgments on a 1-5 scale

#### Environment Setup for LLM Client

Create a `.env` file in project root with your API keys:

```bash
# Option 1: OpenAI
OPENAI_API_KEY=sk-xxx

# Option 2: Aliyun Dashscope (China region by default)
DASHSCOPE_API_KEY=sk-xxx

# For Singapore region, pass region="intl" to LLMClient constructor
```

**Provider auto-detection priority:** If both keys are set, Dashscope takes precedence.

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

# Run tests (three-tier strategy)
python tests/test_sparse_attribution.py  # Unit test (mock, fast)
CUDA_VISIBLE_DEVICES=1 python tests/test_sparse_integration.py  # Integration test (real model)
python src/experiments/demo_sparse_attribution_with_chroma.py --n 3  # E2E demo (with ChromaDB)

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
│   ├── rerankers/       # Cross-encoder reranking (TEI)
│   └── batch_processor.py  # Pipeline orchestration
├── evaluation/          # Attribution quality evaluation
│   ├── base.py          # Abstract evaluator base class
│   ├── drop_one.py      # Drop-one ablation evaluation
│   ├── cross_encoder.py # Cross-encoder scoring evaluation
│   └── llm_judge.py     # LLM-as-a-Judge evaluation
├── experiments/         # Runnable experiments and tests
└── utils/              # Shared utilities
    ├── config.py        # YAML configuration parsing
    ├── logger.py        # Centralized logging
    ├── llm_client.py    # OpenAI-compatible LLM client
    └── similarity.py    # Similarity computation functions

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

## Adding a New Evaluation Method

1. Create `src/evaluation/<method_name>.py` that inherits from `BaseEvaluator`
2. Implement `evaluate(source_text, target_text, span) -> EvaluationResult`
3. Return `EvaluationResult` with:
   - `score`: The evaluation metric value
   - `metric_name`: Descriptive name (e.g., "drop_one_contribution")
   - `metadata`: Dict with additional info (span positions, intermediate values, etc.)
4. Add imports to `src/evaluation/__init__.py`
5. Optional: Implement `evaluate_multiple_spans()` for batch efficiency
6. Create unit tests in `tests/test_<method_name>_evaluation.py`

## Important Implementation Notes

### TEI Service Dependency

- ChromaDB is used **only** for vector storage and retrieval, not for vectorization
- All vectorization goes through TEI service via `TEIVectorizer`
- All reranking goes through TEI service via `TEIReranker` (requires cross-encoder model)
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

SimExtract uses a multi-layered testing approach to ensure both fast iteration and real-world accuracy:

#### 1. Mock-based Unit Tests (`tests/test_*.py`)
- **Purpose**: Fast validation of business logic (windowing, scoring, ranking, edge cases)
- **Example**: `tests/test_sparse_attribution.py` - uses `MockTokenizer` and `MockBGEM3Model`
- **Pros**: No external dependencies, suitable for CI/CD, tests algorithmic correctness
- **Cons**: Cannot verify compatibility with real models (tokenization, API contracts)
- **When to use**: Rapid development, testing logic changes, CI pipeline

#### 2. Integration Tests (`tests/test_*_integration.py`)
- **Purpose**: Validate real model behavior and API contracts
- **Example**: `tests/test_sparse_integration.py` - uses actual BGE-M3 model from `models/` directory
- **Key verifications**:
  - Tokenizer behavior (WordPiece/XLMRoberta tokenization)
  - API return formats (`lexical_weights` uses string keys, not int; weights are `np.float16` with FP16)
  - Character offset mapping accuracy
  - Core function outputs with real embeddings
- **Pros**: Ensures implementation matches real model behavior, catches API assumption errors
- **Cons**: Requires model download, GPU resources, slower execution
- **When to use**: Before merging features, validating mock accuracy, debugging model integration issues
- **Run with**: `CUDA_VISIBLE_DEVICES=1 python tests/test_sparse_integration.py`

**Important findings from integration tests:**
- BGE-M3's `lexical_weights` returns `dict[str, float]` (string keys), not `dict[int, float]`
- With `use_fp16=True`, weights are `numpy.float16` types, not Python `float`
- Tokenizer may include leading spaces in token boundaries (e.g., " learning" instead of "learning")

#### 3. End-to-End Demo Scripts (`src/experiments/demo_*.py`)
- **Purpose**: Manual testing with real ChromaDB data, frontend validation
- **Example**: `src/experiments/demo_sparse_attribution_with_chroma.py`
- **Use case**: Testing full pipeline (ChromaDB → attribution → result display), generating examples for users
- **When to use**: Manual QA, generating demo outputs, testing with production-like data

#### 4. Testing Workflow
```bash
# Quick logic check (during development)
python tests/test_sparse_attribution.py

# Validate real model integration (before merge)
CUDA_VISIBLE_DEVICES=1 python tests/test_sparse_integration.py

# Manual E2E validation with ChromaDB
python src/experiments/demo_sparse_attribution_with_chroma.py --collection xingqiu_chuangye --n 3
```

**Note**: No pytest suite yet; run tests directly with `python tests/test_*.py`

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
