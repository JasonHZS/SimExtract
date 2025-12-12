# SimExtract Agent Brief

## Project Overview
- SimExtract (Similarity Attribution Research Platform) identifies the keywords or spans inside text B that best explain its semantic similarity to text A.
- The goal is to study and compare attribution methods, ensuring every workflow (retrieval, sampling, experiments) can surface fine-grained and interpretable evidence.

## Core Objective
- When A and B are highly similar, extract the span set in B that contributes the most to that similarity and surface it in an explainable format.
- Attribution signals should clarify model decisions, improve vectorization setups, and guide how ChromaDB collections are constructed and tuned.

## Illustrative Example
- Text A: “AI is transforming the healthcare industry.”
- Text B: “Machine learning for medical diagnosis is revolutionizing patient care, although the weather is nice.”
- Expected attribution: “Machine learning for medical diagnosis”

## Research Methods

### 1. Segmented Vectorization
- Split text B into sentences, sliding windows, or semantic units, then compute the vector similarity between each segment and text A to score contributions.
- **Advantages**: simple and efficient; **Status**: scaffolding in place, implementation pending.

### 2. Cross-Encoder Attention Analysis
- Run a BERT Cross-Encoder over `[CLS] A [SEP] B [SEP]`, analyze attention weights from A tokens to B tokens, and turn them into attribution scores.
- **Advantages**: captures deep interactions with solid theoretical grounding; **Status**: scaffolding in place, implementation pending.

### 3. Token-wise (Sparse / Late Interaction)
- Produce token-level weights or embeddings and evaluate each token’s contribution through sparse signals or MaxSim-style late interaction.
- **Implementation cues**:
  - bge-m3 emits lexical weights (BM25-like sparse scores) together with dense vectors for token-level contributions.
  - ColBERT exposes per-token multi-vectors that can be aligned manually to build contribution matrices.
- **Advantages**: fine-grained token view that keeps context; **Status**: scaffolding in place, implementation pending.

### 4. Late Chunking
- Compute document-level embeddings first, then perform intelligent segmentation and aggregation inside the embedding space to locate high-value spans.
- **Advantages**: preserves global context and flexible semantic chunking; **Status**: scaffolding in place, implementation pending.

## Current Focus
- All four methods share the same milestone: complete the missing pieces, connect the data pipeline end-to-end, and compare experimental outcomes.
- Every new feature or experiment should map back to the README goals and example to keep the research “interpretable, fine-grained, reproducible.”

## Repository Guidelines

### Project Structure & Module Organization
Source lives in `src/`. The `data_pipeline` package handles CSV ingestion, TEI vectorization, samplers, and ChromaDB stores; `attribution` contains each research method (segmented, cross-encoder, ColBERT, late chunking) behind shared base classes; `experiments/` holds runnable demonstrations. Runtime scripts sit in `scripts/` (data prep, random sampler, attribution smoke-tests). Configuration YAMLs are under `config/`, while raw inputs belong in `data/` and persisted indexes in `chroma_db/`. Logs stream to `logs/` via `src/utils/logger.py`.

### Build, Test, and Development Commands
Use Python 3.9+ and create a virtual environment (`python -m venv .venv && source .venv/bin/activate`). Install editable dependencies with `pip install -e .`. Typical workflows:
- `python scripts/prepare_data.py`: loads `config/data_prep.yaml`, validates paths, calls TEI, and writes into ChromaDB.
- `python scripts/test_segmented_attribution.py`: offline demonstration that relies on `MockVectorizer`; safe for quick regressions.
- `python -m src.experiments.test_segmented_attribution`: end-to-end run that assumes a live TEI container (see command inside the script header) and populated collections.
- `python scripts/test_random_sampler.py --collection <name> --n 5`: sanity-check retrieval quality for an existing collection.

### Coding Style & Naming Conventions
Follow the current Python style: four-space indentation, `snake_case` functions, `CamelCase` classes, and module-level docstrings summarizing responsibilities. Type hints and descriptive dataclasses (see `src/utils/config.py`) are expected. When adding logging, import `logging` and reuse the shared formatter from `src/utils/logger.py`; prefer structured messages over f-strings with secrets. Configuration defaults belong in YAML examples rather than code constants.

### Testing Guidelines
There is no pytest suite yet, so guard behavior with targeted scripts. Extend the relevant `scripts/test_*.py` or add new ones in `src/experiments/` when a feature spans multiple modules. Use `MockVectorizer` or other fakes for unit-level runs, and reserve TEI-dependent tests for scenarios where you can document the docker command, collection name, and sample output. Keep naming consistent (`test_<area>.py`) so contributors can discover runnable demos quickly.

### Commit & Pull Request Guidelines
Existing history follows short `type: summary` lines (for example, `feat: add initial project structure`). Keep subjects under ~72 characters, add detail in the body if needed, and mention affected modules. PRs should describe the problem, summarize manual tests (with command snippets plus key log excerpts), and link the dataset/config used. Screenshot CLI traces only when output formatting matters. Always call out changes that require regenerating ChromaDB or restarting the TEI service.

### Security & Configuration Tips
Never commit proprietary CSVs or actual TEI endpoints—use the sanitized filenames already referenced in `config/data_prep.yaml.example`. Store secrets (API tokens, service URLs) in local environment variables and read them inside configs. When sharing logs, redact document text unless it comes from the public samples bundled in `data/`.
