# Repository Guidelines

## Project Structure & Module Organization
Source lives in `src/`. The `data_pipeline` package handles CSV ingestion, TEI vectorization, samplers, and ChromaDB stores; `attribution` contains each research method (segmented, cross-encoder, ColBERT, late chunking) behind shared base classes; `experiments/` holds runnable demonstrations. Runtime scripts sit in `scripts/` (data prep, random sampler, attribution smoke-tests). Configuration YAMLs are under `config/`, while raw inputs belong in `data/` and persisted indexes in `chroma_db/`. Logs stream to `logs/` via `src/utils/logger.py`.

## Build, Test, and Development Commands
Use Python 3.9+ and create a virtual env (`python -m venv .venv && source .venv/bin/activate`). Install editable deps with `pip install -e .`. Typical workflows:
- `python scripts/prepare_data.py`: loads `config/data_prep.yaml`, validates paths, calls TEI, and writes into ChromaDB.
- `python scripts/test_segmented_attribution.py`: offline demonstration that relies on `MockVectorizer`; safe for quick regressions.
- `python -m src.experiments.test_segmented_attribution`: end-to-end run that assumes a live TEI container (see command inside the script header) and populated collections.
- `python scripts/test_random_sampler.py --collection <name> --n 5`: sanity-check retrieval quality for an existing collection.

## Coding Style & Naming Conventions
Follow the current Python style: four-space indentation, `snake_case` functions, `CamelCase` classes, and module-level docstrings summarizing responsibilities. Type hints and descriptive dataclasses (see `src/utils/config.py`) are expected. When adding logging, import `logging` and reuse the shared formatter from `src/utils/logger.py`; prefer structured messages over f-strings with secrets. Configuration defaults belong in YAML examples rather than code constants.

## Testing Guidelines
There is no pytest suite yet, so guard behavior with targeted scripts. Extend the relevant `scripts/test_*.py` or add new ones in `src/experiments/` when a feature spans multiple modules. Use `MockVectorizer` or other fakes for unit-level runs, and reserve TEI-dependent tests for scenarios where you can document the docker command, collection name, and sample output. Keep naming consistent (`test_<area>.py`) so contributors can discover runnable demos quickly.

## Commit & Pull Request Guidelines
Existing history follows short `type：summary` lines (e.g., `feat：添加初始项目结构`). Keep subjects under ~72 characters, add detail in the body if needed, and mention affected modules. PRs should describe the problem, summarize manual tests (include command snippets plus key log excerpts), and link the dataset/config used. Screenshot CLI traces only when output formatting matters. Always call out changes that require regenerating ChromaDB or restarting the TEI service.

## Security & Configuration Tips
Never commit proprietary CSVs or actual TEI endpoints—use the sanitized filenames already referenced in `config/data_prep.yaml.example`. Store secrets (API tokens, service URLs) in local environment variables and read them inside configs. When sharing logs, redact document text unless it comes from the public samples bundled in `data/`.
