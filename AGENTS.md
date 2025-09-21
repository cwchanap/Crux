# Repository Guidelines

## Project Structure & Module Organization
The FastAPI service lives in `src/app`, with `main.py` exposing the API surface and helpers such as `storage.py` and `transcriber.py`. CLI utilities (e.g., checkpoint conversion) are under `src/cli`, while orchestration helpers reside in `src/worker.py`. TensorFlow checkpoints and auxiliary assets belong in `models/`, static web artifacts in `static/`, and temporary uploads or renders are routed to `temp_uploads/` and `temp_downloads/`. Tests mirror the API layout in `tests/`, so add new suites alongside the feature they exercise.

## Build, Test, and Development Commands
Install dependencies once with `uv pip install -e .` (append `--extra dev` for tooling). Start the API locally via:
```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Run the worker entry point with `uv run python src/worker.py` when processing jobs outside the request cycle. Key checks:
```bash
uv run pytest
uv run ruff check src tests
uv run black --check src tests
uv run pylint src/app src/cli
```

## Coding Style & Naming Conventions
We target Python 3.13 with 4-space indentation and a 100-character soft line limit (enforced by Ruff and Pylint). Prefer descriptive module and function names (`verb_noun` for actions, `PascalCase` for classes). Keep imports sorted (Ruff’s `I` rules) and avoid eager TensorFlow imports—follow the lazy-loading pattern already used in `src/app/main.py`. Format code with Black before committing.

## Testing Guidelines
Pytest is configured via `pytest.ini` to collect under `tests/`. Name files `test_<feature>.py` and favor small, deterministic fixtures; async endpoints should use `pytest.mark.asyncio`. Mock heavy model calls or set `PRELOAD_MODEL=0` so suites remain fast. Add regression tests whenever touching request handlers, storage adapters, or CLI flows.

## Commit & Pull Request Guidelines
Recent history leans toward Conventional Commits (`feat:`, `refactor:`, `fix:`); keep using that style and keep subjects under 72 characters. Include a concise body detailing context and follow-up steps when needed. Before opening a PR, run the full check stack above, link related issues, and attach screenshots or sample MIDI output for UI-affecting changes. Note any deviations (e.g., skipped tests) explicitly in the description.

## Deployment & Environment Notes
Cloudflare Workers remains the target edge runtime. Update `wrangler.toml` when KV namespaces or account IDs change, and expose environment variables via the deployment platform—not hardcoded constants. Docker runs use the provided `Dockerfile`; rebuild after modifying system-level dependencies.
