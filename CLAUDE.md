# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (append --extra dev for dev tooling)
uv pip install -e .

# Run development server
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_app_endpoints.py

# Run a single test by name
uv run pytest tests/test_app_endpoints.py::test_name

# Linting and formatting checks
uv run ruff check src tests
uv run ruff format --check src tests
uv run pylint src/app src/cli

# Format code before committing
uv run ruff format src tests
```

## Architecture

This is a FastAPI service for drum audio transcription to MIDI using TensorFlow (Magenta E-GMD model), designed for a hybrid local + Cloudflare Workers deployment.

**Source layout:**
- `src/app/main.py` — FastAPI app entry point; handles all HTTP endpoints (`/api/upload`, `/api/transcribe`, `/api/jobs/*`). TensorFlow is never imported at module level — it's lazily loaded only when a transcription job runs. Jobs and MIDI data are currently stored in-memory dicts (`jobs_store`, `midi_store`).
- `src/app/transcriber.py` — `DrumTranscriber` class; audio loading via librosa, inference via the TF2 Magenta model, MIDI generation via pretty-midi.
- `src/app/tf2_magenta_model.py` — TF2-compatible reimplementation of Magenta's onsets+frames architecture (ConvStack, BiLSTM, etc).
- `src/app/storage.py` — `StorageAdapter` with pluggable backends: `"local"` (filesystem) and `"cloudflare_kv"` (Cloudflare KV REST API). Controlled by the `STORAGE_TYPE` env var.
- `src/worker.py` — Cloudflare Workers entry point (uses `js` module). It proxies static files and returns 501 for heavy API endpoints, which are meant to run on a separate GPU service.
- `src/cli/convert.py` — CLI tool for converting TF checkpoints; exposed as `convert-checkpoint` script.
- `export_to_tfjs.py` — Standalone script for exporting models to TensorFlow.js format.

**Request flow:** Upload (`POST /api/upload`) → validates file type via magic bytes + extension → stores in `temp_uploads/`. Transcription (`POST /api/transcribe`) → creates a job → runs `process_audio_task` as a FastAPI `BackgroundTask` → result stored in `midi_store` → downloadable via `GET /api/jobs/{job_id}/download`.

**Key design constraints:**
- Never import TensorFlow at module load time. Follow the lazy-import pattern in `main.py`. Set `PRELOAD_MODEL=0` in tests to keep them fast.
- File upload validation is two-layered: `UploadSizeLimitMiddleware` (checks `Content-Length` header before body is read) + per-chunk size tracking + magic byte validation. M4A files are validated against an audio-brand allowlist (`M4A_AUDIO_BRANDS`).
- CORS is configured via `CORS_ALLOWED_ORIGINS` env var (default: `localhost:4330,localhost:8788`). The size-limit middleware manually injects CORS headers into 413 responses because it runs outside `CORSMiddleware`.

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `STORAGE_TYPE` | `"local"` or `"cloudflare_kv"` | `"local"` |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account ID | — |
| `CLOUDFLARE_KV_NAMESPACE` | KV namespace ID | — |
| `CLOUDFLARE_API_TOKEN` | API token for KV access | — |
| `CORS_ALLOWED_ORIGINS` | Comma-separated allowed origins | `http://localhost:4330,http://localhost:8788` |
| `PRELOAD_MODEL` | Set to `"1"` to load TF model at startup | `"0"` |

## Code Style

- Python 3.12, 4-space indent, 100-character soft line limit (Ruff + Pylint enforced).
- `verb_noun` naming for functions, `PascalCase` for classes.
- Imports sorted by Ruff `I` rules; no eager TF imports.
- Commit style: Conventional Commits (`feat:`, `fix:`, `refactor:`, `chore:`, `test:`), subject under 72 chars.

## Testing

- `pytest.ini` configures `asyncio_mode = auto`; all async tests work without manual `@pytest.mark.asyncio`.
- Coverage runs automatically (`--cov=src`).
- Mock heavy model calls or set `PRELOAD_MODEL=0` (already the default) to keep tests fast.
- Test files follow `test_<feature>.py` naming and live in `tests/`.
