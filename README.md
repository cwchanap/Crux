# Drum Transcription API

A FastAPI-based service for transcribing drum audio to MIDI using TensorFlow 2.x, optimized for Cloudflare Workers deployment.

## Features

- 🎵 Upload MP3/WAV/M4A/FLAC audio files
- 🥁 Automatic drum transcription to MIDI
- 📊 Real-time job status tracking
- 💾 MIDI file download
- 🚀 Cloudflare Workers ready
- 🎨 Modern web UI

## Quick Start

### Local Development

1. Install dependencies using `uv`:
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -e .
```

2. Run the development server:
```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. Open http://localhost:8000 in your browser

## API Endpoints

### Upload Audio for Transcription
```
POST /api/transcribe
Content-Type: multipart/form-data
Body: file (audio file)

Response:
{
  "job_id": "uuid",
  "message": "Job created successfully",
  "status_url": "/api/jobs/{job_id}"
}
```

### Check Job Status
```
GET /api/jobs/{job_id}

Response:
{
  "job_id": "uuid",
  "status": "pending|processing|completed|failed",
  "progress": 0-100,
  "result_url": "/api/jobs/{job_id}/download",
  "error": null
}
```

### Download MIDI Result
```
GET /api/jobs/{job_id}/download

Response: MIDI file download
```

### List All Jobs
```
GET /api/jobs?limit=10&offset=0

Response:
{
  "total": 100,
  "jobs": [...],
  "limit": 10,
  "offset": 0
}
```

## Architecture

The system uses a hybrid architecture:

1. **API Layer** (Cloudflare Workers)
   - Handles HTTP requests
   - Manages job queue
   - Serves static files

2. **Processing Layer** (Separate Service)
   - Runs TensorFlow model
   - Processes audio files
   - Generates MIDI output

3. **Storage Layer** (Cloudflare KV/D1)
   - Stores job metadata
   - Caches MIDI results

## Deployment to Cloudflare Workers

1. Install Wrangler CLI:
```bash
npm install -g wrangler
```

2. Configure your Cloudflare account:
```bash
wrangler login
```

3. Create KV namespace:
```bash
wrangler kv:namespace create "JOBS_KV"
```

4. Update `wrangler.toml` with your KV namespace ID

5. Deploy:
```bash
wrangler publish
```

## Configuration

### Environment Variables

- `STORAGE_TYPE`: "local" or "cloudflare_kv" (default: "local")
- `CLOUDFLARE_ACCOUNT_ID`: Your Cloudflare account ID
- `CLOUDFLARE_KV_NAMESPACE`: KV namespace ID
- `CLOUDFLARE_API_TOKEN`: API token for KV access

### Model Configuration

The drum transcription uses a simplified onset detection approach with:
- Sample rate: 16kHz
- Mel-spectrogram features: 229 bands
- Frequency range: 30-8000 Hz

## Development

### Running Tests
```bash
uv run pytest
```

### Code Formatting
```bash
uv run black app/
uv run ruff app/
```

## Technical Stack

- **FastAPI**: Modern web framework
- **TensorFlow 2.x**: ML framework
- **Librosa**: Audio processing
- **Pretty-MIDI**: MIDI generation
- **Cloudflare Workers**: Edge deployment
- **UV**: Package management

## Limitations

- Maximum file size: 50MB
- Supported formats: MP3, WAV, M4A, FLAC
- Processing time: ~10-30 seconds per minute of audio

## License

MIT
