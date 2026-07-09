---
title: GPT-2 Style Emulation
emoji: 📰
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Style Emulation Project

A FastAPI backend serving fine-tuned GPT-2 style-emulation checkpoints
(generation + StyleSim scoring), with a Gradio UI as a thin HTTP client
in front of it. Ships as a single Docker container for Hugging Face
Spaces, or as two containers via `docker-compose.yml` for local dev.

## Architecture

```
train_and_push.py (separate, run once, GPU) --> HF Hub model repo
                                                       |
                                                       v
                          FastAPI (app/) <-- downloads checkpoints on demand,
                            |    ^            LRU-caches up to N resident in memory
                            |    |
                       HTTP calls only
                            |
                       Gradio UI (ui/) -- no model state of its own
```

The training script referenced here is the one from the earlier phase
of this project (fine-tunes 12 GPT-2-small checkpoints and pushes them,
plus `results.json` / `reference_texts.json`, to a Hub model repo). This
repo picks up from there and only *serves* those artifacts.

## Prerequisites

- A Hugging Face model repo already populated by `train_and_push.py`
  (12 checkpoint subfolders + `reference_texts.json` + `results.json`)
- Docker + Docker Compose, for local dev
- A Hugging Face account + Space, for deployment

## Local development

```bash
cp .env.example .env
# edit .env: set MODEL_REPO to your populated Hub repo, and HF_TOKEN if it's private

docker compose up --build
```

- API: http://localhost:8000/docs (interactive OpenAPI docs)
- UI: http://localhost:7860

Running as two containers means you can iterate on the API alone:
```bash
docker compose up api
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"topic": "Sports", "size": 100, "n_samples": 3}'
```

## Running tests

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest tests/ -v -m "not integration"   # fast, no real Hub/model downloads
pytest tests/ -v                        # full suite, needs network
```

`test_api.py` exercises the full request/response contract against a
fake in-memory model (see `tests/conftest.py`) -- no real checkpoints
needed. `test_ml.py` checks generation/scoring shapes and types against
a tiny untrained GPT-2 config, so it doesn't depend on your fine-tuned
weights either. Neither test suite validates *output quality* -- that's
what the notebook's StyleSim analysis is for.

## Deploying to Hugging Face Spaces

1. Create a Space: **Docker** SDK, any hardware tier (CPU basic works).
2. In the Space's **Settings → Variables and secrets**, set:
   - `MODEL_REPO` = your populated Hub repo id
   - `HF_TOKEN` = a read-scoped token, if that repo is private
   - Any other values from `.env.example` you want to override
3. Push this repo's contents to the Space (git remote, or the CI/CD
   pipeline below).
4. The Space builds the single Dockerfile, which runs `entrypoint.sh`:
   FastAPI starts on an internal port, Gradio starts on the exposed
   port (7860) once the API's `/health` check passes.

### CI/CD (`.github/workflows/deploy.yml`)

On every push to `main`:
1. Runs the fast (non-integration) test suite.
2. If tests pass, force-pushes this repo to your HF Space's git remote.

Requires two things set in your GitHub repo settings:
- **Secret** `HF_TOKEN` — a write-scoped Hugging Face token
- **Variable** `HF_SPACE_ID` — e.g. `your-username/gpt2-style-emulation-demo`

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/generate` | Generate N samples from a (topic, size) checkpoint |
| POST | `/api/v1/evaluate` | Score arbitrary texts against a topic's held-out references |
| POST | `/api/v1/drift` | Run the 5-hop self-conditioned drift trace (Eq. 4) live |
| GET | `/api/v1/health` | Liveness + which checkpoints are currently resident in memory |

Full request/response schemas: `app/api/schemas.py`, or just visit `/docs`.

## Known limitations (documented, not hidden)

- **Rate limiting is in-memory** (via `slowapi`), which is correct for
  this single-container deployment but won't coordinate across multiple
  replicas or workers. Swap in a Redis-backed limiter if you ever scale
  beyond one process.
- **Model cache is process-local**, capped at `MAX_RESIDENT_MODELS`
  (default 3) via LRU eviction. Raise it if your Space has enough RAM
  to hold more of the 12 checkpoints resident at once.
- **CPU-only.** GPT-2-small inference on CPU is workable for a demo but
  not built for high concurrency. If you need GPU, swap the Dockerfile's
  base image and torch install for a CUDA build, and pick a GPU-tier
  Space or your own server.
