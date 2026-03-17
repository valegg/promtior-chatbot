# Promtior Chatbot

A production-ready RAG (Retrieval Augmented Generation) chatbot that answers questions about [Promtior](https://promtior.ai) using LangChain, FAISS, and OpenAI.

## Live Demo

**[https://promtior-chatbot-production-e196.up.railway.app](https://promtior-chatbot-production-e196.up.railway.app)**

```bash
# Health check
curl https://promtior-chatbot-production-e196.up.railway.app/health

# Ask a question
curl -X POST https://promtior-chatbot-production-e196.up.railway.app/chat/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"question": "What services does Promtior offer?"}}'
```

## Features

- Scrapes live content from https://promtior.ai
- Ingests a local PDF (`data/promtior.pdf`)
- Stores embeddings in a local FAISS vector index (persisted to disk)
- Serves a clean dark-themed chat UI at `/`
- Exposes a RAG endpoint at `POST /chat/invoke`
- Deployed on Railway via Docker + GitHub Actions CI/CD

## Quick Start

```bash
git clone https://github.com/valegg/promtior-chatbot.git
cd promtior-chatbot

cp .env.example .env
# Set OPENAI_API_KEY in .env

pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000.

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key (required) |
| `PORT` | Port to bind (default: 8000) |
| `PDF_URL` | Optional public URL to download the PDF from on startup |

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Chat UI (HTML) |
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/chat/invoke` | RAG chain invocation |

### Example API call

```bash
curl -X POST http://localhost:8000/chat/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"question": "What services does Promtior offer?"}}'
```

Response:

```json
{
  "output": "Promtior offers AI consulting services including ..."
}
```

## Project Structure

```
promtior-chatbot/
├── .github/
│   └── workflows/
│       └── deploy.yml   # GitHub Actions CI/CD
├── app/
│   ├── main.py          # FastAPI app + lifespan + endpoints
│   ├── chain.py         # RAG chain (LCEL)
│   ├── ingest.py        # Document loading + FAISS indexing
│   └── static/
│       └── index.html   # Chat UI
├── data/
│   └── promtior.pdf     # Promtior documentation
├── vectorstore/         # FAISS index (auto-generated, gitignored)
├── doc/
│   └── overview.md      # Architecture documentation
├── Dockerfile
├── docker-compose.yml
├── railway.json
├── requirements.txt
└── .env.example
```

## Deploy on Railway

1. Push the repository to GitHub (`.env` must be gitignored)
2. Go to https://railway.app → **New Project → Deploy from GitHub repo**
3. In **Variables** tab, add:
   ```
   OPENAI_API_KEY = your-actual-openai-api-key
   PORT = 8000
   ```
4. Railway auto-detects the `Dockerfile` and deploys
5. Go to **Settings → Networking → Generate Domain** to get the public URL

### CI/CD with GitHub Actions

Every push to `main` triggers an automatic deploy to Railway and runs a health check.

Required GitHub secrets:

| Secret | Description |
|---|---|
| `RAILWAY_TOKEN` | Railway API token (Account → Tokens) |
| `RAILWAY_PUBLIC_URL` | Your Railway public URL |

## Tech Stack

- **Python 3.11**
- **FastAPI** + **LangServe** — API layer
- **LangChain** — RAG orchestration (LCEL)
- **OpenAI** — `gpt-3.5-turbo` (LLM) + `text-embedding-ada-002` (embeddings)
- **FAISS** — local vector store
- **BeautifulSoup4** — web scraping
- **PyPDF** — PDF parsing
- **Railway** — deployment platform
- **GitHub Actions** — CI/CD

See [doc/overview.md](doc/overview.md) for full architecture documentation.
