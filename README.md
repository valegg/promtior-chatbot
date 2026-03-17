# Promtior Chatbot

A production-ready RAG (Retrieval Augmented Generation) chatbot that answers questions about [Promtior](https://promtior.ai) using LangChain, LangServe, FAISS, and OpenAI.

## Features

- Scrapes live content from https://promtior.ai
- Optionally ingests a local PDF (`data/promtior.pdf`)
- Stores embeddings in a local FAISS vector index (persisted to disk)
- Serves a clean dark-themed chat UI at `/`
- Exposes a LangServe endpoint at `/chat/invoke`
- Ready to deploy on Railway via Docker

## Quick Start

```bash
git clone <repo-url>
cd promtior-chatbot

cp .env.example .env
# Set OPENAI_API_KEY in .env

pip install -r requirements.txt

# Optional: add the Promtior PDF
cp /path/to/promtior.pdf data/promtior.pdf

uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000.

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key (required) |

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Chat UI (HTML) |
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/chat/invoke` | LangServe RAG chain invocation |

### Example API call

```bash
curl -X POST http://localhost:8000/chat/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"question": "What services does Promtior offer?"}}'
```

Response:

```json
{
  "output": "Promtior offers AI consulting services including ...",
  "metadata": { ... }
}
```

## Project Structure

```
promtior-chatbot/
├── app/
│   ├── main.py          # FastAPI app + LangServe route
│   ├── chain.py         # RAG chain (LCEL)
│   ├── ingest.py        # Document loading + FAISS indexing
│   └── static/
│       └── index.html   # Chat UI
├── data/
│   └── .gitkeep         # Place promtior.pdf here
├── vectorstore/         # FAISS index (auto-generated, gitignored)
├── doc/
│   └── overview.md      # Architecture documentation
├── Dockerfile
├── railway.json
├── requirements.txt
└── .env.example
```

## Deploy on Railway

1. Push the repository to GitHub (make sure `.env` is in `.gitignore` and NOT committed)

2. Go to https://railway.app and create a new project from your GitHub repo

3. In Railway dashboard → your service → **Variables** tab, add the following secret:

   ```
   OPENAI_API_KEY = your-actual-openai-api-key
   ```

4. Railway will auto-detect the `Dockerfile` and deploy

5. Once deployed, go to **Settings → Networking → Generate Domain** to get your public URL

6. Test the deployment:
   ```bash
   curl https://your-app.railway.app/health
   ```

### Local development

1. Clone the repo
2. Copy `.env.example` to `.env` and fill in your API key
3. Optionally add `data/promtior.pdf`
4. Run with Docker Compose:
   ```bash
   docker compose up -d --build
   ```
   Or without Docker:
   ```bash
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```

See [doc/overview.md](doc/overview.md) for a full architecture diagram and detailed explanation.

## Tech Stack

- **Python 3.11**
- **FastAPI** + **LangServe** — API layer
- **LangChain** — RAG orchestration (LCEL)
- **OpenAI** — `gpt-3.5-turbo` (LLM) + `text-embedding-ada-002` (embeddings)
- **FAISS** — local vector store
- **BeautifulSoup4** — web scraping
- **PyPDF** — PDF parsing
