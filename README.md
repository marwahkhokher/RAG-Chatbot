# RAG Chatbot (NETSOL Internship Project)

A simple Retrieval-Augmented Generation chatbot: FastAPI backend, Qdrant vector database,
local embeddings, and Groq as the LLM — every piece runs on a free tier.

## Architecture

```
User question
    │
    ▼
Embed question (sentence-transformers, local, free)
    │
    ▼
Search Qdrant Cloud (top-k similar chunks)
    │
    ▼
Stuff chunks + question into prompt
    │
    ▼
Groq LLM (Llama 3.1) generates the answer
```

## 1. Prerequisites (all free, no credit card required)

1. **Groq API key** — sign up at https://console.groq.com and create a key under API Keys.
2. **Qdrant Cloud** — sign up at https://cloud.qdrant.io, create a free cluster (1GB),
   copy the cluster URL and API key.
3. **Render account** — sign up at https://render.com (no card needed for the free tier).
4. **GitHub account** with this repo pushed to it.

## 2. Local setup

```bash
git clone <your-repo-url>
cd rag-chatbot
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# now edit .env and paste in your GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY
```

## 3. Add your documents and ingest them

Drop your `.txt`, `.md`, or `.pdf` files into the `data/` folder (a sample file is already
there — delete it once you add your own), then run:

```bash
python -m app.ingest
```

This chunks your documents, embeds them locally, and uploads the vectors to your Qdrant
cluster. Re-run this any time you add new documents.

## 4. Run the chatbot locally

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000 for the built-in chat UI, or call the API directly:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

## 5. Run the tests

```bash
pytest -v
```

## 6. Deploy to Render

1. On https://dashboard.render.com, click **New +** → **Web Service**.
2. Connect your GitHub repo.
3. Render will detect the `render.yaml` blueprint automatically (or choose
   **Docker** as the environment manually if asked).
4. Under **Environment**, add these secrets (values from step 1):
   - `GROQ_API_KEY`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
5. Turn **auto-deploy off** in the service settings — we want deploys to be triggered
   by GitHub Actions only after tests pass, not on every raw push.
6. Go to the service's **Settings → Deploy Hook**, copy the URL it gives you.

## 7. Set up GitHub Actions (CI/CD)

In your GitHub repo, go to **Settings → Secrets and variables → Actions** and add:

| Secret name | Value |
|---|---|
| `RENDER_DEPLOY_HOOK_URL` | the deploy hook URL from step 6 above |

Two workflows are already set up:

- **`.github/workflows/ci.yml`** — runs on every push/PR: lints with `ruff` and runs
  `pytest`.
- **`.github/workflows/deploy.yml`** — runs automatically after CI succeeds on `main`,
  and calls the Render deploy hook, which rebuilds the Docker container and redeploys.

Push to `main` and watch the **Actions** tab in GitHub — CI runs first, then deploy.

## 8. Verify the deployment

Once the deploy workflow finishes, Render will build and start your container
(check the **Logs** tab on your Render service to watch it happen — the first build
takes a few minutes since it installs sentence-transformers). Your live URL will look
like:

```
https://rag-chatbot-xxxx.onrender.com
```

Visit it — you'll see the same chat UI as local, now publicly reachable. Note: on the
free tier the service sleeps after 15 minutes of no traffic, so the first request after
a quiet period takes 30-60 seconds to wake up.

## Project structure

```
rag-chatbot/
├── app/
│   ├── main.py       # FastAPI app + chat UI
│   ├── rag.py        # embeddings, Qdrant, retrieval + generation
│   ├── ingest.py      # document ingestion script
│   └── config.py      # env var loading
├── data/              # your source documents go here
├── tests/
│   └── test_api.py
├── .github/workflows/
│   ├── ci.yml
│   └── deploy.yml
├── Dockerfile
├── render.yaml
├── requirements.txt
└── .env.example
```

## Notes on the free tiers

- **Groq**: free tier has rate limits (requests/minute and tokens/minute) — fine for a
  demo/internship project, not for production traffic.
- **Qdrant Cloud free cluster**: 1GB storage, enough for tens of thousands of chunks.
- **Render free web service**: 750 free instance-hours/month (enough for one service
  running continuously), sleeps after 15 minutes of inactivity, wakes on the next request.
