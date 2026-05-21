# rag-portfolio-v1

Production-style RAG pipeline on GCP (in progress).

## Status
- ✅ Day 1: GCP project + billing + budget alerts
- ✅ Day 2: GCS bucket + Artifact Registry repo
- ✅ Day 3: Local PDF ingestion + recursive chunking (pypdf + langchain-text-splitters)
- ✅ Day 4: Vertex AI `text-embedding-005` + local cosine retrieval

## Stack
Python 3.12 · uv · Vertex AI · GCS · Artifact Registry · Cloud Run (planned)

## Local run
```bash
uv sync
uv run python src/ingest.py
uv run python src/embed.py
uv run python src/search.py "What is multi-head attention?"
```

## Cost
~₹1 spent to date. Budget alerts at 50/90/100% of ₹1000.
