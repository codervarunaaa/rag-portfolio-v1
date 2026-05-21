# rag-portfolio-v1

Production-style RAG pipeline on GCP. Built in shippable daily slices.

## Status
- ✅ **Day 1:** GCP project, billing, ₹1000 budget with 50/90/100% alerts
- ✅ **Day 2:** GCS bucket + Artifact Registry (asia-south1)
- ✅ **Day 3:** Local PDF ingestion + recursive chunking (pypdf + langchain-text-splitters)
- ✅ **Day 4:** Vertex AI `text-embedding-005` + local cosine retrieval
- ✅ **Day 5:** End-to-end RAG with Gemini 2.5 Flash (AI Studio free tier + Vertex fallback)

## Architecture (current)
