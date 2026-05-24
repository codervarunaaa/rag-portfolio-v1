import warnings; warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")
import json
import tempfile
from pathlib import Path
import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.cloud import storage
from ingest import ingest_all

PROJECT = "rag-portfolio-v1"
REGION = "asia-south1"
MODEL_NAME = "text-embedding-005"
BUCKET = "rag-portfolio-v1-pdfs-vaarun"
VECTORS_BLOB = "index/vectors.jsonl"     # lives in GCS, NOT in incoming/
BATCH = 50

def embed_docs(docs):
    vertexai.init(project=PROJECT, location=REGION)
    model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
    out = []
    for i in range(0, len(docs), BATCH):
        batch = docs[i:i+BATCH]
        embs = model.get_embeddings([d["text"] for d in batch])
        for d, e in zip(batch, embs):
            d = dict(d); d["embedding"] = e.values
            out.append(d)
        print(f"  {i+len(batch)}/{len(docs)}")
    return out

def write_vectors_to_gcs(docs):
    client = storage.Client()
    blob = client.bucket(BUCKET).blob(VECTORS_BLOB)
    payload = "\n".join(json.dumps(d) for d in docs)
    blob.upload_from_string(payload, content_type="application/x-ndjson")
    print(f"Wrote {len(docs)} vectors -> gs://{BUCKET}/{VECTORS_BLOB}")

def main():
    docs = ingest_all()
    print(f"\nEmbedding {len(docs)} chunks...")
    vectors = embed_docs(docs)
    write_vectors_to_gcs(vectors)

if __name__ == "__main__":
    main()
