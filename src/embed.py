import json
from pathlib import Path
from vertexai.language_models import TextEmbeddingModel
import vertexai
from ingest import ingest_all

PROJECT = "rag-portfolio-v1"
REGION = "asia-south1"
MODEL_NAME = "text-embedding-005"
OUT_FILE = Path("data/vectors.jsonl")
BATCH = 50  # API max is 250; 50 keeps us safe

def main():
    vertexai.init(project=PROJECT, location=REGION)
    model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
    docs = ingest_all()
    print(f"\nEmbedding {len(docs)} chunks...")

    OUT_FILE.parent.mkdir(exist_ok=True)
    with OUT_FILE.open("w") as f:
        for i in range(0, len(docs), BATCH):
            batch = docs[i:i+BATCH]
            texts = [d["text"] for d in batch]
            embs = model.get_embeddings(texts)
            for d, e in zip(batch, embs):
                d["embedding"] = e.values
                f.write(json.dumps(d) + "\n")
            print(f"  {i+len(batch)}/{len(docs)}")
    print(f"\nSaved -> {OUT_FILE}")

if __name__ == "__main__":
    main()
