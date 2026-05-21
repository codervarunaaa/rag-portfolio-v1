import json, sys
import numpy as np
import vertexai
from vertexai.language_models import TextEmbeddingModel

PROJECT = "rag-portfolio-v1"
REGION = "asia-south1"
MODEL_NAME = "text-embedding-005"

def load_vectors(path="data/vectors.jsonl"):
    docs, vecs = [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            vecs.append(d.pop("embedding"))
            docs.append(d)
    return docs, np.array(vecs)

def search(query: str, k: int = 3):
    vertexai.init(project=PROJECT, location=REGION)
    model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
    q_vec = np.array(model.get_embeddings([query])[0].values)
    docs, vecs = load_vectors()
    # cosine sim (vectors are already unit-norm from Vertex)
    sims = vecs @ q_vec
    top = np.argsort(-sims)[:k]
    for rank, idx in enumerate(top, 1):
        print(f"\n#{rank} score={sims[idx]:.3f} [{docs[idx]['source']} chunk {docs[idx]['chunk_id']}]")
        print(docs[idx]["text"][:300], "...")

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is multi-head attention?"
    print(f"Query: {q}")
    search(q)
