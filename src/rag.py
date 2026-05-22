import warnings; warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")
import os, sys
import numpy as np
from dotenv import load_dotenv
import vertexai
from vertexai.language_models import TextEmbeddingModel
from search import load_vectors

load_dotenv()

PROJECT = "rag-portfolio-v1"
EMBED_REGION = "asia-south1"
GEN_REGION = "us-central1"
EMBED_MODEL = "text-embedding-005"
GEN_MODEL = "gemini-2.5-flash"
TOP_K = 3

SYSTEM = (
    "You answer questions strictly from the provided context. "
    "If the context doesn't contain the answer, say so. "
    "Cite sources inline as [filename chunk N]."
)

# ---- Load-once cache ----
_DOCS = None
_VECS = None
_EMBED_MODEL = None

def _ensure_loaded():
    global _DOCS, _VECS, _EMBED_MODEL
    if _DOCS is None:
        vertexai.init(project=PROJECT, location=EMBED_REGION)
        _EMBED_MODEL = TextEmbeddingModel.from_pretrained(EMBED_MODEL)
        _DOCS, _VECS = load_vectors()
    return _DOCS, _VECS, _EMBED_MODEL

def retrieve(query: str, k: int = TOP_K):
    docs, vecs, model = _ensure_loaded()
    q_vec = np.array(model.get_embeddings([query])[0].values)
    sims = vecs @ q_vec
    top = np.argsort(-sims)[:k]
    return [(docs[i], float(sims[i])) for i in top]

def build_context(hits):
    return "\n\n---\n\n".join(
        f"[{d['source']} chunk {d['chunk_id']}] (score={s:.3f})\n{d['text']}"
        for d, s in hits
    )

def generate_aistudio(prompt: str):
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(model=GEN_MODEL, contents=prompt)
    u = resp.usage_metadata
    return resp.text, {"in": u.prompt_token_count, "out": u.candidates_token_count}

def generate_vertex(prompt: str):
    from vertexai.generative_models import GenerativeModel
    vertexai.init(project=PROJECT, location=GEN_REGION)
    model = GenerativeModel(GEN_MODEL)
    resp = model.generate_content(
        [prompt],
        generation_config={"max_output_tokens": 1024, "temperature": 0.2},
    )
    u = resp.usage_metadata
    return resp.text, {"in": u.prompt_token_count, "out": u.candidates_token_count}

def answer(query: str, backend: str = "aistudio", k: int = TOP_K):
    """Returns a dict — used by both CLI and API."""
    hits = retrieve(query, k)
    context = build_context(hits)
    prompt = f"{SYSTEM}\n\nContext:\n{context}\n\nQuestion: {query}"
    used = backend
    try:
        gen = generate_aistudio if backend == "aistudio" else generate_vertex
        text, tokens = gen(prompt)
    except Exception as e:
        if backend == "aistudio" and "503" in str(e):
            text, tokens = generate_vertex(prompt)
            used = "vertex (fallback)"
        else:
            raise
    return {
        "query": query,
        "answer": text,
        "backend": used,
        "sources": [
            {"source": d["source"], "chunk_id": d["chunk_id"], "score": round(s, 3)}
            for d, s in hits
        ],
        "tokens": tokens,
    }

if __name__ == "__main__":
    args = sys.argv[1:]
    backend = "aistudio"
    if args and args[0] in ("--vertex", "--aistudio"):
        backend = args.pop(0).lstrip("-")
    query = " ".join(args) or "What is multi-head attention?"
    r = answer(query, backend)
    print(f"\nQ: {r['query']}\nBackend: {r['backend']}\n")
    print("Retrieved:")
    for s in r["sources"]:
        print(f"  - {s['source']} chunk {s['chunk_id']} (score={s['score']})")
    print(f"\nA: {r['answer']}\n")
    print(f"Tokens: in={r['tokens']['in']}, out={r['tokens']['out']}")
