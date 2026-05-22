import os, sys
import numpy as np
from dotenv import load_dotenv
import vertexai
from vertexai.language_models import TextEmbeddingModel
from search import load_vectors

load_dotenv()

PROJECT = "rag-portfolio-v1"
EMBED_REGION = "asia-south1"      # embeddings work fine in Mumbai
GEN_REGION = "us-central1"        # Gemini gen models live here
EMBED_MODEL = "text-embedding-005"
GEN_MODEL = "gemini-2.5-flash"    # try -flash first; fall back to -flash-lite if needed
TOP_K = 3

SYSTEM = (
    "You answer questions strictly from the provided context. "
    "If the context doesn't contain the answer, say so. "
    "Cite sources inline as [filename chunk N]."
)

def retrieve(query: str, k: int = TOP_K):
    vertexai.init(project=PROJECT, location=EMBED_REGION)
    model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)
    q_vec = np.array(model.get_embeddings([query])[0].values)
    docs, vecs = load_vectors()
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

def ask(query: str, backend: str = "aistudio"):
    hits = retrieve(query)
    context = build_context(hits)
    prompt = f"{SYSTEM}\n\nContext:\n{context}\n\nQuestion: {query}"
    try:
        gen = generate_aistudio if backend == "aistudio" else generate_vertex
        answer, tokens = gen(prompt)
    except Exception as e:
        msg = str(e)
        if backend == "aistudio" and "503" in msg:
            print("⚠️ AI Studio 503 — falling back to Vertex")
            answer, tokens = generate_vertex(prompt)
            backend = "vertex (fallback)"
        else:
            raise

    print(f"\nQ: {query}\nBackend: {backend}\n")
    print("Retrieved:")
    for d, s in hits:
        print(f"  - {d['source']} chunk {d['chunk_id']} (score={s:.3f})")
    print(f"\nA: {answer}\n")
    print(f"Tokens: in={tokens['in']}, out={tokens['out']}")

if __name__ == "__main__":
    args = sys.argv[1:]
    backend = "aistudio"
    if args and args[0] in ("--vertex", "--aistudio"):
        backend = args.pop(0).lstrip("-")
    query = " ".join(args) or "What is multi-head attention?"
    ask(query, backend)
