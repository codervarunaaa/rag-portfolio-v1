from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from rag import answer, _ensure_loaded
import rag


@asynccontextmanager
async def lifespan(app):
    try:
        _ensure_loaded()
    except Exception:
        pass
    yield


app = FastAPI(title="rag-portfolio-v1", version="1.2", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    backend: str = Field("aistudio", pattern="^(aistudio|vertex)$")
    k: int = Field(3, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return answer(req.query, backend=req.backend, k=req.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _run_reindex():
    from embed import main as run_embed
    run_embed()
    rag._DOCS = rag._VECS = None
    _ensure_loaded()


@app.post("/reindex")
async def reindex(request: Request):
    obj = ""
    try:
        body = await request.json()
        obj = body.get("name", "") if isinstance(body, dict) else ""
    except Exception:
        pass

    if obj:
        if obj.startswith("index/"):
            return {"status": "skipped", "reason": "index write", "object": obj}
        if not obj.lower().endswith(".pdf"):
            return {"status": "skipped", "reason": "not a PDF", "object": obj}

    try:
        _run_reindex()
        return {"status": "reindexed", "trigger_object": obj or "manual"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
