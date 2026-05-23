from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag import answer, _ensure_loaded

@asynccontextmanager
async def lifespan(app):
    _ensure_loaded()  # load vectors + model once at boot
    yield

app = FastAPI(title="rag-portfolio-v1", version="1.0", lifespan=lifespan)

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    backend: str = Field("aistudio", pattern="^(aistudio|vertex)$")
    k: int = Field(3, ge=1, le=10)

@app.on_event("startup")
def warmup():
    # Load vectors + embedding model once at boot
    _ensure_loaded()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest):
    try:
        return answer(req.query, backend=req.backend, k=req.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
