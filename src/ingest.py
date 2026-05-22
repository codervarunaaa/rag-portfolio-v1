import os
import tempfile
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google.cloud import storage
from google.api_core.retry import Retry

BUCKET = "rag-portfolio-v1-pdfs-vaarun"
LOCAL_DIR = Path("data/pdfs")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
USE_GCS = os.getenv("USE_GCS", "1") == "1"

DOWNLOAD_RETRY = Retry(initial=1.0, maximum=10.0, multiplier=2.0, timeout=300.0)

def _splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)

def chunk_text(text: str) -> list[str]:
    return _splitter().split_text(text)

def _iter_gcs_pdfs():
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    for blob in bucket.list_blobs():
        if blob.name.lower().endswith(".pdf"):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                blob.download_to_filename(tmp.name, retry=DOWNLOAD_RETRY, timeout=120)
                yield blob.name, tmp.name

def _iter_local_pdfs():
    for pdf in sorted(LOCAL_DIR.glob("*.pdf")):
        yield pdf.name, str(pdf)

def ingest_all() -> list[dict]:
    docs = []
    source_iter = _iter_gcs_pdfs() if USE_GCS else _iter_local_pdfs()
    where = f"gs://{BUCKET}" if USE_GCS else str(LOCAL_DIR)
    print(f"Ingesting from: {where}\n")
    for name, path in source_iter:
        text = extract_text(path)
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            docs.append({"source": name, "chunk_id": i, "text": c})
        print(f"{name}: {len(text)} chars -> {len(chunks)} chunks")
        if USE_GCS:
            os.unlink(path)
    return docs

if __name__ == "__main__":
    docs = ingest_all()
    print(f"\nTotal chunks: {len(docs)}\n")
    for d in docs[:3]:
        print(f"--- {d['source']} #{d['chunk_id']} ---")
        print(d["text"][:300], "...\n")
