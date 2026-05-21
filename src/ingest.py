from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_DIR = Path("data/pdfs")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)

def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)

def ingest_all() -> list[dict]:
    docs = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        text = extract_text(pdf)
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            docs.append({"source": pdf.name, "chunk_id": i, "text": c})
        print(f"{pdf.name}: {len(text)} chars -> {len(chunks)} chunks")
    return docs

if __name__ == "__main__":
    docs = ingest_all()
    print(f"\nTotal chunks: {len(docs)}\n")
    for d in docs[:3]:
        print(f"--- {d['source']} #{d['chunk_id']} ---")
        print(d["text"][:300], "...\n")
