import sys; sys.path.insert(0, "src")
from ingest import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP

def test_chunk_basic():
    text = "word " * 1000          # ~5000 chars
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= CHUNK_SIZE + 50 for c in chunks)  # small slack for separators

def test_chunk_short_text():
    chunks = chunk_text("short")
    assert chunks == ["short"]

def test_chunk_empty():
    assert chunk_text("") == []

def test_overlap_configured():
    assert CHUNK_OVERLAP < CHUNK_SIZE
    assert CHUNK_OVERLAP > 0
