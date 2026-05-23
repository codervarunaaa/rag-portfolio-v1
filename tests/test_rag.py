import sys; sys.path.insert(0, "src")
import numpy as np
from unittest.mock import patch, MagicMock
import rag

def test_build_context_format():
    hits = [
        ({"source": "a.pdf", "chunk_id": 3, "text": "hello"}, 0.91),
        ({"source": "b.pdf", "chunk_id": 7, "text": "world"}, 0.85),
    ]
    ctx = rag.build_context(hits)
    assert "a.pdf chunk 3" in ctx
    assert "hello" in ctx
    assert "0.91" in ctx
    assert "---" in ctx  # separator between chunks

def test_retrieve_ranks_by_similarity():
    # Mock the loaded store: 3 docs, 3 unit vectors
    docs = [
        {"source": "a.pdf", "chunk_id": 0, "text": "x"},
        {"source": "b.pdf", "chunk_id": 1, "text": "y"},
        {"source": "c.pdf", "chunk_id": 2, "text": "z"},
    ]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    fake_model = MagicMock()
    fake_model.get_embeddings.return_value = [MagicMock(values=[1.0, 0.0])]

    with patch.object(rag, "_ensure_loaded", return_value=(docs, vecs, fake_model)):
        hits = rag.retrieve("query", k=2)
    # query [1,0] most similar to doc a [1,0], then c [0.7,0.7]
    assert hits[0][0]["source"] == "a.pdf"
    assert hits[1][0]["source"] == "c.pdf"
    assert len(hits) == 2

def test_answer_uses_fallback_on_503():
    docs = [{"source": "a.pdf", "chunk_id": 0, "text": "ctx"}]
    vecs = np.array([[1.0, 0.0]])
    fake_model = MagicMock()
    fake_model.get_embeddings.return_value = [MagicMock(values=[1.0, 0.0])]

    with patch.object(rag, "_ensure_loaded", return_value=(docs, vecs, fake_model)), \
         patch.object(rag, "generate_aistudio", side_effect=Exception("503 UNAVAILABLE")), \
         patch.object(rag, "generate_vertex", return_value=("answer", {"in": 10, "out": 5})):
        result = rag.answer("q", backend="aistudio")
    assert result["backend"] == "vertex (fallback)"
    assert result["answer"] == "answer"
