import json
import numpy as np
from google.cloud import storage
from google.api_core.retry import Retry

BUCKET = "rag-portfolio-v1-pdfs-vaarun"
VECTORS_BLOB = "index/vectors.jsonl"
DL_RETRY = Retry(initial=1.0, maximum=10.0, multiplier=2.0, timeout=300.0)

def load_vectors():
    client = storage.Client()
    blob = client.bucket(BUCKET).blob(VECTORS_BLOB)
    text = blob.download_as_text(retry=DL_RETRY, timeout=120)
    docs, vecs = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        vecs.append(d.pop("embedding"))
        docs.append(d)
    return docs, np.array(vecs)
