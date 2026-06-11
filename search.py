import json
from pathlib import Path

import numpy as np


def top_k(query_emb, doc_embeddings, k=5):
    scores = doc_embeddings @ query_emb
    ranked = np.argsort(scores)[::-1][:k]
    return ranked, scores[ranked]


def build_index(model, corpus_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError


def load_index(index_dir):
    index_dir = Path(index_dir)
    embeddings = np.load(index_dir / "embeddings.npy")
    with open(index_dir / "ids.json") as f:
        ids = json.load(f)
    return embeddings, ids
