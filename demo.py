"""Neural search demo over the Jurafsky & Martin book corpus.

query -> encoder -> embedding -> cosine search over J&M chunks -> top-k

  python demo.py                                  # interactive, squad_v2 model
  python demo.py --model wiki                      # use the wiki checkpoint
  python demo.py --query "what is tokenization?"   # one-shot
  python demo.py --query "n-gram models" --k 3
"""

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, logging

logging.set_verbosity_error()

CORPUS = "data/processed/corpus.jsonl"
VOCAB_SIZE, MAX_LEN, PROJ = 30522, 128, 128

MODELS = {
    "squad_v2": dict(embed=256, heads=4, layers=2, ffn=512, dropout=0.3,
                     ckpt="checkpoints/squad_v2/best_model.pt"),
    "wiki":     dict(embed=256, heads=4, layers=2, ffn=512, dropout=0.3,
                     ckpt="checkpoints/wiki/best_model.pt"),
    "squad":    dict(embed=512, heads=8, layers=3, ffn=1024, dropout=0.1,
                     ckpt="checkpoints/squad/best_model.pt"),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, cfg["embed"], padding_idx=0)
        self.position_embedding = nn.Embedding(MAX_LEN, cfg["embed"])
        layer = nn.TransformerEncoderLayer(
            d_model=cfg["embed"], nhead=cfg["heads"],
            dim_feedforward=cfg["ffn"], dropout=cfg["dropout"], batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg["layers"])
        self.projection = nn.Linear(cfg["embed"], PROJ)
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, ids, mask):
        x = self.token_embedding(ids)
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        x = self.dropout(x + self.position_embedding(pos))
        x = self.transformer(x, src_key_padding_mask=(mask == 0))
        m = mask.unsqueeze(-1).float()
        x = (x * m).sum(1) / m.sum(1)
        return F.normalize(self.projection(x), dim=-1)

    @torch.no_grad()
    def encode(self, texts):
        t = tokenizer(texts, padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt").to(device)
        return self.forward(t["input_ids"], t["attention_mask"])

    @torch.no_grad()
    def encode_long(self, texts):
        # j&m chunks are 200-300 words, way more than 128 tokens. if we just
        # truncate we throw away half the chunk. instead we cut each chunk into
        # 128-token windows, encode each window, and average them. this way the
        # whole chunk gets represented, not only its beginning.
        out = []
        for text in texts:
            ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
            windows = [ids[i:i + MAX_LEN] for i in range(0, len(ids), MAX_LEN)]
            embs = []
            for w in windows:
                w = w.unsqueeze(0).to(device)
                embs.append(self.forward(w, torch.ones_like(w)))
            mean = torch.stack(embs).mean(0)
            out.append(F.normalize(mean, dim=-1))
        return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="squad_v2")
    ap.add_argument("--query", default=None, help="one-shot query (else interactive)")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    cfg = MODELS[args.model]
    model = TransformerEncoder(cfg).to(device)
    model.load_state_dict(torch.load(cfg["ckpt"], map_location=device))
    model.eval()

    docs = [json.loads(line)["document"] for line in open(CORPUS)]

    # encoding the whole book every run is slow, so cache the doc embeddings.
    # if the corpus size changed we just rebuild (cheap safety check).
    cache = f"data/processed/corpus_emb_{args.model}.npy"
    if os.path.exists(cache) and np.load(cache).shape[0] == len(docs):
        doc_emb = np.load(cache)
        print(f"[{args.model}] loaded cached index ({len(docs)} chunks)")
    else:
        print(f"[{args.model}] encoding {len(docs)} J&M chunks on {device}...")
        doc_emb = model.encode_long(docs).cpu().numpy()
        np.save(cache, doc_emb)

    def search(query):
        q_emb = model.encode([query]).cpu().numpy()[0]
        scores = doc_emb @ q_emb
        for rank, i in enumerate(np.argsort(-scores)[:args.k], 1):
            print(f"\n[{rank}] score={scores[i]:.3f}\n{docs[i][:400]}...")

    if args.query:
        search(args.query)
        return

    print("type a query (empty line to quit):")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            break
        search(query)


if __name__ == "__main__":
    main()
