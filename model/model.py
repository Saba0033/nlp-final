import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(self, backbone, embedding_dim):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.backbone = AutoModel.from_pretrained(backbone)
        hidden = self.backbone.config.hidden_size
        self.projection = nn.Linear(hidden, embedding_dim)

    def encode(self, texts):
        raise NotImplementedError

    def forward(self, texts):
        return self.encode(texts)


def infonce_loss(query_emb, doc_emb, temperature=0.05):
    logits = query_emb @ doc_emb.T / temperature
    labels = torch.arange(len(query_emb), device=query_emb.device)
    return F.cross_entropy(logits, labels)
