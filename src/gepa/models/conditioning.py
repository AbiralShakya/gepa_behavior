from __future__ import annotations

from typing import List
import torch
import torch.nn as nn


class TextConditioner(nn.Module):
    """Lightweight text embedding module for prompt conditioning.

    Uses a hashing-based vocabulary to avoid external dependencies.
    """

    def __init__(self, embedding_dim: int = 128, vocab_size: int = 5000, n_layers: int = 2, nhead: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Identity()

    @staticmethod
    def _tokenize(text: str) -> List[int]:
        # Simple whitespace tokenization + hashing into fixed vocab
        tokens = text.lower().strip().split()
        return [abs(hash(tok)) % 5000 for tok in tokens] or [0]

    def forward(self, prompt_texts: List[str]) -> torch.Tensor:
        # Returns [B, D]
        max_len = 32
        batch_ids = []
        for t in prompt_texts:
            ids = self._tokenize(t)[:max_len]
            # pad
            ids = ids + [0] * (max_len - len(ids))
            batch_ids.append(ids)
        x = torch.tensor(batch_ids, dtype=torch.long, device=self.embedding.weight.device)
        emb = self.embedding(x)
        h = self.encoder(emb)
        return h[:, 0, :]  # CLS-style pooling
