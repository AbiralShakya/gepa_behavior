from __future__ import annotations

from typing import Optional, List
import torch
import torch.nn as nn

from .conditioning import TextConditioner


class BehaviorTransformer(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, d_model: int = 256, nhead: int = 8, num_layers: int = 4, prompt_embed_dim: int = 128, use_prompt: bool = True):
        super().__init__()
        self.use_prompt = use_prompt
        self.prompt_conditioner = TextConditioner(embedding_dim=prompt_embed_dim) if use_prompt else None
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, action_dim)
        self.prompt_proj = nn.Linear(prompt_embed_dim, d_model) if use_prompt else None

    def forward(self, x: torch.Tensor, prompts: Optional[List[str]] = None) -> torch.Tensor:
        # x: [B, T, D]
        z = self.input_proj(x)
        if self.use_prompt and prompts is not None:
            with torch.no_grad():
                p = self.prompt_conditioner(prompts)  # [B, P]
            p = self.prompt_proj(p).unsqueeze(1)  # [B, 1, d_model]
            z = torch.cat([p, z], dim=1)  # prepend prompt token
        h = self.encoder(z)
        return self.head(h[:, -1, :])
