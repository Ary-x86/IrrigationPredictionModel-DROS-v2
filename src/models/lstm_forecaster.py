"""Track B forecaster: 2-layer LSTM with line-id embedding and 5-horizon head.

Architecture (per approved plan):
- line-id embedding (n_lines, 8)
- 2-layer LSTM, 64 hidden units each, dropout 0.2 between layers
- last-step hidden state concatenated with embedding
- linear head -> 5 regression outputs (h=1,3,6,12,24)

Backing: Hamdaoui 2024 (PRISMA review of SM forecasters flags 2-layer LSTM
as the sweet spot on data-scarce tabular time-series); Dhanke 2025 DLISA.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LSTMConfig:
    input_dim: int
    n_lines: int = 3
    line_embed_dim: int = 8
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    horizons: int = 5


class LSTMForecaster(nn.Module):
    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.line_embed = nn.Embedding(cfg.n_lines, cfg.line_embed_dim)
        self.lstm = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.line_embed_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.horizons),
        )

    def forward(self, seq: torch.Tensor, line_id: torch.Tensor) -> torch.Tensor:
        # seq: (B, T, F), line_id: (B,)
        out, _ = self.lstm(seq)            # (B, T, H)
        last = out[:, -1, :]               # (B, H)
        emb = self.line_embed(line_id)     # (B, E)
        h = torch.cat([last, emb], dim=-1) # (B, H+E)
        return self.head(h)                # (B, horizons)
