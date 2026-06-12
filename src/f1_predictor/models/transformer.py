"""Permutation-equivariant cross-driver Transformer with a delta head.

Input [B, N, F] features + [B, N] driver indices + [B, N] validity mask.
Output [B, N] predicted *deltas* (places gained = current_rank - final_position),
matching the LightGBM baseline's target. No positional encoding (drivers are an
unordered set); a learned driver-identity embedding carries per-driver priors.

Padded slots are excluded from attention via ``src_key_padding_mask`` so they
cannot leak into the active drivers' predictions. The raw head output is
returned (no -1e4 fill): masking is handled by the loss (valid-slot mean) and at
scoring time (score = predicted_delta - current_rank, padded slots dropped).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DriverDeltaNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        num_drivers: int = 30,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.driver_embed = nn.Embedding(num_drivers, d_model, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LayerNorm, more stable on small data
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(
        self,
        features: torch.Tensor,
        driver_idx: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        # [B, N, d] = projected features + driver-identity embedding.
        h = self.input_proj(features) + self.driver_embed(driver_idx)
        # src_key_padding_mask: True = ignore that slot as an attention key.
        h = self.encoder(h, src_key_padding_mask=~valid)
        return self.head(h).squeeze(-1)  # [B, N] raw delta
