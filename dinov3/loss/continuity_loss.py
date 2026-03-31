# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ContinuityNTXentLoss(nn.Module):
    def __init__(self, *, temperature: float = 0.1, normalize: bool = True):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, features: torch.Tensor, pair_indices: torch.Tensor):
        if features.ndim != 2:
            raise ValueError(f"expected features to have shape [B, D], got {tuple(features.shape)}")
        batch_size = features.shape[0]
        if batch_size < 2 or batch_size % 2 != 0:
            raise ValueError(f"expected an even batch size >= 2, got {batch_size}")
        if pair_indices.shape != (batch_size,):
            raise ValueError(
                f"expected pair_indices to have shape [{batch_size}], got {tuple(pair_indices.shape)}"
            )

        pair_indices = pair_indices.to(device=features.device, dtype=torch.long)
        proj = F.normalize(features, dim=-1) if self.normalize else features
        sim = proj @ proj.t()
        logits = sim / self.temperature
        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(eye_mask, float("-inf"))
        loss = F.cross_entropy(logits, pair_indices)

        with torch.no_grad():
            row_ids = torch.arange(batch_size, device=features.device)
            pos_cosine = sim[row_ids, pair_indices].mean()
            neg_mask = ~eye_mask
            neg_mask[row_ids, pair_indices] = False
            if neg_mask.any():
                neg_cosine = sim[neg_mask].mean()
            else:
                neg_cosine = sim.new_tensor(0.0)
            stats = {
                "continuity_num_pairs": sim.new_tensor(batch_size // 2, dtype=torch.float32),
                "continuity_pos_cosine_mean": pos_cosine,
                "continuity_neg_cosine_mean": neg_cosine,
            }
        return loss, stats
