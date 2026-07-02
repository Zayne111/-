from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def boxes_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return torch.stack([cx, cy, w, h], dim=-1)


def normalize_score(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    v = valid.bool()
    x_masked = x.masked_fill(~v, 0.0)
    count = v.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
    mean = x_masked.sum(dim=-1, keepdim=True) / count
    var = ((x - mean) ** 2).masked_fill(~v, 0.0).sum(dim=-1, keepdim=True) / count
    return (x - mean) / torch.sqrt(var + 1e-6)


def build_ranker_features(
    boxes: torch.Tensor,
    aes: torch.Tensor,
    coverage: torch.Tensor,
    valid: torch.Tensor,
    extra_features: torch.Tensor | None,
    use_aesthetic: bool = True,
    use_extra_features: bool = True,
) -> torch.Tensor:
    geom = boxes_to_cxcywh(boxes)
    cx, cy, w, h = geom.unbind(dim=-1)

    area = (w * h).clamp(min=1e-6)
    aspect = (w / h).clamp(min=1e-6)

    dcx = torch.zeros_like(cx)
    dcy = torch.zeros_like(cy)
    dw = torch.zeros_like(w)
    dh = torch.zeros_like(h)
    dcx[:, 1:] = cx[:, 1:] - cx[:, :-1]
    dcy[:, 1:] = cy[:, 1:] - cy[:, :-1]
    dw[:, 1:] = w[:, 1:] - w[:, :-1]
    dh[:, 1:] = h[:, 1:] - h[:, :-1]

    center_dist = torch.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
    thirds_dist = (
        torch.minimum(torch.abs(cx - 1.0 / 3.0), torch.abs(cx - 2.0 / 3.0))
        + torch.minimum(torch.abs(cy - 1.0 / 3.0), torch.abs(cy - 2.0 / 3.0))
    )

    aes_feat = normalize_score(aes, valid) if use_aesthetic else torch.zeros_like(aes)

    feat = torch.stack(
        [
            aes_feat,
            coverage,
            cx,
            cy,
            w,
            h,
            area,
            aspect,
            dcx,
            dcy,
            dw,
            dh,
            center_dist,
            thirds_dist,
        ],
        dim=-1,
    )

    if use_extra_features and extra_features is not None:
        feat = torch.cat([feat, extra_features.to(feat.dtype)], dim=-1)

    return feat.masked_fill(~valid.bool().unsqueeze(-1), 0.0)


@dataclass
class DecodeResult:
    path_indices: torch.Tensor
    path_boxes: torch.Tensor
    node_score: torch.Tensor


class CropAwareRanker(nn.Module):
    def __init__(
        self,
        in_dim: int = 29,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        use_aesthetic: bool = True,
        use_extra_features: bool = True,
        decode_transition_weight: float = 0.02,
        pos_lambda: float = 0.25,
        scale_lambda: float = 0.15,
    ) -> None:
        super().__init__()
        self.use_aesthetic = bool(use_aesthetic)
        self.use_extra_features = bool(use_extra_features)
        self.decode_transition_weight = float(decode_transition_weight)
        self.pos_lambda = float(pos_lambda)
        self.scale_lambda = float(scale_lambda)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        boxes: torch.Tensor,
        aes: torch.Tensor,
        coverage: torch.Tensor,
        valid: torch.Tensor,
        extra_features: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        feat = build_ranker_features(
            boxes=boxes,
            aes=aes,
            coverage=coverage,
            valid=valid,
            extra_features=extra_features,
            use_aesthetic=self.use_aesthetic,
            use_extra_features=self.use_extra_features,
        )
        score = self.net(feat).squeeze(-1)
        return {"node_score": score.masked_fill(~valid.bool(), -1e9)}

    @staticmethod
    def transition_score(prev_boxes: torch.Tensor, curr_boxes: torch.Tensor, pos_lambda: float, scale_lambda: float) -> torch.Tensor:
        prev = boxes_to_cxcywh(prev_boxes)
        curr = boxes_to_cxcywh(curr_boxes)
        pos_dist = torch.cdist(prev[..., :2], curr[..., :2], p=2)
        scale_dist = (curr[:, None, :, 2:] - prev[:, :, None, 2:]).abs().sum(dim=-1)
        return -(float(pos_lambda) * pos_dist + float(scale_lambda) * scale_dist)

    @torch.no_grad()
    def decode(
        self,
        boxes: torch.Tensor,
        aes: torch.Tensor,
        coverage: torch.Tensor,
        valid: torch.Tensor,
        extra_features: torch.Tensor | None = None,
    ) -> DecodeResult:
        raw = self.forward(boxes, aes, coverage, valid, extra_features=extra_features)
        node_score = raw["node_score"]
        b, t, n, _ = boxes.shape

        dp = node_score[:, 0].clone()
        parent = torch.full((b, t, n), -1, dtype=torch.long, device=boxes.device)

        for ti in range(1, t):
            trans = self.transition_score(
                boxes[:, ti - 1],
                boxes[:, ti],
                self.pos_lambda,
                self.scale_lambda,
            )
            trans = trans * self.decode_transition_weight
            total = dp.unsqueeze(-1) + trans + node_score[:, ti].unsqueeze(1)
            total = total.masked_fill(~valid[:, ti].bool().unsqueeze(1), -1e9)
            best_val, best_idx = total.max(dim=1)
            dp = best_val
            parent[:, ti] = best_idx

        last_idx = dp.masked_fill(~valid[:, -1].bool(), -1e9).argmax(dim=1)
        path_idx = torch.zeros((b, t), dtype=torch.long, device=boxes.device)
        path_idx[:, -1] = last_idx
        batch_ids_1d = torch.arange(b, device=boxes.device)
        for ti in range(t - 1, 0, -1):
            path_idx[:, ti - 1] = parent[batch_ids_1d, ti, path_idx[:, ti]]

        batch_ids = batch_ids_1d.unsqueeze(1).expand(b, t)
        time_ids = torch.arange(t, device=boxes.device).unsqueeze(0).expand(b, t)
        return DecodeResult(
            path_indices=path_idx,
            path_boxes=boxes[batch_ids, time_ids, path_idx],
            node_score=node_score,
        )

