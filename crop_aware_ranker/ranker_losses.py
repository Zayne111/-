from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from crop_aware_ranker.ranker_model import boxes_to_cxcywh


def candidate_iou_to_multi_gt(boxes: torch.Tensor, gt_boxes_all: torch.Tensor) -> torch.Tensor:
    boxes_e = boxes.unsqueeze(1)
    gt_e = gt_boxes_all.unsqueeze(3)
    x1 = torch.maximum(boxes_e[..., 0], gt_e[..., 0])
    y1 = torch.maximum(boxes_e[..., 1], gt_e[..., 1])
    x2 = torch.minimum(boxes_e[..., 2], gt_e[..., 2])
    y2 = torch.minimum(boxes_e[..., 3], gt_e[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_p = (boxes_e[..., 2] - boxes_e[..., 0]).clamp(min=0) * (boxes_e[..., 3] - boxes_e[..., 1]).clamp(min=0)
    area_g = (gt_e[..., 2] - gt_e[..., 0]).clamp(min=0) * (gt_e[..., 3] - gt_e[..., 1]).clamp(min=0)
    return (inter / (area_p + area_g - inter).clamp(min=1e-6)).mean(dim=1)


def path_iou_to_multi_gt(pred_boxes: torch.Tensor, gt_boxes_all: torch.Tensor) -> torch.Tensor:
    pred_e = pred_boxes.unsqueeze(1)
    gt = gt_boxes_all
    x1 = torch.maximum(pred_e[..., 0], gt[..., 0])
    y1 = torch.maximum(pred_e[..., 1], gt[..., 1])
    x2 = torch.minimum(pred_e[..., 2], gt[..., 2])
    y2 = torch.minimum(pred_e[..., 3], gt[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_p = (pred_e[..., 2] - pred_e[..., 0]).clamp(min=0) * (pred_e[..., 3] - pred_e[..., 1]).clamp(min=0)
    area_g = (gt[..., 2] - gt[..., 0]).clamp(min=0) * (gt[..., 3] - gt[..., 1]).clamp(min=0)
    return (inter / (area_p + area_g - inter).clamp(min=1e-6)).mean(dim=1)


def soft_target_loss(node_score: torch.Tensor, target_score: torch.Tensor, valid: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = node_score.masked_fill(~valid.bool(), -1e9)
    target_score = target_score.masked_fill(~valid.bool(), -1e9)
    target_prob = torch.softmax(target_score / max(float(temperature), 1e-6), dim=-1)
    return -(target_prob * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def pairwise_ranking_loss(
    node_score: torch.Tensor,
    target_score: torch.Tensor,
    valid: torch.Tensor,
    margin: float,
    target_gap: float,
) -> torch.Tensor:
    score_i = node_score.unsqueeze(-1)
    score_j = node_score.unsqueeze(-2)
    target_i = target_score.unsqueeze(-1)
    target_j = target_score.unsqueeze(-2)
    valid_pair = valid.bool().unsqueeze(-1) & valid.bool().unsqueeze(-2)
    mask = ((target_i - target_j) > float(target_gap)) & valid_pair
    if not mask.any():
        return node_score.new_tensor(0.0)
    return F.relu(float(margin) - (score_i - score_j))[mask].mean()


@torch.no_grad()
def smooth_teacher_path(
    boxes: torch.Tensor,
    target_score: torch.Tensor,
    valid: torch.Tensor,
    pos_lambda: float = 0.8,
    scale_lambda: float = 0.4,
) -> torch.Tensor:
    b, t, n, _ = boxes.shape
    geom = boxes_to_cxcywh(boxes)
    node = target_score.masked_fill(~valid.bool(), -1e9)
    dp = node[:, 0].clone()
    parent = torch.full((b, t, n), -1, dtype=torch.long, device=boxes.device)

    for ti in range(1, t):
        prev = geom[:, ti - 1]
        curr = geom[:, ti]
        pos_dist = torch.cdist(prev[..., :2], curr[..., :2], p=2)
        scale_dist = (curr[:, None, :, 2:] - prev[:, :, None, 2:]).abs().sum(dim=-1)
        trans = -(float(pos_lambda) * pos_dist + float(scale_lambda) * scale_dist)
        total = dp.unsqueeze(-1) + trans + node[:, ti].unsqueeze(1)
        total = total.masked_fill(~valid[:, ti].bool().unsqueeze(1), -1e9)
        best_val, best_idx = total.max(dim=1)
        dp = best_val
        parent[:, ti] = best_idx

    last_idx = dp.masked_fill(~valid[:, -1].bool(), -1e9).argmax(dim=1)
    path_idx = torch.zeros((b, t), dtype=torch.long, device=boxes.device)
    path_idx[:, -1] = last_idx
    batch_ids = torch.arange(b, device=boxes.device)
    for ti in range(t - 1, 0, -1):
        path_idx[:, ti - 1] = parent[batch_ids, ti, path_idx[:, ti]]
    return path_idx


def smoothness_loss(path_boxes: torch.Tensor) -> torch.Tensor:
    geom = boxes_to_cxcywh(path_boxes)
    center = geom[..., :2]
    scale = geom[..., 2:]
    if center.size(1) < 2:
        return path_boxes.new_tensor(0.0)
    dc = center[:, 1:] - center[:, :-1]
    ds = scale[:, 1:] - scale[:, :-1]
    loss = dc.norm(dim=-1).mean() + ds.abs().sum(dim=-1).mean()
    if center.size(1) >= 3:
        acc = dc[:, 1:] - dc[:, :-1]
        loss = loss + 0.5 * acc.norm(dim=-1).mean()
    return loss


def compute_ranker_losses(model, batch: Dict[str, torch.Tensor], cfg, device: torch.device):
    boxes = batch["boxes"].to(device)
    aes = batch["aes"].to(device)
    coverage = batch["coverage"].to(device)
    valid = batch["valid"].to(device)
    extra = batch.get("extra_features", None)
    if extra is not None:
        extra = extra.to(device)
    gt_all = batch["gt_boxes_all"].to(device)

    raw = model(boxes, aes, coverage, valid, extra_features=extra)
    node_score = raw["node_score"]
    target_score = candidate_iou_to_multi_gt(boxes, gt_all).masked_fill(~valid.bool(), -1e9)
    target_idx = target_score.argmax(dim=-1)

    hard_ce = F.cross_entropy(node_score.reshape(-1, node_score.size(-1)), target_idx.reshape(-1))
    listwise = soft_target_loss(node_score, target_score, valid, cfg.listwise_temperature)
    pairwise = pairwise_ranking_loss(node_score, target_score, valid, cfg.pairwise_margin, cfg.pairwise_target_gap)

    teacher_idx = smooth_teacher_path(boxes, target_score, valid)
    path_ce = F.cross_entropy(node_score.reshape(-1, node_score.size(-1)), teacher_idx.reshape(-1))

    prob = torch.softmax(node_score, dim=-1)
    soft_path = (prob.unsqueeze(-1) * boxes).sum(dim=2)
    path_iou = path_iou_to_multi_gt(soft_path, gt_all).mean()

    b, t, _, _ = boxes.shape
    batch_ids = torch.arange(b, device=device).unsqueeze(1).expand(b, t)
    time_ids = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
    teacher_boxes = boxes[batch_ids, time_ids, teacher_idx]
    teacher_iou = path_iou_to_multi_gt(teacher_boxes, gt_all).mean()

    decode = model.decode(boxes, aes, coverage, valid, extra_features=extra)
    hard_iou = path_iou_to_multi_gt(decode.path_boxes, gt_all).mean()
    node_argmax_idx = node_score.argmax(dim=-1)
    node_argmax_boxes = boxes[batch_ids, time_ids, node_argmax_idx]
    node_argmax_iou = path_iou_to_multi_gt(node_argmax_boxes, gt_all).mean()

    smooth = smoothness_loss(decode.path_boxes)

    total = (
        cfg.hard_ce_weight * hard_ce
        + cfg.listwise_weight * listwise
        + cfg.pairwise_weight * pairwise
        + cfg.path_ce_weight * path_ce
        + cfg.path_iou_weight * (1.0 - path_iou)
        + cfg.smooth_weight * smooth
    )

    stats = {
        "loss": float(total.detach().cpu()),
        "hard_ce": float(hard_ce.detach().cpu()),
        "listwise": float(listwise.detach().cpu()),
        "pairwise": float(pairwise.detach().cpu()),
        "path_ce": float(path_ce.detach().cpu()),
        "soft_path_iou": float(path_iou.detach().cpu()),
        "hard_path_iou": float(hard_iou.detach().cpu()),
        "node_argmax_iou": float(node_argmax_iou.detach().cpu()),
        "teacher_path_iou": float(teacher_iou.detach().cpu()),
        "smooth": float(smooth.detach().cpu()),
    }
    return total, stats
