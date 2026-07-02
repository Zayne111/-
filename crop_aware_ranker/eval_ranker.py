from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import build_section_namespace, load_full_config, merge_cli_overrides
from crop_aware_ranker.ranker_data import FeatureCacheDataset, collate_fn
from crop_aware_ranker.ranker_losses import candidate_iou_to_multi_gt, path_iou_to_multi_gt
from crop_aware_ranker.ranker_model import CropAwareRanker


def parse_video_id_to_int(video_id: str) -> int:
    digits = "".join(ch for ch in video_id if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse video id: {video_id}")
    return int(digits)


def denorm_boxes(boxes: torch.Tensor, frame_size: torch.Tensor) -> torch.Tensor:
    h, w = int(frame_size[0].item()), int(frame_size[1].item())
    out = boxes.clone()
    out[:, [0, 2]] = out[:, [0, 2]] * w
    out[:, [1, 3]] = out[:, [1, 3]] * h
    return out


def export_txt(path_boxes: torch.Tensor, frame_size: torch.Tensor, video_id: str, aspect_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_num = parse_video_id_to_int(video_id)
    out_path = out_dir / f"{vid_num:03d}_{aspect_name}.txt"
    px = denorm_boxes(path_boxes, frame_size).round().long().cpu().numpy()
    with open(out_path, "w", encoding="utf-8") as f:
        for x1, y1, x2, y2 in px.tolist():
            f.write(f"{x1},{y1},{x2},{y2}\n")


@torch.no_grad()
def run_eval(cfg) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not getattr(cfg, "cpu", False) else "cpu")
    ckpt = torch.load(cfg.checkpoint, map_location="cpu")
    args = ckpt["args"]
    model = CropAwareRanker(
        in_dim=args["in_dim"],
        hidden_dim=args["hidden_dim"],
        dropout=args["dropout"],
        use_aesthetic=args["use_aesthetic"],
        use_extra_features=args["use_extra_features"],
        decode_transition_weight=args["decode_transition_weight"],
        pos_lambda=args["pos_lambda"],
        scale_lambda=args["scale_lambda"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = FeatureCacheDataset(cfg.test_list, cfg.feature_cache_root, cfg.aspect_name)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "mean_iou": 0.0,
        "oracle_iou": 0.0,
        "node_argmax_iou": 0.0,
        "selected_valid_ratio": 0.0,
    }
    per_video = []
    count = 0

    for batch in loader:
        boxes = batch["boxes"].to(device)
        aes = batch["aes"].to(device)
        coverage = batch["coverage"].to(device)
        valid = batch["valid"].to(device)
        extra = batch["extra_features"].to(device)
        gt = batch["gt_boxes_all"].to(device)
        name = batch["name"][0]
        frame_size = batch["frame_size"][0]

        decode = model.decode(boxes, aes, coverage, valid, extra_features=extra)
        node_idx = decode.node_score.argmax(dim=-1)
        b, t, _, _ = boxes.shape
        batch_ids = torch.arange(b, device=device).unsqueeze(1).expand(b, t)
        time_ids = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
        node_boxes = boxes[batch_ids, time_ids, node_idx]

        cand_iou = candidate_iou_to_multi_gt(boxes, gt).masked_fill(~valid.bool(), -1.0)
        row = {
            "name": name,
            "mean_iou": path_iou_to_multi_gt(decode.path_boxes, gt).mean().item(),
            "node_argmax_iou": path_iou_to_multi_gt(node_boxes, gt).mean().item(),
            "oracle_iou": cand_iou.max(dim=-1).values.mean().item(),
            "selected_valid_ratio": valid[batch_ids, time_ids, decode.path_indices].float().mean().item(),
        }
        for k in metrics:
            metrics[k] += row[k]
        per_video.append(row)
        count += 1

        if getattr(cfg, "export_txt", False):
            export_txt(decode.path_boxes[0], frame_size, name, cfg.aspect_name, out_dir)

    summary = {k: v / max(count, 1) for k, v in metrics.items()}
    summary["oracle_gap"] = summary["oracle_iou"] - summary["mean_iou"]
    print(json.dumps(summary, indent=2))
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "metrics_per_video.json", "w", encoding="utf-8") as f:
        json.dump(per_video, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--export_txt", action="store_true", default=None)
    return p.parse_args()


if __name__ == "__main__":
    cli = parse_args()
    full = load_full_config(cli.config)
    cfg = build_section_namespace(full, "eval")
    cfg = merge_cli_overrides(cfg, cli)
    run_eval(cfg)

