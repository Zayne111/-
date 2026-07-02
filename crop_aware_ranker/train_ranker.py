from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import build_section_namespace, load_full_config, merge_cli_overrides
from crop_aware_ranker.ranker_data import FeatureCacheDataset, collate_fn
from crop_aware_ranker.ranker_losses import compute_ranker_losses
from crop_aware_ranker.ranker_model import CropAwareRanker


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    total = {}
    count = 0
    for batch in loader:
        _, stats = compute_ranker_losses(model, batch, cfg, device)
        for k, v in stats.items():
            total[k] = total.get(k, 0.0) + v
        count += 1
    return {k: v / max(count, 1) for k, v in total.items()}


def train(cfg) -> None:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not getattr(cfg, "cpu", False) else "cpu")

    train_ds = FeatureCacheDataset(cfg.train_list, cfg.feature_cache_root, cfg.aspect_name)
    val_ds = FeatureCacheDataset(cfg.val_list, cfg.feature_cache_root, cfg.aspect_name)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)

    model = CropAwareRanker(
        in_dim=cfg.in_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        use_aesthetic=cfg.use_aesthetic,
        use_extra_features=cfg.use_extra_features,
        decode_transition_weight=cfg.decode_transition_weight,
        pos_lambda=cfg.pos_lambda,
        scale_lambda=cfg.scale_lambda,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ranker_config_used.json", "w", encoding="utf-8") as f:
        json.dump(vars(cfg), f, indent=2)

    best = -1.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = {}
        count = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, stats = compute_ranker_losses(model, batch, cfg, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            for k, v in stats.items():
                running[k] = running.get(k, 0.0) + v
            count += 1
        scheduler.step()

        train_stats = {k: v / max(count, 1) for k, v in running.items()}
        val_stats = evaluate(model, val_loader, cfg, device)
        print(f"Epoch {epoch:03d}")
        print("  train:", json.dumps(train_stats))
        print("  val:  ", json.dumps(val_stats))

        ckpt = {"model": model.state_dict(), "args": vars(cfg), "epoch": epoch, "val": val_stats}
        torch.save(ckpt, out_dir / "last.pt")
        score = val_stats["hard_path_iou"]
        if score > best:
            best = score
            torch.save(ckpt, out_dir / "best.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    cli = parse_args()
    full = load_full_config(cli.config)
    cfg = build_section_namespace(full, "train")
    cfg = merge_cli_overrides(cfg, cli)
    train(cfg)

