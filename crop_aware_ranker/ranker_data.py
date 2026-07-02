from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureCacheDataset(Dataset):
    def __init__(self, video_list_file: str, feature_cache_root: str, aspect_name: str) -> None:
        with open(video_list_file, "r", encoding="utf-8") as f:
            self.video_ids = [line.strip() for line in f if line.strip()]
        self.feature_root = Path(feature_cache_root) / aspect_name

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id = self.video_ids[idx]
        path = self.feature_root / f"{video_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Feature cache not found: {path}")
        arr = np.load(path, allow_pickle=False)
        sample = {
            "boxes": torch.from_numpy(arr["boxes"]).float(),
            "aes": torch.from_numpy(arr["aes"]).float(),
            "coverage": torch.from_numpy(arr["coverage"]).float(),
            "valid": torch.from_numpy(arr["valid"]).bool(),
            "frame_size": torch.from_numpy(arr["frame_size"]).long(),
            "name": video_id,
        }
        if "extra_features" in arr:
            sample["extra_features"] = torch.from_numpy(arr["extra_features"]).float()
        else:
            t, n = sample["boxes"].shape[:2]
            sample["extra_features"] = torch.zeros((t, n, 15), dtype=torch.float32)
        if "gt_boxes_all" in arr:
            sample["gt_boxes_all"] = torch.from_numpy(arr["gt_boxes_all"]).float()
        else:
            raise KeyError(f"{path} does not contain gt_boxes_all. Rebuild features with gt_all_annotators=true.")
        return sample


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(batch) != 1:
        raise ValueError("FeatureCacheDataset currently expects batch_size=1 because videos have variable length.")
    b = batch[0]
    out = {}
    for key in ["boxes", "aes", "coverage", "valid", "extra_features", "frame_size", "gt_boxes_all"]:
        out[key] = b[key].unsqueeze(0)
    out["name"] = [b["name"]]
    return out

