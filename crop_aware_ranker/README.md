# Crop-Aware Aesthetic Ranker

This folder contains a standalone route for the paper story:

> Generic image aesthetic scores are poorly aligned with video crop annotations. We bridge this gap by learning a crop-aware aesthetic ranker conditioned on SAM2 object tubes and temporal smoothness.

The code reuses the `.npz` feature cache produced by `build_features.py`. It does not rerun SAM2 or Charm. Each candidate crop is represented by:

- crop geometry,
- Charm aesthetic score,
- SAM2 object coverage,
- proposal features derived from object-tube anchors and motion,
- temporal delta features.

The ranker is supervised by the mean IoU between each candidate crop and the six RetargetVid annotators. Charm is treated as an auxiliary input, not as the final target.

## Files

- `ranker_model.py`: frame-wise crop-aware ranker.
- `ranker_data.py`: feature-cache dataset.
- `ranker_losses.py`: RetargetVid IoU targets and ranking losses.
- `train_ranker.py`: training entrypoint.
- `eval_ranker.py`: evaluation and RetargetVid `.txt` export.
- `ranker_config.yaml`: default config.

## Recommended Workflow

First rebuild features with the current feature schema:

```bash
python /root/1/build_features.py \
  --config /root/1/RetargetVid.yaml \
  --split all \
  --force_rebuild
```

Do not use `--force_sam2_rebuild` unless SAM2 outputs are missing.

Train:

```bash
python /root/1/crop_aware_ranker/train_ranker.py \
  --config /root/1/crop_aware_ranker/ranker_config.yaml
```

Evaluate and export RetargetVid result files:

```bash
python /root/1/crop_aware_ranker/eval_ranker.py \
  --config /root/1/crop_aware_ranker/ranker_config.yaml \
  --checkpoint /root/autodl-tmp/crop_aware_ranker/best.pt \
  --export_txt
```

Official RetargetVid evaluation:

```bash
python /root/1/eval_retarget.py /root/autodl-tmp/results
```

## Paper Ablations

Recommended ablation table:

- no aesthetic score,
- Charm-only prior,
- geometry + SAM2 coverage,
- geometry + SAM2 + Charm,
- geometry + SAM2 + Charm + proposal features,
- crop-aware ranker + temporal decoding.

