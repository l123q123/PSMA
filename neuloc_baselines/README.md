# NeuLoc Baselines: V3 and Ray Diffusion v1

This directory is independent of the existing PSMA/PASM code. It archives two
Z-preserving localization baselines, not the later direct-support/overlap experiments.

| Profile | Image conditioning | Ray prediction | Server checkpoint reference |
| --- | --- | --- | --- |
| `configs/v3.json` | Encoder -> 3072 Z tokens -> query decoder | Direct direction/moment heads -> CAR | `outputs/comprehensive_v1_full_ray_8gpu_e30_corrected/last.pt` |
| `configs/diffusion_v1.json` | Same Z-preserving image model | Four diffusion blocks, x0 prediction, DDIM -> CAR | `outputs/comprehensive_v1_raydiff_v1_7gpu_e10_scannetfix_e2to9/last.pt` |

The profiles are records, not configuration files automatically loaded by `train.py`.
Both use 256x256 images, 8x8 patches, dimension 768, 12 encoder layers, 12 decoder
layers, and 12 attention heads. Stage-specific support/token counts are
100x80, 50x160, 20x400, and 8x1024. Always use `--support-views 0` to retain the
manifest's support count.

## Scope and Provenance

- Shared runtime comes from a pre-direct-support compatibility source snapshot.
  It is not claimed to be a byte-for-byte copy of every historical training file.
- Dormant relative-pose/local-refinement compatibility helpers remain in that
  snapshot; both archived profiles disable them. They are not additional baselines.
- No Global direct-support modules, overlap confidence head, or zero-terminal-SNR
  experimental schedule is enabled or required by these profiles.
- `provenance.json` records source hashes, checkpoint references, and checkpoint
  metadata. `validation.json` records the actual archive checks.
- Checkpoint shapes/configuration are authoritative. Some legacy checkpoints did
  not save optimizer state, CLI arguments, or random seeds; exact optimizer/RNG
  resumption is therefore impossible from those files alone. V3's original launch
  learning rates have not been recovered and are deliberately not invented.
- Data, manifests containing full image/pose records, weights, caches, and training
  outputs are excluded from Git. They remain in the server project. There are no
  private keys or credentials in this directory.

## Install and Test

Run commands from this directory so its `neuloc_lvsm` package is selected:

```bash
python -m pip install -e '.[test]'
OMP_NUM_THREADS=1 python -m pytest tests -q
```

Archive validation: 24 tests passed. Both real server checkpoints loaded strictly;
one real 100-support episode produced exactly matching Z, direction, and moment
outputs against the source project (diffusion used 16 sampling steps and seed 91).
This checks archive compatibility, not a new accuracy benchmark.

The archived package uses the same import name as the server project. Do not mix
the two directories on `PYTHONPATH`.

## Evaluate Existing Weights

Paths below assume the original server checkout is available. Replace them on
another machine; no weights or datasets are downloaded automatically.

```bash
SOURCE=/data/wenhong/NeuLoc-lvsm
CUDA_VISIBLE_DEVICES=1 python -m neuloc_lvsm.evaluate \
  --checkpoint "$SOURCE/outputs/comprehensive_v1_full_ray_8gpu_e30_corrected/last.pt" \
  --manifest "$SOURCE/data/comprehensive_v1/seen_val_without_scannet.json" \
  --output outputs/v3_seen.json --image-size 256 --support-views 0

CUDA_VISIBLE_DEVICES=1 python -m neuloc_lvsm.evaluate \
  --checkpoint "$SOURCE/outputs/comprehensive_v1_raydiff_v1_7gpu_e10_scannetfix_e2to9/last.pt" \
  --manifest "$SOURCE/data/comprehensive_v1/seen_val_without_scannet.json" \
  --output outputs/diffusion_v1_seen.json --image-size 256 --support-views 0 \
  --diffusion-steps 16
```

`CUDA_VISIBLE_DEVICES=1` means physical GPU1; internal `cuda:0` is then GPU1.
CAR center errors use normalized coordinates, not meters.

## Diffusion Training Recipe

The historical initial diffusion run loaded V3 and trained with
`data/comprehensive_v1/train.json`. It was resumed after epoch 0 and again after
epoch 1. The final continuation used the manifests below, replacing legacy
ScanNet training episodes with overlap8 episodes and excluding ScanNet from
validation. The retained final checkpoint is epoch 9.

The following starts a **new fine-tuning run** from the final diffusion weights;
it is not a claim of bitwise historical reproduction:

```bash
SOURCE=/data/wenhong/NeuLoc-lvsm
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.run \
  --standalone --nproc_per_node=7 train.py \
  --manifest "$SOURCE/data/comprehensive_v1/train_replace_legacy_scannet_overlap8x4.json" \
  --val-manifest "$SOURCE/data/comprehensive_v1/seen_val_without_scannet.json" \
  --unseen-val-manifest "$SOURCE/data/comprehensive_v1/unseen_val_without_scannet.json" \
  --lvsm-checkpoint "$SOURCE/outputs/comprehensive_v1_raydiff_v1_7gpu_e10_scannetfix_e2to9/last.pt" \
  --output outputs/diffusion_v1_finetune \
  --ray-diffusion --diffusion-layers 4 --diffusion-train-steps 50 \
  --diffusion-sample-steps 16 --moment-scale 1.0 \
  --epochs 1 --batch-size 1 --num-workers 4 --image-size 256 --support-views 0 \
  --dim 768 --latent-count 3072 --encoder-layers 12 --decoder-layers 12 --heads 12 \
  --amp-dtype bf16 --lr 2e-6 --new-lr 2e-5
```

For V3 fine-tuning, select its checkpoint and omit the diffusion flags; training
hyperparameters for that new experiment must be chosen explicitly.

## Metric Interpretation

Diffusion training predicts clean rays from GT rays corrupted at random timesteps.
Deployment/validation starts from pure Gaussian noise. Low denoising loss does not
by itself prove accurate image-based localization. Evaluate training and held-out
samples with the same pure-noise sampler when comparing their pose accuracy.

Historical diffusion epoch 9 on `seen_val_without_scannet.json`:

| Stage | Episodes | Rotation mean | Normalized center mean |
| --- | ---: | ---: | ---: |
| Global | 768 | 21.5039 deg | 0.511627 |
| Semi | 2530 | 14.1776 deg | 0.321851 |
| Dense | 6991 | 8.3169 deg | 0.195273 |
| Fine | 16824 | 6.9507 deg | 0.165218 |
| All, episode-weighted | 27113 | 8.3895 deg | 0.197396 |

The overall 8.3895-degree number is not Global-only accuracy. These historical
metrics belong to their exact checkpoint and split, not to every V3 or diffusion
checkpoint. The V3 reference is archived for lineage and comparison; its historical
checkpoint metrics are preserved without claiming it beats the diffusion baseline.
