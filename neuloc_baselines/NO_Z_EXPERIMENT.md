# No-Z Diffusion Baseline

This opt-in experiment changes the scene representation, not the localization
objective. No overlap head, confidence gating, or new cross-attention is added.
The immutable reference source is Git commit `b38abda`.

The original 50-step run was paused after epoch0. Its outputs remain available;
the corrected 100-step noise-coverage experiment is documented in
`PAPER_SCHEDULE.md` and `configs/no_z_diffusion_paper_v2.json`.

## Architecture

Original: pooled support tokens + learned Z -> 12-layer encoder -> Z outputs;
query tokens + Z outputs -> 12-layer decoder -> original 4-layer ray denoiser -> CAR.

No-Z: pooled support tokens only -> the same 12-layer encoder -> all support
outputs; query tokens + encoded support outputs -> the same 12-layer decoder ->
the unchanged ray denoiser -> CAR.

The frozen `latent_tokens` tensor is stored only for strict initialization and
legacy checkpoint shape inference; it is absent from both encoder and decoder
inputs. `RayOutput.latent` now contains support memory in no-Z mode, not Z tokens.
Token positions retain view-major ownership, even after cross-view mixing.
Invalid views and trailing token-budget padding are masked in both transformers.

## Constraints and Validation

- All original checkpoint tensors load strictly; no new learned parameters.
- Existing per-view pools and 100/50/20/8-support counts remain unchanged.
- Default `--no-z` is false; legacy checkpoint behavior remains available.
- Four-stage dataset mixture, loss, learning rates, and linear noise schedule
  follow the recovered diffusion continuation recipe in `configs/diffusion_v1.json`.
- Initialization uses the final diffusion checkpoint, so this is adaptation of a
  trained model, not a bitwise reproduction of its earlier V3-to-diffusion training.
- Training telemetry labels GT-noised pose metrics `denoise_rot/denoise_center`.
  Each epoch separately evaluates 28 fixed training examples per stage from pure
  noise, and evaluates the entire original seen/unseen splits stage by stage.
- Fixed sampler seeds and saved `metrics.jsonl` separate actual sampling quality
  from denoising loss. A small training probe is not full training-set accuracy.
- `trainbest.pt` is selected on denoising loss, `seenbest.pt`/`valbest.pt` on
  equal-stage validation score. Prefer held-out sampling metrics for conclusions.
- An inactive Z checkpoint tensor does not mean the no-Z architecture still uses
  a unified scene bottleneck. Conversely, preserving ownership does not guarantee
  that a future overlap head will learn accurate confidence.

## Execution

Use the independent server checkout `/data/wenhong/NeuLoc-no-z-v1` and the original
data/weights under `/data/wenhong/NeuLoc-lvsm`. The earlier experimental checkout
and all baseline checkpoints are left unchanged.
The launcher uses the original project as its working directory to resolve
relative manifest paths, but executes only the new checkout's absolute script.
The audit checks every unique manifest image/camera path before GPU tests.

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python no_z_experiment.py audit
OMP_NUM_THREADS=1 python no_z_experiment.py smoke
python no_z_experiment.py train --epochs 10
```

The script refuses to overwrite run directories, verifies code hashes before
launch, and enforces physical GPUs 1-7. Audit and smoke include optimizer steps,
but the real run initializes afresh from the original diffusion checkpoint.

The initial preflight samples just one episode per stage and split. Those
samples reveal a nontrivial zero-shot degradation after removing Z; they are not
an accuracy estimate or evidence that fine-tuning will recover the baseline.
Do not describe a finite-loss smoke test as successful localization.

## Work Checklist

- [x] Test-first no-Z token/mask, frame-order, gradients and checkpoint contracts.
- [x] Preserve default Z path and original diffusion mathematics.
- [x] Add explicitly named denoising telemetry and fixed pure-noise train probe.
- [x] Audit original manifests and complete four-stage full-size GPU backward tests.
- [x] Complete seven-GPU mixed-stage smoke and strict checkpoint reload.
- [x] Launch the original mixed-stage fine-tuning recipe in an independent run.
- [ ] Assess pure-noise train/seen/unseen results against the original model.

The seven-GPU smoke completed 12 optimizer steps and evaluated all four stages
on the train probe and both validation splits. A relative-data-path failure from
the first launch is retained in `smoke_7gpu_failed_relative_paths`; the corrected
launcher is covered by a regression test. The full test suite has 41 passing tests.
The formal run is `outputs/no_z_diffusion_fourstage_v1/train_7gpu_e10` with 10
epochs, freshly initialized from the reference diffusion weights, not smoke weights.
