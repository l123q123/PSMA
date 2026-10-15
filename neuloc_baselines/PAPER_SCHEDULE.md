# Paper-Style Noise Coverage

## Scope

The legacy baseline sampled training timesteps uniformly from 0 through 49;
it never trained only at timestep 50. Its linear beta range [0.0001, 0.02]
left terminal alpha_bar about 0.60295, while inference began at pure Gaussian
noise. Low denoising error therefore did not establish pure-noise localization.

The new opt-in profile matches RayDiffusion's public training configuration:
100 training timesteps with linear beta in [0.0001, 0.2]. Terminal alpha_bar is
about 0.0000214, giving clean-ray coefficient 0.004626 and noise coefficient
0.999989. Training samples one timestep uniformly per episode, with clean x0
targets. It does not backpropagate through a full reverse trajectory.

The architecture remains the conservative no-Z baseline. No confidence head,
gating, extra attention layer, loss replacement or data augmentation is added.
Initialize from the original final diffusion weights, not the failed no-Z run.
This is weight transfer under a changed schedule, not optimizer continuation.

## Inference and Checkpoints

The new `ddpm_mean` sampler starts at Gaussian rays and traverses all timesteps
99 through 0 using adjacent-step DDPM posterior means without extra random
noise. Clean-ray estimates are canonicalized before the posterior update;
the evolving noisy state is clipped to [-5, 5]. The final predicted clean rays
are passed to CAR. Conditioning runs once per episode.

This follows the paper's deterministic DDPM principle, not its entire method.
We retain NeuLoc conditioning, geometry, losses and CAR; no early-stop setting
is tuned or claimed to reproduce the paper. This sampler is not DDIM with eta=0.

Schedule bounds, step count, sampler and moment scale are saved in model_config.
Loading old checkpoints without those fields preserves their legacy schedule
and DDIM sampler. Resume rejects a changed diffusion configuration; explicit
`--lvsm-checkpoint` weight transfer is required for a changed schedule.

## Data and Execution

Training keeps the original 20,632 mixed-stage episodes. Seen validation is a
fixed 5,000-episode subset, proportionally stratified by stage then dataset/scene.
Original manifest indices and hash are saved in `seen_selection.json`. Stage
counts are Global142, Semi466, Dense1289, Fine3103. Unseen remains all 3,204 cases.
The same subset and sampler seeds must be used for baseline comparisons.

Only physical GPUs1-7 are used. All launches run with the original project as
the working directory so relative data paths resolve, but execute the new source.
Old source was archived before modification; old outputs remain untouched.

```bash
cd /data/wenhong/NeuLoc-no-z-v1
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python no_z_experiment.py audit --paper
OMP_NUM_THREADS=1 python no_z_experiment.py smoke --paper
python no_z_experiment.py train --paper --epochs 10
```

Output root: `outputs/no_z_diffusion_paper_v2`. Formal training requires fresh
code/data/checkpoint hash checks and a passing seven-GPU smoke. The smoke does
not initialize formal training. Per-epoch histograms record all 100 sampled
training timesteps; pose quality is evaluated separately from pure noise.

Reference code:
- https://github.com/jasonyzhang/RayDiffusion/blob/main/conf/config.yaml
- https://github.com/jasonyzhang/RayDiffusion/blob/main/train.py
- https://github.com/jasonyzhang/RayDiffusion/blob/main/ray_diffusion/inference/ddpm.py
