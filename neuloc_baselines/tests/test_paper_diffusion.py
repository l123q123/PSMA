"""Paper-style noise coverage must survive training, sampling and reload."""
from collections import Counter

import pytest
import torch

from neuloc_lvsm.diffusion import RayDiffusionSchedule
from neuloc_lvsm.model_diffusion import RayDiffusionNeuLoc


def tiny_model(**kwargs):
    return RayDiffusionNeuLoc(image_size=16, dim=16, heads=4, latent_count=5,
        encoder_layers=2, decoder_layers=2, diffusion_layers=1, no_z=True, **kwargs)


def test_paper_schedule_matches_official_terminal_distribution():
    net = tiny_model(diffusion_steps=100, diffusion_beta_start=1e-4, diffusion_beta_end=.2)
    schedule = net.diffusion_schedule
    assert float(schedule.alphas_cumprod[-1]) == pytest.approx(2.1399665e-5, rel=1e-5)
    clean = torch.ones(1, 4, 6)
    last = schedule.q_sample(clean, torch.tensor([99]), noise=torch.zeros_like(clean))
    torch.testing.assert_close(last, torch.full_like(clean, .004625977), atol=1e-7, rtol=1e-5)


def test_training_uniformly_samples_all_noise_levels():
    schedule = RayDiffusionSchedule(100, beta_end=.2)
    timesteps = schedule.sample_timesteps(20000, torch.device('cpu'), generator=torch.Generator().manual_seed(13))
    assert timesteps.dtype == torch.long
    assert set(timesteps.tolist()) == set(range(100))
    counts = torch.bincount(timesteps, minlength=100)
    assert counts.min() > 140 and counts.max() < 260


@pytest.mark.parametrize('t', [0, 50, 99])
def test_deterministic_ddpm_uses_posterior_mean(t):
    schedule = RayDiffusionSchedule(100, beta_end=.2)
    clean, noisy = torch.randn(2, 4, 6), torch.randn(2, 4, 6)
    times = torch.full((2,), t, dtype=torch.long)
    alpha = schedule.alphas_cumprod[t]
    previous = schedule.alphas_cumprod[t - 1] if t else torch.tensor(1.)
    beta = schedule.betas[t]
    expected = (previous.sqrt() * beta / (1 - alpha) * clean
        + (1 - beta).sqrt() * (1 - previous) / (1 - alpha) * noisy) if t else clean
    torch.testing.assert_close(schedule.ddpm_mean_step(noisy, clean, times), expected)


def test_sampler_visits_all_timesteps_and_reloads_identically(tmp_path):
    from neuloc_lvsm.evaluate import load_model
    from test_no_z import inputs
    net = tiny_model(diffusion_steps=100, diffusion_beta_end=.2, diffusion_sampler='ddpm_mean').eval()
    observed = []
    handle = net.diffusion_time_mlp.register_forward_pre_hook(lambda m, args: observed.append(args[0].clone()))
    x = inputs()
    with torch.no_grad():
        original = net.sample(*x, sample_steps=100, generator=torch.Generator().manual_seed(29))
    handle.remove()
    from neuloc_lvsm.diffusion import sinusoidal_timestep_embedding
    assert len(observed) == 100
    for i, embedding in enumerate(observed):
        torch.testing.assert_close(embedding, sinusoidal_timestep_embedding(torch.tensor([99 - i]), 16))
    checkpoint = tmp_path / 'paper.pt'
    torch.save({'model': net.state_dict(), 'model_config': {'no_z': True, **net.diffusion_config()}}, checkpoint)
    restored = load_model(checkpoint, torch.device('cpu'), 16)
    assert restored.diffusion_config() == net.diffusion_config()
    output = restored.sample(*x, sample_steps=100, generator=torch.Generator().manual_seed(29))
    torch.testing.assert_close(original.directions, output.directions, atol=0, rtol=0)
    torch.testing.assert_close(original.moments, output.moments, atol=0, rtol=0)
    with pytest.raises(ValueError, match='all training timesteps'):
        restored.sample(*x, sample_steps=16)


def test_legacy_checkpoint_still_loads_original_schedule(tmp_path):
    from neuloc_lvsm.evaluate import load_model
    net = tiny_model()
    checkpoint = tmp_path / 'legacy.pt'
    torch.save({'model': net.state_dict(), 'model_config': {'no_z': True}}, checkpoint)
    restored = load_model(checkpoint, torch.device('cpu'), 16)
    assert restored.diffusion_schedule.steps == 50
    assert float(restored.diffusion_schedule.betas[-1]) == pytest.approx(.02)
    assert restored.diffusion_sampler == 'ddim'


def test_resume_rejects_different_schedule(tmp_path):
    from train import load_resume_model
    checkpoint = tmp_path / 'old.pt'
    old = tiny_model()
    torch.save({'model': old.state_dict(), 'model_config': {'no_z': True}}, checkpoint)
    changed = tiny_model(diffusion_steps=100, diffusion_beta_end=.2)
    with pytest.raises(ValueError, match='diffusion.*configuration'):
        load_resume_model(changed, checkpoint)


def test_parser_exposes_schedule_and_sampler():
    from train import build_parser, validate_training_args
    args = build_parser().parse_args(['--manifest', 'x', '--ray-diffusion',
        '--diffusion-train-steps', '100', '--diffusion-beta-end', '.2',
        '--diffusion-sampler', 'ddpm_mean', '--diffusion-sample-steps', '100'])
    validate_training_args(args)
    args.diffusion_sample_steps = 16
    with pytest.raises(ValueError, match='all training timesteps'):
        validate_training_args(args)


def test_seen_subset_is_fixed_stratified_and_totals_5000():
    from no_z_experiment import stratified_indices
    episodes = [{'stage': stage, 'scene_id': f'scene{j % 5}'}
        for stage, count in [('global', 768), ('semi', 2530), ('dense', 6991), ('fine', 16824)]
        for j in range(count)]
    selected = stratified_indices(episodes, 5000, seed=20261015)
    assert len(selected) == len(set(selected)) == 5000
    assert selected == stratified_indices(episodes, 5000, seed=20261015)
    counts = Counter(episodes[i]['stage'] for i in selected)
    for stage in counts:
        original = sum(e['stage'] == stage for e in episodes)
        assert abs(counts[stage] - original * 5000 / len(episodes)) < 1
        assert len({episodes[i]['scene_id'] for i in selected if episodes[i]['stage'] == stage}) == 5


def test_paper_launch_arguments_do_not_resume_old_training(tmp_path):
    import no_z_experiment as experiment
    args = experiment.arguments(tmp_path, paper=True)
    assert args[args.index('--diffusion-train-steps') + 1] == '100'
    assert args[args.index('--diffusion-beta-end') + 1] == '0.2'
    assert args[args.index('--diffusion-sampler') + 1] == 'ddpm_mean'
    assert args[args.index('--diffusion-sample-steps') + 1] == '100'
    assert '--resume' not in args
    assert args[args.index('--lvsm-checkpoint') + 1] == str(experiment.BASELINE)
