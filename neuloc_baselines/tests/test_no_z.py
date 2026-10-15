"""No-Z removes only the shared latent bottleneck, never the pretrained encoder."""
import tempfile
from pathlib import Path

import pytest
import torch

from neuloc_lvsm.model_diffusion import RayDiffusionNeuLoc


def model(no_z=True, checkpoint=False):
    return RayDiffusionNeuLoc(image_size=16, dim=16, heads=4, latent_count=5,
        encoder_layers=2, decoder_layers=2, pool_token_counts=(2, 4),
        diffusion_layers=1, diffusion_steps=4, no_z=no_z,
        gradient_checkpointing=checkpoint)


def inputs(views=3):
    k = torch.tensor([[12., 0., 8.], [0., 12., 8.], [0., 0., 1.]])[None, None].repeat(1, views, 1, 1)
    poses = torch.eye(4)[None, None].repeat(1, views, 1, 1)
    poses[0, :, 0, 3] = torch.arange(views).float()
    return torch.rand(1, views, 3, 16, 16), k, poses, torch.rand(1, 3, 16, 16), k[:, 0]


def test_encoder_and_decoder_keep_support_slots_and_masks():
    net = model().eval()
    x = inputs()
    mask = torch.tensor([[True, False, True]])
    observed = {}
    def capture_encoder(module, args, kwargs):
        observed['encoder_tokens'] = args[0].shape[1]
        observed['encoder_mask'] = kwargs['key_padding_mask']
    def capture_decoder(module, args, kwargs):
        observed['decoder_tokens'] = args[0].shape[1]
        observed['decoder_mask'] = kwargs['key_padding_mask']
    net.encoder[0].register_forward_pre_hook(capture_encoder, with_kwargs=True)
    net.decoder[0].register_forward_pre_hook(capture_decoder, with_kwargs=True)
    result = net(*x, support_mask=mask, view_token_count=2, support_token_budget=9,
                 noisy_rays=torch.randn(1, 4, 6), timesteps=torch.tensor([3]))
    assert observed['encoder_tokens'] == 9
    assert observed['decoder_tokens'] == 13
    expected = torch.tensor([[True, True, False, False, True, True, False, False, False]])
    torch.testing.assert_close(observed['encoder_mask'], expected)
    torch.testing.assert_close(observed['decoder_mask'], torch.cat((torch.ones(1, 4, dtype=torch.bool), expected), 1))
    assert result.latent.shape == (1, 9, 16)


def test_no_new_parameters_and_latent_has_no_forward_effect():
    torch.manual_seed(10)
    reference = model(False).eval()
    net = model().eval()
    net.load_state_dict(reference.state_dict(), strict=True)
    assert set(net.state_dict()) == set(reference.state_dict())
    assert not net.latent_tokens.requires_grad
    x = inputs()
    with torch.no_grad():
        a = net.condition_features(*x, view_token_count=2)[1]
        net.latent_tokens.fill_(1e8)
        b = net.condition_features(*x, view_token_count=2)[1]
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_masked_slots_and_budget_padding_do_not_affect_query():
    net = model().eval()
    images, ks, poses, query, qk = inputs()
    with torch.no_grad():
        a = net.condition_features(images, ks, poses, query, qk,
            support_mask=torch.tensor([[True, False, True]]), view_token_count=2, support_token_budget=9)[1]
        b = net.condition_features(images[:, [0, 2]], ks[:, [0, 2]], poses[:, [0, 2]], query, qk, view_token_count=2)[1]
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_support_permutation_preserves_frame_mapping():
    net = model().eval()
    images, ks, poses, query, qk = inputs()
    order = torch.tensor([2, 0, 1])
    with torch.no_grad():
        a = net.condition_features(images, ks, poses, query, qk, view_token_count=2)
        b = net.condition_features(images[:, order], ks[:, order], poses[:, order], query, qk, view_token_count=2)
    torch.testing.assert_close(a[0].reshape(1, 3, 2, 16)[:, order], b[0].reshape(1, 3, 2, 16), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(a[1], b[1], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('views,tokens', [(3, 2), (2, 2), (2, 4), (2, 4)])
def test_all_stage_paths_backpropagate_through_pretrained_encoder(checkpoint, views, tokens):
    net = model(checkpoint=checkpoint).train()
    result = net(*inputs(views), view_token_count=tokens,
        noisy_rays=torch.randn(1, 4, 6), timesteps=torch.tensor([3]))
    result.raw_rays.square().mean().backward()
    for p in (net.encoder[0].attn.to_qkv.weight, net.encoder[-1].attn.to_qkv.weight,
              net.decoder[-1].attn.to_qkv.weight, net.diffusion_head.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0 and torch.isfinite(p.grad).all()
    assert net.latent_tokens.grad is None


def test_no_z_checkpoint_reloads_mode_and_sampling():
    from neuloc_lvsm.evaluate import load_model
    net = model().eval()
    # Default pools at this tiny image size are sufficient for strict reload.
    net = RayDiffusionNeuLoc(image_size=16, dim=16, heads=4, latent_count=5,
        encoder_layers=2, decoder_layers=2, diffusion_layers=1, diffusion_steps=4, no_z=True).eval()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'no_z.pt'
        torch.save({'model': net.state_dict(), 'model_config': {'no_z': True, 'diffusion_train_steps': 4}}, path)
        restored = load_model(path, torch.device('cpu'), 16)
    assert restored.no_z_enabled and not restored.latent_tokens.requires_grad
    out = restored.sample(*inputs(), sample_steps=4)
    assert out.latent.shape == (1, 12, 16)
    assert torch.isfinite(out.directions).all() and torch.isfinite(out.moments).all()


def test_training_no_z_is_opt_in():
    from train import build_parser
    parser = build_parser()
    assert not parser.parse_args(['--manifest', 'x']).no_z
    assert parser.parse_args(['--manifest', 'x', '--no-z']).no_z


def test_fixed_probe_retains_each_stage_and_does_not_mutate_dataset():
    from types import SimpleNamespace
    from train import fixed_training_probe
    dataset = SimpleNamespace(episodes=[{'stage': s, 'id': j} for s in ('global', 'semi', 'dense', 'fine') for j in range(10)])
    selected = fixed_training_probe(dataset, 3)
    assert len(selected.episodes) == 12
    assert len(dataset.episodes) == 40
    for stage in ('global', 'semi', 'dense', 'fine'):
        assert [e['id'] for e in selected.episodes if e['stage'] == stage] == [0, 4, 9]


def test_training_progress_labels_denoising_metrics():
    from train import _progress_postfix_train
    result = _progress_postfix_train(.3, .4, rotation_deg=12., center_error=.2)
    assert result['denoise_rot'] == '12.000deg'
    assert result['denoise_center'] == '0.2000'


def test_experiment_launch_resolves_original_relative_data_paths(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import no_z_experiment as experiment
    output = tmp_path / 'outputs'
    (output / 'smoke_7gpu').mkdir(parents=True)
    (output / 'smoke_7gpu/passed.json').write_text('{}')
    monkeypatch.setattr(experiment, 'ROOT', tmp_path)
    monkeypatch.setattr(experiment, 'OUTPUT', output)
    (output / 'preflight.json').write_text(json.dumps({'passed': True, 'source_hashes': {}}))
    observed = {}
    def launch(command, **kwargs):
        observed.update(kwargs)
        observed['command'] = command
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(experiment.subprocess, 'Popen', launch)
    experiment.run_training()
    assert observed['cwd'] == experiment.SOURCE
    assert str(tmp_path / 'train.py') in observed['command']
    assert observed['env']['CUDA_VISIBLE_DEVICES'] == '1,2,3,4,5,6,7'
