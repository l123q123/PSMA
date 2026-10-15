"""Only the two Z-preserving baseline profiles belong to this archive."""
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name,diffusion', [('v3', False), ('diffusion_v1', True)])
def test_profile_preserves_z_and_disables_unrelated_experiments(name, diffusion):
    profile = json.loads((ROOT / 'configs' / f'{name}.json').read_text())
    assert profile['model']['ray_diffusion'] is diffusion
    assert profile['model']['latent_count'] == 3072
    assert profile['model']['encoder_layers'] == profile['model']['decoder_layers'] == 12
    assert profile['model']['relative_pose'] is False
    assert profile['model']['local_refinement'] is False
    assert profile['training']['support_views'] == 0


@pytest.mark.parametrize('diffusion', [False, True])
def test_baseline_checkpoint_roundtrip_and_ray_generation(tmp_path, diffusion):
    from neuloc_lvsm.model import LVSMNeuLoc
    from neuloc_lvsm.model_diffusion import RayDiffusionNeuLoc
    from neuloc_lvsm.evaluate import load_model
    cls = RayDiffusionNeuLoc if diffusion else LVSMNeuLoc
    extra = dict(diffusion_layers=1, diffusion_steps=4) if diffusion else {}
    model = cls(image_size=16, dim=16, latent_count=4, encoder_layers=1,
                decoder_layers=1, heads=4, **extra).eval()
    assert not any(k.startswith(('global_', 'confidence_head')) for k in model.state_dict())
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'model': model.state_dict(), 'model_config': {'diffusion_train_steps': 4}}, checkpoint)
    restored = load_model(checkpoint, torch.device('cpu'), image_size=16)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    images = torch.rand(1, 3, 3, 16, 16)
    k = torch.tensor([[12., 0., 8.], [0., 12., 8.], [0., 0., 1.]])[None, None].repeat(1, 3, 1, 1)
    poses = torch.eye(4)[None, None].repeat(1, 3, 1, 1)
    poses[0, :, 0, 3] = torch.arange(3).float()
    args = (images, k, poses, torch.rand(1, 3, 16, 16), k[:, 0])
    with torch.no_grad():
        output = restored.sample(*args, sample_steps=4, view_token_count=4) if diffusion else restored(*args, view_token_count=4)
    assert output.latent.shape == (1, 4, 16)
    assert output.directions.shape == (1, 4, 3)
    assert torch.isfinite(output.moments).all()


def test_legacy_diffusion_noise_schedule_is_preserved():
    from neuloc_lvsm.diffusion import RayDiffusionSchedule
    actual = RayDiffusionSchedule(50)
    expected = torch.cumprod(1 - torch.linspace(1e-4, 2e-2, 50), dim=0)
    torch.testing.assert_close(actual.alphas_cumprod, expected, atol=0, rtol=0)
