"""Audited four-stage no-Z fine-tuning; all GPU work excludes physical GPU0."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
import random
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
SOURCE = Path('/data/wenhong/NeuLoc-lvsm')
BASELINE = SOURCE / 'outputs/comprehensive_v1_raydiff_v1_7gpu_e10_scannetfix_e2to9/last.pt'
OUTPUT = ROOT / 'outputs/no_z_diffusion_fourstage_v1'
MANIFESTS = {
    'train': SOURCE / 'data/comprehensive_v1/train_replace_legacy_scannet_overlap8x4.json',
    'seen': SOURCE / 'data/comprehensive_v1/seen_val_without_scannet.json',
    'unseen': SOURCE / 'data/comprehensive_v1/unseen_val_without_scannet.json',
}
STAGES = {'global': (100, 80), 'semi': (50, 160), 'dense': (20, 400), 'fine': (8, 1024)}
PAPER = False


def stratified_indices(episodes, total, seed):
    """Proportional fixed sampling by stage, then dataset/scene; no prefix bias."""
    if not 0 < total <= len(episodes):
        raise ValueError('subset size must be positive and no larger than the manifest')
    def allocate(groups, count):
        size = sum(len(items) for items in groups.values())
        quotas = {key: count * len(items) // size for key, items in groups.items()}
        remainder = count - sum(quotas.values())
        priority = sorted(groups, key=lambda key: (-(count * len(groups[key]) % size), str(key)))
        for key in priority[:remainder]:
            quotas[key] += 1
        return quotas
    stages = {stage: [] for stage in STAGES}
    for index, episode in enumerate(episodes):
        stages[episode.get('stage', 'global')].append(index)
    rng = random.Random(seed)
    selected = []
    for stage, count in allocate(stages, total).items():
        if not count:
            continue
        scenes = {}
        for index in stages[stage]:
            episode = episodes[index]
            key = (str(episode.get('dataset', 'unknown')),
                str(episode.get('scene_id', episode.get('scene', 'unknown'))))
            scenes.setdefault(key, []).append(index)
        for key, quota in allocate(scenes, count).items():
            selected.extend(rng.sample(scenes[key], quota))
    return sorted(selected)


def prepare_seen_subset():
    original = MANIFESTS['seen']
    payload = json.loads(original.read_text())
    episodes = payload['episodes'] if isinstance(payload, dict) else payload
    indices = stratified_indices(episodes, 5000, 20261015)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    subset = dict(payload) if isinstance(payload, dict) else {}
    subset['episodes'] = [episodes[index] for index in indices]
    destination = OUTPUT / 'seen5000.json'
    destination.write_text(json.dumps(subset) + '\n')
    (OUTPUT / 'seen_selection.json').write_text(json.dumps(dict(original_manifest=str(original),
        original_sha256=sha256(original), seed=20261015, original_indices=indices,
        stage_counts=dict(Counter(e.get('stage', 'global') for e in subset['episodes']))), indent=2) + '\n')
    MANIFESTS['seen'] = destination


def diffusion_kwargs():
    return dict(diffusion_steps=100, diffusion_beta_start=1e-4, diffusion_beta_end=.2,
        diffusion_sampler='ddpm_mean') if PAPER else dict(diffusion_steps=50)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def code_hashes():
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(ROOT.rglob('*.py'))
            if not any(part in ('outputs', '__pycache__', '.git') for part in path.relative_to(ROOT).parts)}


def arguments(directory, *, smoke=False, epochs=10, paper=None):
    paper = PAPER if paper is None else paper
    result = ['--manifest', str(MANIFESTS['train']), '--val-manifest', str(MANIFESTS['seen']),
        '--unseen-val-manifest', str(MANIFESTS['unseen']), '--output', str(directory),
        '--lvsm-checkpoint', str(BASELINE), '--ray-diffusion', '--no-z',
        '--diffusion-layers', '4', '--diffusion-train-steps', '50', '--diffusion-sample-steps', '16',
        '--moment-scale', '1.0', '--diffusion-ray-loss-weight', '.5', '--diffusion-pose-loss-weight', '1.0',
        '--epochs', '1' if smoke else str(epochs), '--batch-size', '1', '--num-workers', '4',
        '--image-size', '256', '--support-views', '0', '--max-support-views', '0',
        '--dim', '768', '--latent-count', '3072', '--encoder-layers', '12', '--decoder-layers', '12',
        '--heads', '12', '--amp-dtype', 'bf16', '--lr', '2e-6', '--new-lr', '2e-5',
        '--seed', '20261015', '--val-every', '1', '--ddp-timeout-minutes', '120',
        '--train-probe-per-stage', '7' if smoke else '28',
        '--val-max-episodes', '7' if smoke else '0', '--unseen-val-max-episodes', '7' if smoke else '0']
    if smoke:
        result += ['--max-steps', '12']
    if paper:
        result[result.index('--diffusion-train-steps') + 1] = '100'
        result[result.index('--diffusion-sample-steps') + 1] = '100'
        result += ['--diffusion-beta-start', '0.0001', '--diffusion-beta-end', '0.2',
            '--diffusion-sampler', 'ddpm_mean']
    return result


def audit():
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Subset
    from neuloc_lvsm.data import EpisodeDataset
    from neuloc_lvsm.evaluate import load_model, predict, query_camera_rays
    from neuloc_lvsm.model_diffusion import RayDiffusionNeuLoc
    from neuloc_lvsm.solver import solve_car_pose
    from train import configure_cuda_performance, _masked_pose_loss, fixed_training_probe

    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1', 'preflight uses physical GPU1 only'
    if PAPER:
        prepare_seen_subset()
    torch.set_num_threads(1)
    configure_cuda_performance()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {'source_hashes': code_hashes(), 'baseline': str(BASELINE), 'manifests': {}, 'checks': []}
    cases = []
    checked_assets = set()
    for split, path in MANIFESTS.items():
        ds = EpisodeDataset(path, image_size=256, support_views=0)
        counts = Counter(e.get('stage', 'global') for e in ds.episodes)
        assert set(counts) == set(STAGES)
        for e in ds.episodes:
            stage = e.get('stage', 'global')
            views, tokens = STAGES[stage]
            assert len(e['support']) == views, (split, stage, len(e['support']))
            assert e['view_token_count'] == tokens
            assert e['support_token_budget'] == 8192
            for item in [e['query'], *e['support']]:
                for field in ('image', 'image_path', 'camera', 'camera_path'):
                    if item.get(field):
                        asset = SOURCE / item[field]
                        if asset not in checked_assets:
                            assert asset.is_file(), f'missing manifest asset: {asset}'
                            checked_assets.add(asset)
        report['manifests'][split] = dict(path=str(path), sha256=sha256(path), episodes=len(ds), stages=dict(counts))
        # Retain all 100/50/20/8 supports, never a default four-view slice.
        for stage, (views, tokens) in STAGES.items():
            index = next(i for i, e in enumerate(ds.episodes) if e.get('stage', 'global') == stage)
            batch = next(iter(DataLoader(Subset(ds, [index]), batch_size=1)))
            assert batch['support_images'].shape == (1, views, 3, 256, 256)
            assert batch['support_mask'].all()
            cases.append((split, stage, index, batch))
        if split == 'train':
            probe = fixed_training_probe(ds, 28)
            report['training_probe'] = [dict(stage=e.get('stage', 'global'), scene=e.get('scene_id'),
                query_image=e['query'].get('image', e['query'].get('image_path'))) for e in probe.episodes]
            del probe
        print('MANIFEST_AUDIT', split, report['manifests'][split], flush=True)
        del ds
        gc.collect()
    report['checked_asset_count'] = len(checked_assets)
    assert report['manifests']['train']['episodes'] == 20632
    payload = torch.load(BASELINE, map_location='cpu', weights_only=True)
    torch.manual_seed(20261015)
    model = RayDiffusionNeuLoc(image_size=256, dim=768, latent_count=3072, encoder_layers=12,
        decoder_layers=12, heads=12, diffusion_layers=4, no_z=True, **diffusion_kwargs()).cuda()
    model.load_state_dict(payload['model'], strict=True)
    assert all(torch.equal(value, model.state_dict()[key].cpu()) for key, value in payload['model'].items())
    report['strict_initialization'] = True
    report['baseline_sha256'] = sha256(BASELINE)
    report['stored_parameters'] = sum(p.numel() for p in model.parameters())
    assert not model.latent_tokens.requires_grad
    assert not any(name.startswith(('global_', 'confidence_head')) for name in model.state_dict())
    report['terminal_alpha_bar'] = float(model.diffusion_schedule.alphas_cumprod[-1])
    report['diffusion_config'] = model.diffusion_config()
    if PAPER:
        assert report['terminal_alpha_bar'] < 3e-5
        assert report['manifests']['seen']['episodes'] == 5000
    del payload
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-6, weight_decay=.05)
    for j, (split, stage, index, cpu_batch) in enumerate(cases[:4]):
        model.train()
        batch = {key: value.cuda() for key, value in cpu_batch.items()}
        local = query_camera_rays(batch['query_intrinsics'], 256, 8)
        gt_dirs = F.normalize((batch['query_pose'][:, None, :3, :3] @ local[..., None]).squeeze(-1), dim=-1)
        gt_moments = torch.cross(batch['query_pose'][:, None, :3, 3].expand_as(gt_dirs), gt_dirs, dim=-1)
        x0 = torch.cat((gt_dirs, gt_moments), -1)
        times = torch.tensor([((99, 50, 0, 80) if PAPER else (49, 25, 0, 40))[j]], device='cuda')
        noisy = model.diffusion_schedule.q_sample(x0, times)
        optimizer.zero_grad(set_to_none=True)
        seen_encoder = []
        seen_decoder = []
        h1 = model.encoder[0].register_forward_pre_hook(lambda m, a: seen_encoder.append(a[0].shape[1]))
        h2 = model.decoder[0].register_forward_pre_hook(lambda m, a: seen_decoder.append(a[0].shape[1]))
        started = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(batch['support_images'], batch['support_intrinsics'], batch['support_poses'],
                batch['query_image'], batch['query_intrinsics'], batch['support_type_ids'], batch['support_mask'],
                view_token_count=STAGES[stage][1], support_token_budget=8192, noisy_rays=noisy, timesteps=times)
            pose = solve_car_pose(out.directions, out.moments, local)
            loss = (F.mse_loss(out.raw_rays, x0)
                + .5 * (F.smooth_l1_loss(out.directions, gt_dirs) + .5 * F.smooth_l1_loss(out.moments, gt_moments))
                + _masked_pose_loss(pose, batch['query_pose'], batch['stage_id'].reshape(-1) == 0))
        h1.remove()
        h2.remove()
        assert seen_encoder == [8192] and seen_decoder == [9216]
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in (model.encoder[-1].attn.to_qkv.weight, model.decoder[-1].attn.to_qkv.weight,
                          model.image_tokenizer.weight, model.diffusion_head.weight):
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0
        assert model.latent_tokens.grad is None
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        record = dict(stage=stage, timestep=int(times[0]), loss=float(loss.detach()), seconds=time.monotonic() - started,
            encoder_tokens=seen_encoder[0], decoder_tokens=seen_decoder[0], peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30)
        report['checks'].append(record)
        print('GPU_BACKWARD_PASS', record, flush=True)
    del optimizer, out, loss
    gc.collect()
    torch.cuda.empty_cache()
    # Save/reload the new mode, but never initialize formal training from this probe.
    probe_checkpoint = OUTPUT / 'roundtrip.pt'
    torch.save({'model': model.state_dict(), 'model_config': {'no_z': True, **model.diffusion_config()}}, probe_checkpoint)
    restored = load_model(probe_checkpoint, torch.device('cpu'), 256)
    assert restored.no_z_enabled
    assert restored.diffusion_config() == model.diffusion_config()
    assert all(torch.equal(value.cpu(), restored.state_dict()[key]) for key, value in model.state_dict().items())
    report['strict_roundtrip'] = True
    del restored, model
    gc.collect()
    torch.cuda.empty_cache()
    report['initialization_comparison'] = {}
    for no_z in (False, True):
        model = load_model(BASELINE, torch.device('cuda'), 256)
        model.no_z_enabled = no_z
        model.latent_tokens.requires_grad_(not no_z)
        if PAPER and no_z:
            from neuloc_lvsm.diffusion import RayDiffusionSchedule
            model.diffusion_schedule = RayDiffusionSchedule(100, beta_end=.2).cuda()
            model.diffusion_sampler = 'ddpm_mean'
        records = []
        for j, (split, stage, index, cpu_batch) in enumerate(cases):
            batch = {key: value.cuda() for key, value in cpu_batch.items()}
            with torch.inference_mode(), torch.random.fork_rng(devices=[0]):
                torch.manual_seed(6000 + j)
                output, pose, metrics = predict(model, batch, 256, diffusion_steps=100 if PAPER and no_z else 16)
            assert torch.isfinite(output.directions).all() and torch.isfinite(output.moments).all()
            records.append(dict(split=split, stage=stage, index=index, **metrics))
            print('INITIAL_SAMPLING', no_z, records[-1], flush=True)
        report['initialization_comparison']['no_z' if no_z else 'original_z'] = records
        del model, output, pose
        gc.collect()
        torch.cuda.empty_cache()
    report['passed'] = True
    (OUTPUT / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    print('NO_Z_PREFLIGHT_PASSED', flush=True)


def run_training(smoke=False, epochs=10):
    preflight = json.loads((OUTPUT / 'preflight.json').read_text())
    assert preflight['passed'] and preflight['source_hashes'] == code_hashes()
    if PAPER:
        MANIFESTS['seen'] = OUTPUT / 'seen5000.json'
        assert preflight['baseline_sha256'] == sha256(BASELINE)
        assert all(preflight['manifests'][key]['sha256'] == sha256(path) for key, path in MANIFESTS.items())
    if not smoke:
        assert (OUTPUT / 'smoke_7gpu/passed.json').is_file()
        if PAPER:
            assert json.loads((OUTPUT / 'smoke_7gpu/launch.json').read_text())['source_hashes'] == code_hashes()
    directory = OUTPUT / ('smoke_7gpu' if smoke else 'train_7gpu_e10')
    if directory.exists():
        raise RuntimeError(f'refusing to overwrite {directory}')
    directory.mkdir(parents=True)
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=7',
               str(ROOT / 'train.py')] + arguments(directory, smoke=smoke, epochs=epochs)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='1,2,3,4,5,6,7', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    (directory / 'launch.json').write_text(json.dumps(dict(command=command, source_hashes=code_hashes(),
        gpu_ids=env['CUDA_VISIBLE_DEVICES'], baseline=str(BASELINE), working_directory=str(SOURCE),
        epochs=1 if smoke else epochs), indent=2) + '\n')
    with (directory / 'train.log').open('w') as log:
        process = subprocess.Popen(command, env=env, cwd=SOURCE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    (directory / 'train.pid').write_text(str(process.pid) + '\n')
    print(json.dumps(dict(pid=process.pid, output=str(directory), smoke=smoke)), flush=True)
    if smoke:
        code = process.wait()
        if code:
            raise RuntimeError(f'DDP smoke failed ({code}), inspect {directory}/train.log')
        import torch
        from neuloc_lvsm.evaluate import load_model
        restored = load_model(directory / 'last.pt', torch.device('cpu'), 256)
        assert restored.no_z_enabled
        if PAPER:
            assert restored.diffusion_schedule.steps == 100
            assert restored.diffusion_sampler == 'ddpm_mean'
            assert float(restored.diffusion_schedule.alphas_cumprod[-1]) < 3e-5
        saved = torch.load(directory / 'last.pt', map_location='cpu', weights_only=True)
        assert set(saved['seen_val_metrics_by_stage']) == set(STAGES)
        assert set(saved['train_pure_noise_metrics_by_stage']) == set(STAGES)
        assert set(saved['unseen_val_metrics_by_stage']) == set(STAGES)
        if PAPER:
            histogram = saved['training_timestep_counts']
            assert len(histogram) == 100 and sum(histogram) == 12 * 7
            assert all(sum(histogram[start:start + 25]) > 0 for start in range(0, 100, 25))
        (directory / 'passed.json').write_text(json.dumps({'ddp_smoke': True, 'strict_reload': True, 'all_stages_evaluated': True}) + '\n')
        print('DDP_SMOKE_PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('audit', 'smoke', 'train'))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--paper', action='store_true', help='100-step paper-style schedule, deterministic DDPM, fixed seen5000')
    args = parser.parse_args()
    if args.paper:
        PAPER = True
        OUTPUT = ROOT / 'outputs/no_z_diffusion_paper_v2'
    os.chdir(SOURCE)
    if args.mode == 'audit':
        audit()
    else:
        run_training(smoke=args.mode == 'smoke', epochs=args.epochs)
