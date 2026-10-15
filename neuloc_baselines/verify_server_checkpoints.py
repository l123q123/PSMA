"""Strictly load server-only weights and compare archived/reference ray outputs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset

from neuloc_lvsm.data import EpisodeDataset
from neuloc_lvsm.evaluate import load_model


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('verification requires physical GPU1 only')
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    source = args.source.resolve()
    spec = importlib.util.spec_from_file_location('reference_neuloc_lvsm', source / 'neuloc_lvsm/__init__.py',
        submodule_search_locations=[str(source / 'neuloc_lvsm')])
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    reference_loader = importlib.import_module('reference_neuloc_lvsm.evaluate').load_model
    dataset = EpisodeDataset(source / 'data/global100_overlap_q96_verified/seen_val.json', image_size=256, support_views=0)
    batch = next(iter(DataLoader(Subset(dataset, [0]), batch_size=1)))
    batch = {k: v.cuda() for k, v in batch.items()}
    inputs = tuple(batch[k] for k in ('support_images', 'support_intrinsics', 'support_poses', 'query_image', 'query_intrinsics'))
    kwargs = dict(support_type_ids=batch['support_type_ids'], support_mask=batch['support_mask'],
                  view_token_count=80, support_token_budget=8192)
    provenance = {'archive_date': '2026-10-15', 'source_project': str(source),
                  'source_policy': 'Pre-direct-support compatibility snapshot; checkpoint compatibility and numerical outputs are verified, not byte-identical historical training source.',
                  'checkpoints': {}}
    validation = dict(device='physical GPU1', torch_version=str(torch.__version__),
                      batch_shape=list(batch['support_images'].shape), profiles={})
    for profile in ('v3', 'diffusion_v1'):
        config = json.loads((root / 'configs' / f'{profile}.json').read_text())
        checkpoint = source / config['checkpoint_reference']
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = load_model(checkpoint, torch.device('cpu'), 256)
        assert set(payload['model']) == set(model.state_dict())
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in payload['model'].items())
        provenance['checkpoints'][profile] = dict(path=str(checkpoint), sha256=sha256(checkpoint),
            epoch=payload.get('epoch'), loss=payload.get('loss'), model_config=payload.get('model_config', {}),
            seen_val_metrics=payload.get('seen_val_metrics'), seen_val_metrics_by_stage=payload.get('seen_val_metrics_by_stage'))
        del payload
        model.cuda().eval()
        with torch.inference_mode():
            output = (model.sample(*inputs, **kwargs, sample_steps=16,
                generator=torch.Generator(device='cuda').manual_seed(91)) if profile == 'diffusion_v1' else model(*inputs, **kwargs))
        archived = (output.latent.cpu(), output.directions.cpu(), output.moments.cpu())
        del model, output
        gc.collect()
        torch.cuda.empty_cache()
        reference = reference_loader(checkpoint, torch.device('cuda'), 256)
        with torch.inference_mode():
            output = (reference.sample(*inputs, **kwargs, sample_steps=16,
                generator=torch.Generator(device='cuda').manual_seed(91)) if profile == 'diffusion_v1' else reference(*inputs, **kwargs))
        comparison = (output.latent.cpu(), output.directions.cpu(), output.moments.cpu())
        deltas = {}
        for name, archived_tensor, reference_tensor in zip(('latent', 'directions', 'moments'), archived, comparison):
            torch.testing.assert_close(archived_tensor, reference_tensor, atol=1e-6, rtol=1e-5)
            deltas[name + '_max_abs_delta'] = float((archived_tensor - reference_tensor).abs().max())
        validation['profiles'][profile] = dict(strict_state_load=True, all_checkpoint_tensors_equal=True,
            latent_shape=list(archived[0].shape), **deltas)
        print('VERIFIED', profile, validation['profiles'][profile], flush=True)
        del reference, output
        gc.collect()
        torch.cuda.empty_cache()
    provenance['archived_source_sha256'] = {str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob('*.py')) if '__pycache__' not in path.parts}
    (root / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (root / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print('ARCHIVE_CHECKPOINT_VERIFICATION_PASSED', flush=True)


if __name__ == '__main__':
    main()
