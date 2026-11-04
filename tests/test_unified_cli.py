"""Installed merged executor keeps the real Fibo game and explicit fixture gate."""
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from fibo_particle_slider.lifecycle import synthetic_config


def test_unified_plan_check_periodic_checkpoints_and_resume(tmp_path):
    config = synthetic_config(tmp_path / 'cache', steps=2)
    configuration = tmp_path / 'config.json'
    configuration.write_text(json.dumps(config.as_dict()))
    profile = Path(__file__).resolve().parents[1] / 'configs/training-fixture-profile.json'
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def cli(operation, *arguments, expected=0):
        result = subprocess.run([sys.executable, '-m', 'particle_sliders.session_cli', operation,
            '--config', str(configuration), '--profile', str(profile), '--no-reporting', *arguments],
            cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=45)
        assert result.returncode == expected, result.stdout + result.stderr
        # Training streams progress before the final JSON receipt.
        return json.loads(result.stdout[result.stdout.index('{'):]) if expected == 0 else result.stderr

    plan = cli('plan')
    assert plan['resolved']['profile']['status'] == 'fixture'
    assert plan['training']['options']['dummy'] is True
    checked = cli('train', '--check')
    assert checked['status'] == 'checked' and checked['qualification'] == 'fixture'
    assert not (tmp_path / 'cache').exists()

    first, resumed = tmp_path / 'first', tmp_path / 'resumed'
    cli('train', '--out', str(first), '--selection', 'ema', '--export-format', 'both', '--stop-after', '1')
    checkpoint = first / 'artifacts/checkpoints/step-000001/training.pt'
    assert checkpoint.is_file()
    cli('train', '--out', str(resumed), '--resume', str(checkpoint.parent),
        '--selection', 'ema', '--export-format', 'both')
    manifest = json.loads((resumed / 'artifacts/manifest.json').read_text())
    assert manifest['completed_steps'] == 2
    assert (resumed / 'artifacts/checkpoints/step-000002/checkpoint.json').is_file()
    packet = torch.load(resumed / 'artifacts/adapter-v2.pt', weights_only=True)
    assert packet['selection'] == 'ema'
    assert packet['specification']['configuration']['dummy'] is True
    from fibo_particle_slider.runtime import CORE_COMMIT
    assert packet['specification']['configuration']['core_commit'] == CORE_COMMIT

    native = config.as_dict()
    native['options']['dummy'] = False
    configuration.write_text(json.dumps(native))
    assert 'fixture profile requires explicit dummy=true' in cli('train', '--check', expected=2)
