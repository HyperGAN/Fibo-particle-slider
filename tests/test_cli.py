"""Installed train/resume -> both exports -> fresh native-only CLI process."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image
import torch

from particle_sliders.session_cli import main
from fibo_particle_slider.lifecycle import synthetic_config


def test_cli_both_exports_resume_and_fresh_native_only_inference(tmp_path):
    config = synthetic_config(tmp_path / 'cache', steps=2)
    config_file = tmp_path / 'config.json'
    config_file.write_text(json.dumps(config.as_dict()))
    first, resumed = tmp_path / 'first', tmp_path / 'resumed'
    main(['train', '--config', str(config_file), '--stop-after', '1', '--out', str(first),
          '--selection', 'ema', '--export-format', 'both'])
    main(['train', '--config', str(config_file), '--resume', str(first / 'training.pt'),
          '--out', str(resumed), '--selection', 'ema', '--export-format', 'both'])
    manifest = json.loads((resumed / 'manifest.json').read_text())
    assert manifest['completed_steps'] == 2
    assert set(manifest['files']) == {'training.pt', 'metrics.jsonl', 'adapter.pt', 'adapter-v2.pt'}
    for name, checksum in manifest['files'].items():
        assert hashlib.sha256((resumed / name).read_bytes()).hexdigest() == checksum
    legacy = torch.load(resumed / 'adapter.pt', weights_only=True)
    native = torch.load(resumed / 'adapter-v2.pt', weights_only=True)
    assert legacy['schema'] == 'particle-sliders-image-session-v1-inference'
    assert native['schema'] == 'particle-sliders-native-inference-v2'
    assert legacy['payload']['selection'] == native['selection'] == 'ema'
    shutil.rmtree(tmp_path / 'cache')
    image_file = tmp_path / 'source.png'
    Image.new('RGB', (8, 8), (64, 20, 30)).save(image_file)
    request_file = tmp_path / 'native.json'
    request_file.write_text(json.dumps(dict(prompt='keep jacket', images=[str(image_file)], output_type='latent')))
    import fibo_particle_slider
    target = str(Path(fibo_particle_slider.__file__).resolve().parent.parent)
    code = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.field_critic import FieldCritic
import particle_sliders.fibo_native as shared
def forbidden(*args, **kwargs):
    raise AssertionError('fresh native inference constructed training state')
ConditionalPG09.__init__ = FieldCritic.__init__ = torch.optim.Optimizer.__init__ = forbidden
shared.FiboCachedRows = shared.write_cache = shared.build_fibo_game = forbidden
from particle_sliders.session_cli import main
artifact, native, output = sys.argv[2:]
main(['infer', '--artifact', artifact, '--native', native, '--strength', '.5', '--seed', '7', '--out', output])
value = torch.load(output, weights_only=True)
assert value['selection'] == 'ema' and value['kind'] == 'native render'
assert value['origin']['completed_steps'] == 2
assert torch.isfinite(value['value']).all() and tuple(value['value'].shape) == (1, 4, 2, 2)
assert not torch.cuda.is_initialized()
'''
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    output_file = tmp_path / 'latent.pt'
    result = subprocess.run([sys.executable, '-I', '-c', code, target, str(resumed / 'adapter-v2.pt'),
                             str(request_file), str(output_file)], cwd='/tmp', env=environment,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(Path(str(output_file) + '.json').read_text())
    assert receipt['selection'] == 'ema' and receipt['artifact_sha256'] == native['sha256']
