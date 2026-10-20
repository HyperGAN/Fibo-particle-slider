"""Standalone distribution gates; never load a model or require native packages."""
from email import message_from_bytes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest


@pytest.fixture(scope='module')
def distribution(tmp_path_factory):
    root = tmp_path_factory.mktemp('fibo-distribution')
    source = root / 'source'
    shutil.copytree(Path(__file__).resolve().parents[1], source,
        ignore=shutil.ignore_patterns('.git', 'build', 'dist', '*.egg-info', '__pycache__', '.pytest_cache'))
    script = '''
from pathlib import Path
from setuptools import build_meta
import os, tarfile
root = Path.cwd().parent
dist = root / 'dist'
dist.mkdir()
archive = dist / build_meta.build_sdist(str(dist))
with tarfile.open(archive) as handle:
    handle.extractall(root / 'sdist')
os.chdir(next((root / 'sdist').iterdir()))
build_meta.build_wheel(str(dist))
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=source,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return root


def test_distribution_retains_sources_and_no_weights(distribution):
    archive, = (distribution / 'dist').glob('*.tar.gz')
    with tarfile.open(archive) as handle:
        names = {str(Path(name).relative_to(Path(name).parts[0])) for name in handle.getnames()}
    assert {'README.md', 'NATIVE-GUIDE.md', 'backend.lock.json', 'validation.json', 'VALIDATION.md',
            'configs/jacket-edit.json', 'tests/test_distribution.py'} <= names
    wheel, = (distribution / 'dist').glob('*.whl')
    with zipfile.ZipFile(wheel) as handle:
        names = set(handle.namelist())
        assert {'fibo_particle_slider/backend.lock.json', 'fibo_particle_slider/fibo_provenance.json',
                'fibo_particle_slider/example_config.json', 'fibo_particle_slider/NATIVE-GUIDE.md'} <= names
        assert not any(name.endswith(('.safetensors', '.pt', '.pth', '.ckpt')) for name in names)
        metadata_path, = (name for name in names if name.endswith('.dist-info/METADATA'))
        metadata = message_from_bytes(handle.read(metadata_path))
        dependency = next(value for value in metadata.get_all('Requires-Dist') if value.startswith('particle-sliders-core'))
        lock = json.loads(handle.read('fibo_particle_slider/backend.lock.json'))
        assert lock['shared_commit'] in dependency and '#subdirectory=packages/particle-sliders-core' in dependency


def test_installed_wheel_identity_and_offline_import(distribution):
    import particle_sliders
    import particlegan
    wheel, = (distribution / 'dist').glob('*.whl')
    installed = distribution / 'installed'
    with zipfile.ZipFile(wheel) as handle:
        handle.extractall(installed)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join((str(installed),
            str(Path(particle_sliders.__file__).resolve().parent.parent),
            str(Path(particlegan.__file__).resolve().parent.parent))))
    script = '''
import importlib.abc, json, hashlib, sys
from importlib.resources import files
from pathlib import Path
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'diffusers', 'transformers', 'peft', 'conceptmod', 'huggingface_hub'}:
            raise AssertionError('unexpected optional native import: ' + fullname)
sys.meta_path.insert(0, NoNative())
import fibo_particle_slider as product
import particle_sliders.fibo_native as shared
from particle_sliders.pg09 import _released_sources
import torch
assert Path(product.__file__).is_relative_to(Path.cwd() / 'installed')
assert product.FiboParticleSlider is shared.FiboParticleSlider
assert product.FiboEditResponseAdapter is shared.FiboEditResponseAdapter
lock = json.loads(files(product).joinpath('backend.lock.json').read_text())
assert hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest() == lock['shared_backend_sha256']
assert _released_sources()
assert product.MODEL_ID == lock['model_id'] == 'briaai/Fibo-2.0-edit-turbo'
assert product.MODEL_REVISION == lock['model_revision']
assert product.DEFAULTS == {'num_inference_steps': 4, 'guidance_scale': 1.0}
assert not torch.cuda.is_initialized()
print('offline installed wrapper and verified released PG source identity passed')
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=distribution,
                            env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_source_and_resources_contain_no_local_receipts_or_private_paths():
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / 'backend.lock.json').read_text())
    packaged = json.loads((root / 'src/fibo_particle_slider/backend.lock.json').read_text())
    assert lock == packaged
    for path in [*root.glob('*.md'), *root.glob('*.json'), *root.glob('configs/*.json'),
                 *root.glob('src/fibo_particle_slider/*.json'), *root.glob('src/fibo_particle_slider/*.md')]:
        text = path.read_text()
        assert not any(prefix in text for prefix in ('/ml2/', '/home/', '/mnt/', 'ghp_', 'gho_'))
    assert hashlib.sha256((root / 'src/fibo_particle_slider/fibo_provenance.json').read_bytes()).hexdigest() == lock['provenance_sha256']
