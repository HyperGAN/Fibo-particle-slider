"""Review regressions: source closure, cohort labels and output capabilities."""
from importlib.resources import files
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

from fibo_particle_slider.inference import INFERENCE
from fibo_particle_slider.lifecycle import BACKEND, synthetic_config
from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import SessionConfig


@pytest.mark.parametrize('damage', ['extra.py', 'extra.pyc', 'extra.so', 'missing', 'changed'])
def test_source_closure_rejects_additions_and_mutations_before_allocation(tmp_path, monkeypatch, damage):
    import particle_sliders
    from fibo_particle_slider.synthetic import Pipeline
    config = synthetic_config(tmp_path / 'cache')
    lock = json.loads(files('fibo_particle_slider').joinpath('backend.lock.json').read_text())
    original = Path(particle_sliders.__file__).resolve().parent.parent
    root = tmp_path / 'core'
    for name in lock['shared_source_sha256']:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / name, destination)
    if damage == 'missing':
        (root / 'particle_sliders/fibo_native.py').unlink()
    elif damage == 'changed':
        (root / 'particle_sliders/fibo_native.py').write_text('# changed source\n')
    else:
        (root / 'particle_sliders' / damage).write_bytes(b'unlisted module')
    def forbidden(*args, **kwargs):
        raise AssertionError('bad core closure allocated a model')
    monkeypatch.setattr(Pipeline, '__init__', forbidden)
    monkeypatch.setattr(particle_sliders, '__file__', str(root / 'particle_sliders/__init__.py'))
    with pytest.raises(ValueError, match='source closure differs'):
        BACKEND.open(config)
    assert not (tmp_path / 'cache').exists()


@pytest.mark.parametrize('profile,cohort', [
    ('legacy', 'fibo-hypergan-hndl-v2-cpu'),
    ('hypergan-hndl-v2', 'fibo-core012-legacy-network-cpu'),
    ('hypergan-hndl-v2', 'fibo-core082-synthetic-v1'),
    ('hypergan-hndl-v2', 'fibo-hypergan-hndl-v1-cpu'),
])
def test_reserved_cohort_names_cannot_rebind_network_profile(tmp_path, monkeypatch, profile, cohort):
    from fibo_particle_slider.synthetic import Pipeline
    with pytest.raises(ValueError, match='cohort name differs'):
        synthetic_config(tmp_path / 'cache', network_profile=profile, cohort=cohort)
    # JSON/manual configuration must enforce the same rule as the helper.
    options = synthetic_config(tmp_path / 'cache', network_profile=profile).options
    def forbidden(*args, **kwargs):
        raise AssertionError('mislabeled cohort allocated a model')
    monkeypatch.setattr(Pipeline, '__init__', forbidden)
    with pytest.raises(ValueError, match='cohort name differs'):
        BACKEND.open(SessionConfig('fibo', cohort, **options))
    assert not (tmp_path / 'cache').exists()


def test_custom_cohorts_preserve_explicit_names_and_profile_identity(tmp_path):
    assert synthetic_config(tmp_path / 'a', cohort='owned-edit-run').cohort == 'owned-edit-run'
    assert synthetic_config(tmp_path / 'b', network_profile='legacy').cohort == 'fibo-core012-legacy-network-cpu'
    assert synthetic_config(tmp_path / 'c').cohort == 'fibo-hypergan-hndl-v2-cpu'


def test_fixture_pil_output_rejected_before_training_and_inference_allocation(tmp_path, monkeypatch):
    from fibo_particle_slider.synthetic import Pipeline
    config = synthetic_config(tmp_path / 'cache', steps=2)
    with BACKEND.open(config) as session:
        packet = session.export_inference()
    references = [Image.new('RGB', (8, 8))]
    native = dict(prompt='keep jacket', images=references, output_type='pil')
    def forbidden(*args, **kwargs):
        raise AssertionError('unsupported PIL output allocated a model')
    monkeypatch.setattr(Pipeline, '__init__', forbidden)
    with pytest.raises(ValueError, match='output_type'):
        BACKEND.open(config, operation='render', references=True, output_type='latent', native_request=native)
    with pytest.raises(ValueError, match='output_type'):
        open_inference(packet, backend=INFERENCE, references=True, output_type='latent', native_request=native)


def test_pil_reference_inputs_remain_valid_for_latent_fixture_outputs(tmp_path):
    references = [Image.new('RGB', (8, 8))]
    native = dict(prompt='keep jacket', images=references, output_type='latent')
    with BACKEND.open(synthetic_config(tmp_path / 'cache'), operation='render', references=True,
                      output_type='latent', native_request=native) as session:
        assert session.capabilities.render_outputs == ('latent',)
        packet = session.export_inference()
        assert session.render(**native).images[0].shape == (1, 4, 2, 2)
    with open_inference(packet, backend=INFERENCE, references=True, output_type='latent',
                        native_request=native) as inference:
        assert inference.render(**native).images[0].shape == (1, 4, 2, 2)
