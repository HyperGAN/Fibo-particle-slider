"""Native-only mock adoption checks, pending the next admitted CPU slot."""
from copy import deepcopy
import shutil

import pytest
import torch
from PIL import Image

from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import UnsupportedOperation
from fibo_particle_slider.inference import INFERENCE
from fibo_particle_slider.lifecycle import BACKEND, synthetic_config


def test_native_only_raw_ema_render_after_training_cache_removed(tmp_path, monkeypatch):
    references = [Image.new('RGB', (8, 8), (64, 20, 30))]
    native = dict(prompt='keep jacket', images=references)
    config = synthetic_config(tmp_path / 'cache', steps=2)
    expected, packets = {}, {}
    with BACKEND.open(config) as training:
        training.train()
        state = deepcopy(training.game.network.state_dict())
        for selection in ('raw', 'ema'):
            packets[selection] = training.export_inference(selection=selection)
            training.game.network.load_state_dict(packets[selection]['network'])
            expected[selection] = training.render(**native, strength=.5, output_type='latent').images[0]
            training.game.network.load_state_dict(state)
    shutil.rmtree(tmp_path / 'cache')
    import particle_sliders.fibo_native as shared
    from particle_sliders.pg09 import ConditionalPG09
    def forbidden(*args, **kwargs):
        raise AssertionError('native-only inference constructed training state')
    for name in ('FiboCachedRows', 'write_cache', 'build_fibo_game'):
        monkeypatch.setattr(shared, name, forbidden)
    monkeypatch.setattr(ConditionalPG09, '__init__', forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, '__init__', forbidden)
    before = torch.get_rng_state().clone()
    for selection, packet in packets.items():
        assert set(packet['specification']['configuration']) == {'dummy'}
        with open_inference(packet, backend=INFERENCE, references=True, output_type='latent',
                            native_request=native) as inference:
            assert inference.selection == selection
            assert torch.equal(inference.render(**native, strength=.5).images[0], expected[selection])
            for strength in (0., .5, 1.):
                assert torch.isfinite(inference.render(**native, strength=strength).images[0]).all()
            assert not any(hasattr(inference, key) for key in ('game', 'rows', 'critic', 'optimizer'))
            with pytest.raises(UnsupportedOperation):
                inference.response()
            with pytest.raises(UnsupportedOperation):
                inference.train()
    assert torch.equal(before, torch.get_rng_state())
    assert not (tmp_path / 'cache').exists()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('damage', ['references', 'mask'])
def test_native_request_preflight_precedes_model_allocation(tmp_path, monkeypatch, damage):
    from fibo_particle_slider.synthetic import Pipeline
    def forbidden(*args, **kwargs):
        raise AssertionError('bad request allocated a pipeline')
    monkeypatch.setattr(Pipeline, '__init__', forbidden)
    native = dict(prompt='keep jacket', images=[Image.new('RGB', (8, 8))])
    if damage == 'references':
        native['images'] *= 5
    else:
        native['mask'] = Image.new('L', (4, 4))
    with pytest.raises(ValueError):
        BACKEND.open(synthetic_config(tmp_path / 'cache'), operation='render',
                     references=True, output_type='latent', native_request=native)
    assert not (tmp_path / 'cache').exists()


def test_native_observation_failure_restores_buffers_and_runtime(tmp_path):
    with BACKEND.open(synthetic_config(tmp_path / 'cache')) as training:
        pipeline = training.game.backend.pipeline
        pipeline.transformer.register_buffer('native_count', torch.zeros(()), persistent=False)
        original = pipeline.transformer.native_count
        original_calls = list(pipeline.calls)
        original_seen = list(pipeline.transformer.seen)
        def fail(**kwargs):
            pipeline.transformer.native_count.add_(1)
            pipeline.temporary_native_state = 'discard'
            pipeline.calls.append('discard')
            pipeline.transformer.seen.append('discard')
            raise RuntimeError('injected native observation failure')
        training.binding.render = fail
        with pytest.raises(RuntimeError, match='injected native observation failure'):
            training.render(prompt='keep jacket', images=[Image.new('RGB', (8, 8))], output_type='latent')
        assert pipeline.transformer.native_count is original and original.item() == 0
        assert 'native_count' in pipeline.transformer._non_persistent_buffers_set
        assert not hasattr(pipeline, 'temporary_native_state')
        assert pipeline.calls == original_calls
        assert pipeline.transformer.seen == original_seen


def test_update_admission_decline_preserves_training_state(tmp_path):
    with BACKEND.open(synthetic_config(tmp_path / 'cache'), before_update=lambda request: False) as training:
        before = training.checkpoint()['sha256']
        with pytest.raises(UnsupportedOperation, match='admission rejected'):
            training.step()
        assert training.checkpoint()['sha256'] == before
