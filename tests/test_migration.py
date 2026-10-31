"""Product checks for the exact a794 modern source cohort; no native weights."""
from copy import deepcopy

from PIL import Image
import pytest
import torch

from fibo_particle_slider.lifecycle import BACKEND, synthetic_config
from fibo_particle_slider.inference import INFERENCE
from fibo_particle_slider.runtime import CORE_COMMIT
from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.pg09 import _hash


def test_native_packet_binds_exact_core_and_rejects_old_preview_configuration(tmp_path, monkeypatch):
    from fibo_particle_slider.synthetic import Pipeline
    with BACKEND.open(synthetic_config(tmp_path / 'cache')) as session:
        packet = session.export_inference()
    assert packet['specification']['configuration']['core_commit'] == CORE_COMMIT
    old = deepcopy(packet)
    old['specification']['configuration'].pop('core_commit')
    old['sha256'] = _hash({key: value for key, value in old.items() if key != 'sha256'})
    def forbidden(*args, **kwargs):
        raise AssertionError('old preview specification allocated a pipeline')
    monkeypatch.setattr(Pipeline, '__init__', forbidden)
    with pytest.raises(ValueError, match='native-only configuration'):
        open_inference(old, backend=INFERENCE, references=True, output_type='latent',
                       native_request=dict(prompt='keep jacket', images=[Image.new('RGB', (8, 8))]))


def test_ema_disabled_raw_resume_and_exact_off_render(tmp_path):
    config = synthetic_config(tmp_path / 'cache', steps=2, cohort='fibo-a794-ema-disabled')
    options = config.options
    options['ema_decay'] = None
    config = SessionConfig(config.backend, config.cohort, **options)
    references = [Image.new('RGB', (8, 8), (64, 20, 30))]
    with BACKEND.open(config) as session:
        off = session.render(prompt='keep jacket', images=references, strength=0., output_type='latent').images[0]
        session.step()
        packet = session.checkpoint()
        assert session.ema.values is None
    with BACKEND.open(config) as resumed:
        resumed.restore(packet)
        resumed.train()
        assert torch.equal(off, resumed.render(prompt='keep jacket', images=references,
                                              strength=0., output_type='latent').images[0])
        raw = resumed.export_inference(selection='raw')
        with pytest.raises(ValueError, match='configured EMA'):
            resumed.export_inference(selection='ema')
    with open_inference(raw, backend=INFERENCE, references=True, output_type='latent',
                        native_request=dict(prompt='keep jacket', images=references)) as inference:
        assert torch.equal(off, inference.render(prompt='keep jacket', images=references, strength=0.).images[0])


@pytest.mark.parametrize('trigger', ['close', 'step'])
def test_product_cleanup_failure_is_retryable_and_preserves_primary_error(tmp_path, monkeypatch, trigger):
    import fibo_particle_slider.inference as module
    from fibo_particle_slider.synthetic import Pipeline
    from particle_sliders.fibo_native import target_modules
    pipe = Pipeline()
    roots = [pipe.transformer, pipe.text_encoder, pipe.vae]
    modes = [(m, m.training) for root in roots for m in root.modules()]
    flags = [(p, p.requires_grad) for root in roots for p in root.parameters()]
    forwards = [(m, m.forward) for m in target_modules(pipe.transformer).values()]
    original = module.native_cleanup
    attempts = []
    def cleanup(root):
        restore = original(root)
        def retryable():
            if root is pipe.transformer:
                attempts.append(1)
                if len(attempts) == 1:
                    raise RuntimeError('injected product cleanup failure')
            restore()
        return retryable
    monkeypatch.setattr(module, 'native_cleanup', cleanup)
    session = BACKEND.open(synthetic_config(tmp_path / 'cache'), pipeline=pipe)
    try:
        if trigger == 'step':
            def fail():
                raise RuntimeError('primary update failure')
            monkeypatch.setattr(session.game, 'step', fail)
            with pytest.raises(RuntimeError, match='primary update failure') as error:
                session.step()
            assert any('product cleanup failure' in note for note in error.value.__notes__)
        else:
            with pytest.raises(RuntimeError, match='product cleanup failure'):
                session.close()
        assert not session.closed
        session.close()
        assert session.closed and len(attempts) == 2
        assert all(m.training == value for m, value in modes)
        assert all(p.requires_grad == value for p, value in flags)
        assert all(m.forward == value for m, value in forwards)
    finally:
        session.close()
