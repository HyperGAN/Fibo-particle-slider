"""Complete installed Fibo lifecycle using staged CPU EDIT mocks and real PG."""
import torch
import pytest
from particle_sliders.lifecycle import open_session, UnsupportedOperation
from particle_sliders.pg09 import _hash
from fibo_particle_slider.lifecycle import synthetic_config


def test_factory_training_resume_export_evaluate_and_cleanup(tmp_path):
    torch.set_num_threads(1)
    config = synthetic_config(tmp_path / 'cache')
    before = torch.get_rng_state().clone()
    with open_session(config) as full:
        assert torch.equal(before, torch.get_rng_state())
        pipeline = full.game.backend.pipeline
        projections = list(full.game.backend.slider.adapters.values())
        originals = [(projection.org_forward.__self__, projection.org_forward) for projection in projections]
        full.train()
        expected = full.checkpoint()['sha256']
    assert all(module.forward == original for module, original in originals)
    with open_session(config) as split:
        split.train(stop_after=2)
        split.checkpoint(tmp_path / 'training.pt')
    with open_session(config, resume=tmp_path / 'training.pt') as resumed:
        resumed.train()
        assert resumed.checkpoint()['sha256'] == expected
        before_observe = resumed.checkpoint()['sha256']
        for strength in (0., .5, 1.):
            assert torch.isfinite(resumed.evaluate(strength=strength)).all()
        assert before_observe == resumed.checkpoint()['sha256']
        resumed.export(tmp_path / 'adapter.pt', selection='ema')
        expected_ema = _hash(resumed.ema.values)
    with open_session(config, artifact=tmp_path / 'adapter.pt', operation='evaluate') as loaded:
        from PIL import Image
        assert _hash(loaded.game.network.state_dict()) == expected_ema
        scheduler = loaded.game.backend.pipeline.scheduler
        references = [Image.new('RGB', (8, 8), (64, 20, 30))]
        for strength in (0., .5, 1.):
            output = loaded.render(prompt='keep jacket', images=references, strength=strength,
                                   output_type='latent')
            assert torch.isfinite(output.images[0]).all()
        assert loaded.game.backend.pipeline.scheduler is scheduler and scheduler.calls == 0
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.cuda.is_initialized()


def test_render_preflight_requires_reference_images(tmp_path):
    config = synthetic_config(tmp_path / 'cache')
    with pytest.raises(UnsupportedOperation, match='reference'):
        open_session(config, operation='render')
    assert not (tmp_path / 'cache').exists()


@pytest.mark.parametrize('change', [
    {'dummy': 'yes'}, {'response_scale': True}, {'response_scale': '1'},
    {'index_sha256': 'not-a-checksum'}, {'pairs': []},
])
def test_invalid_factory_options_fail_before_pipeline_allocation(tmp_path, monkeypatch, change):
    from particle_sliders.lifecycle import SessionConfig
    from fibo_particle_slider.synthetic import Pipeline
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid configuration allocated a pipeline')
    monkeypatch.setattr(Pipeline, '__init__', forbidden)
    options = synthetic_config(tmp_path / 'cache').options
    options.update(change)
    with pytest.raises(ValueError):
        open_session(SessionConfig('fibo', 'invalid-fibo-preflight', **options))
    assert not (tmp_path / 'cache').exists()


@pytest.mark.parametrize('setup_failure', [False, True])
def test_supplied_mock_pipeline_restores_host_ownership(tmp_path, monkeypatch, setup_failure):
    import particle_sliders.fibo_native as native
    from fibo_particle_slider.lifecycle import BACKEND
    from fibo_particle_slider.synthetic import Pipeline
    pipeline = Pipeline()
    pipeline.transformer.train()
    pipeline.transformer.noise_refiner[0].eval()
    pipeline.vae.eval()
    next(pipeline.text_encoder.parameters()).requires_grad_(False)
    pipeline.transformer.gradient_checkpointing = False
    original_checkpoint = object()
    pipeline.transformer._gradient_checkpointing_func = original_checkpoint
    roots = [pipeline.transformer, pipeline.text_encoder, pipeline.vae]
    modes = [(m, m.training) for root in roots for m in root.modules()]
    flags = [(p, p.requires_grad) for root in roots for p in root.parameters()]
    forwards = [(m, m.forward) for m in native.target_modules(pipeline.transformer).values()]
    if setup_failure:
        def fail(*args, **kwargs):
            raise RuntimeError('injected game setup failure')
        monkeypatch.setattr(native, 'build_fibo_game', fail)
    config = synthetic_config(tmp_path / 'cache', cohort='fibo-owned-pipeline-v2')
    if setup_failure:
        with pytest.raises(RuntimeError, match='injected game setup failure'):
            BACKEND.open(config, pipeline=pipeline)
    else:
        with BACKEND.open(config, pipeline=pipeline) as session:
            assert session.game.backend.pipeline is pipeline
            assert all(not p.requires_grad for p in pipeline.transformer.parameters())
        session.close()
    assert all(m.training == training for m, training in modes)
    assert all(p.requires_grad == requires_grad for p, requires_grad in flags)
    assert all(m.forward == forward for m, forward in forwards)
    assert pipeline.transformer.gradient_checkpointing is False
    assert pipeline.transformer._gradient_checkpointing_func is original_checkpoint
