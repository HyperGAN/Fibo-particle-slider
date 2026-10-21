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
