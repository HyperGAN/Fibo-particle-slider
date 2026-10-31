"""Actual HyperGAN/HNDL execution on a tiny EDIT host; no model downloads."""
from copy import deepcopy

import pytest
import torch

from fibo_particle_slider.lifecycle import BACKEND, synthetic_config
from particle_sliders.fibo_hndl import HNDLFieldCritic, HNDLParticleProjection, PROFILE
from particle_sliders.lifecycle import CheckpointError
from particle_sliders.pg09 import _hash


def test_real_hndl_game_updates_only_adapter_bank_and_critic(tmp_path):
    from fibo_particle_slider.synthetic import Pipeline
    pipe = Pipeline()
    frozen = [p for root in (pipe.transformer, pipe.text_encoder, pipe.vae) for p in root.parameters()]
    originals = [p.detach().clone() for p in frozen]
    with BACKEND.open(synthetic_config(tmp_path / 'cache'), pipeline=pipe) as session:
        slider = session.game.backend.slider
        assert slider.network_profile == PROFILE == 'hypergan-hndl-v2'
        assert session.config.cohort == 'fibo-hypergan-hndl-v2-cpu'
        assert isinstance(session.game.critic, HNDLFieldCritic)
        assert all(isinstance(p, HNDLParticleProjection) for p in slider.adapters.values())
        assert session.game._signature['pg_version'] == '0.9.4'
        assert session.game._signature['penalty_input_domain'] == 'auto'
        bank = slider.particles.detach().clone()
        critic = _hash(session.game.critic.state_dict())
        session.train()
        assert not torch.equal(bank, slider.particles)
        assert critic != _hash(session.game.critic.state_dict())
        assert all(not p.requires_grad and p.grad is None for p in frozen)
        assert all(torch.equal(p, value) for p, value in zip(frozen, originals))
        parameters = list(slider.parameters())
        assert len(parameters) == len({id(p) for p in parameters})
        assert sum(p is slider.particles for p in parameters) == 1
        projection = next(iter(slider.adapters.values()))
        x = torch.randn(1, 2, projection.input_width, requires_grad=True)
        with slider.scaled(0.):
            assert torch.equal(projection(x), projection.org_forward(x))
        with slider.scaled(1.):
            result = projection(x)
            bank_gradient, input_gradient = torch.autograd.grad(result.sum(), (slider.particles, x))
        assert torch.isfinite(bank_gradient).all() and bank_gradient.abs().sum() > 0
        assert torch.isfinite(input_gradient).all() and input_gradient.abs().sum() > 0


def test_network_and_pg_profile_restore_refusals_are_transactional(tmp_path):
    with BACKEND.open(synthetic_config(tmp_path / 'hndl')) as modern:
        modern.step()
        before = modern.checkpoint()
        with BACKEND.open(synthetic_config(tmp_path / 'legacy', network_profile='legacy')) as legacy:
            previous = legacy.checkpoint()['sha256']
            with pytest.raises(CheckpointError):
                legacy.restore(before)
            assert legacy.checkpoint()['sha256'] == previous
            with pytest.raises(CheckpointError):
                modern.restore(legacy.checkpoint())
            assert modern.checkpoint()['sha256'] == before['sha256']
        old_pg = deepcopy(modern.game.state_dict())
        old_pg['config']['pg_version'] = '0.9.1'
        unchanged = _hash(modern.game.state_dict())
        with pytest.raises(ValueError, match='original pinned'):
            modern.game.load_state_dict(old_pg)
        assert _hash(modern.game.state_dict()) == unchanged


def test_hndl_source_failure_precedes_host_allocation(tmp_path, monkeypatch):
    import particle_sliders.fibo_hndl as hndl
    from fibo_particle_slider.synthetic import Pipeline
    config = synthetic_config(tmp_path / 'cache')
    def forbidden(*args, **kwargs):
        raise AssertionError('unverified builder allocated a model')
    monkeypatch.setattr(Pipeline, '__init__', forbidden)
    original = hndl._sha
    monkeypatch.setattr(hndl, '_sha', lambda p: '0' * 64 if str(p).endswith('hndl_networks.py') else original(p))
    with pytest.raises(ValueError, match='builder closure'):
        BACKEND.open(config)
    assert not (tmp_path / 'cache').exists()


def test_text_to_image_is_rejected_before_allocation(tmp_path, monkeypatch):
    from fibo_particle_slider.synthetic import Pipeline
    config = synthetic_config(tmp_path / 'cache')
    def forbidden(*args, **kwargs):
        raise AssertionError('text-to-image request allocated an EDIT model')
    monkeypatch.setattr(Pipeline, '__init__', forbidden)
    with pytest.raises(ValueError, match='EDIT requires'):
        BACKEND.open(config, operation='render', references=True,
                     native_request={'prompt': 'a red jacket', 'images': []})
