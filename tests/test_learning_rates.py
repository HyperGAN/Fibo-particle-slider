"""Admit finite LR controls and exercise real shared optimizer/resume behavior."""
import pytest
from fibo_particle_slider.learning_rates import recipe_overrides


@pytest.mark.parametrize('key', ['lr', 'd_lr_mult', 'prior_lr_mult'])
@pytest.mark.parametrize('value', [True, False, 0, -1, float('nan'), float('inf'), '0.001'])
def test_invalid_rates(key, value):
    with pytest.raises(ValueError):
        recipe_overrides(**{key: value})


def test_optional_rates_keep_preset_defaults():
    assert recipe_overrides() == {}


def test_actual_generator_critic_bank_rates_and_exact_resume(tmp_path):
    from fibo_particle_slider.lifecycle import BACKEND, synthetic_config
    from particle_sliders.lifecycle import CheckpointError, SessionConfig
    config = synthetic_config(tmp_path/'cache', lr=2e-5, d_lr_mult=3, prior_lr_mult=4)
    with BACKEND.open(config) as full:
        game = full.game
        assert (game.recipe.lr, game.recipe.d_lr_mult, game.recipe.prior_lr_mult) == (2e-5,3,4)
        bank = game.backend.slider.particles
        for group in game.opt_g.param_groups:
            for parameter in group['params']:
                assert group['lr'] == pytest.approx(8e-5 if parameter is bank else 2e-5)
        assert all(group['lr'] == pytest.approx(6e-5) for group in game.opt_d.param_groups)
        full.train()
        expected = full.checkpoint()['sha256']
    with BACKEND.open(config) as partial:
        partial.train(stop_after=2)
        packet = partial.checkpoint()
    with BACKEND.open(config) as resumed:
        resumed.restore(packet)
        resumed.train()
        assert resumed.checkpoint()['sha256'] == expected
    changed = SessionConfig(config.backend, config.cohort, **{**config.options, 'lr':3e-5})
    with BACKEND.open(changed) as changed_session:
        with pytest.raises(CheckpointError):
            changed_session.restore(packet)


@pytest.mark.parametrize('cohort',['fibo-hypergan-hndl-v2-cpu','fibo-core012-legacy-network-cpu'])
def test_pre_lr_cohort_names_require_archived_checkout(cohort):
    from fibo_particle_slider.runtime import validate_cohort
    with pytest.raises(ValueError, match='archived checkout'):
        validate_cohort(cohort,'hypergan-hndl-v2')
