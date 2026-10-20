"""Thin standalone Fibo derivative; the reviewed shared core owns all behavior."""
from particle_sliders.fibo_native import (
    DEFAULTS, DIFFUSERS_COMMIT, FORMAT, MODEL_ID, MODEL_REVISION,
    FiboCachedRows, FiboEditResponseAdapter, FiboParticleSlider,
    build_fibo_game, staged_pipeline, write_cache,
)

__all__ = [
    'DEFAULTS', 'DIFFUSERS_COMMIT', 'FORMAT', 'MODEL_ID', 'MODEL_REVISION',
    'FiboCachedRows', 'FiboEditResponseAdapter', 'FiboParticleSlider',
    'build_fibo_game', 'staged_pipeline', 'write_cache',
]
__version__ = '0.1.0'
