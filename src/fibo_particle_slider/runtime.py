"""Exact modern cohort admission; no runtime replacement or network fallback."""
from importlib.metadata import version

PROFILE = 'hypergan-hndl-v2'


def validate_profile(profile):
    if profile not in ('legacy', PROFILE):
        raise ValueError('unsupported Fibo network profile')
    import hashlib
    import json
    from importlib.resources import files
    import particle_sliders
    from pathlib import Path
    lock = json.loads(files('fibo_particle_slider').joinpath('backend.lock.json').read_text())
    source_root = Path(particle_sliders.__file__).resolve().parent.parent
    if any(not (source_root / name).is_file() or
           hashlib.sha256((source_root / name).read_bytes()).hexdigest() != digest
           for name, digest in lock['shared_source_sha256'].items()):
        raise ValueError('Fibo core source differs; use the exact pinned cohort')
    from particle_sliders.pg094 import PG_VERSION, PG_SOURCE_PROFILE
    from particle_sliders.pg09 import _released_sources
    if version('particle-sliders-core') != '0.12.0' or PG_VERSION != '0.9.4':
        raise ValueError('Fibo modern cohort requires core0.12.0 and ParticleGAN0.9.4')
    _released_sources(profile=PG_SOURCE_PROFILE)
    if profile == PROFILE:
        from particle_sliders.fibo_hndl import PROFILE as actual, verified_builder
        if actual != PROFILE:
            raise ValueError('Fibo HNDL profile differs from this cohort')
        verified_builder()
