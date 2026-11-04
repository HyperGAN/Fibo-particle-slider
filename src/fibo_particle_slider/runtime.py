"""Exact modern cohort admission; no runtime replacement or network fallback."""
from importlib.metadata import version

PROFILE = 'hypergan-hndl-v2'
CORE_COMMIT = 'ddb41423bcefc1fe76cd94d0f9b367353b3ea137'
CORE_VERSION = '0.12.0'


def validate_profile(profile):
    if profile not in ('legacy', PROFILE):
        raise ValueError('unsupported Fibo network profile')
    import hashlib
    import json
    from importlib.resources import files
    import particle_sliders
    from pathlib import Path
    lock = json.loads(files('fibo_particle_slider').joinpath('backend.lock.json').read_text())
    if lock['shared_commit'] != CORE_COMMIT or lock['shared_version'] != CORE_VERSION:
        raise ValueError('Fibo source lock differs from the cleared modern cohort')
    source_root = Path(particle_sliders.__file__).resolve().parent.parent
    actual = {}
    for package in ('particle_sliders', 'concept_slider_core'):
        for path in (source_root / package).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                actual[path.relative_to(source_root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != lock['shared_source_sha256']:
        raise ValueError('Fibo core source closure differs; use the exact pinned cohort')
    from particle_sliders.pg094 import PG_VERSION, PG_SOURCE_PROFILE
    from particle_sliders.pg09 import _released_sources
    if version('particle-sliders-core') != CORE_VERSION or PG_VERSION != '0.9.4':
        raise ValueError('Fibo modern cohort requires core0.12.0 and ParticleGAN0.9.4')
    _released_sources(profile=PG_SOURCE_PROFILE)
    if profile == PROFILE:
        from particle_sliders.fibo_hndl import PROFILE as actual, verified_builder
        if actual != PROFILE:
            raise ValueError('Fibo HNDL profile differs from this cohort')
        verified_builder()


COHORTS = {
    PROFILE: 'fibo-hypergan-hndl-v2-cpu',
    'legacy': 'fibo-core012-legacy-network-cpu',
}


def validate_cohort(cohort, profile):
    """Reserve source/profile cohort names; custom experiment names stay opaque."""
    if not isinstance(cohort, str) or not cohort:
        raise ValueError('Fibo requires a nonempty cohort name')
    modern_name = cohort == COHORTS[PROFILE] or cohort.startswith(COHORTS[PROFILE] + '-')
    if (cohort.startswith('fibo-hypergan-hndl-') and (profile != PROFILE or not modern_name) or
            cohort.startswith('fibo-core012-legacy-') and profile != 'legacy' or
            cohort.startswith(('fibo-core082-', 'fibo-core081-'))):
        raise ValueError('Fibo cohort name differs from the selected network/source profile')
