"""Local staged Fibo factory: load, attach, cache, train and render in one session."""
from particle_sliders.lifecycle import Backend, Binding, Capabilities, checkpoint_cleanup


def validate(options):
    allowed = {'model_dir', 'manifest', 'cache_dir', 'index_sha256', 'response_scale',
               'steps', 'batch', 'seed', 'device', 'base_precision', 'ema_decay', 'pairs', 'dummy'}
    if set(options) - allowed:
        raise ValueError('unknown Fibo session option')
    if not isinstance(options.get('manifest'), dict) or not options.get('cache_dir'):
        raise ValueError('Fibo requires a local manifest and cache_dir')
    if not options.get('dummy') and not options.get('index_sha256') and not options.get('pairs'):
        raise ValueError('supply an existing cache index SHA256 or local capture pairs')
    if options.get('dummy') and (options['manifest'].get('qualification') != 'CPU fixture only' or
                                 options.get('device', 'cpu') != 'cpu'):
        raise ValueError('Fibo mock requires CPU and a CPU-fixture-only manifest')
    from math import isfinite
    if not isfinite(options.get('response_scale', 0)) or options.get('response_scale', 0) <= 0:
        raise ValueError('Fibo response_scale must be positive and finite')
    from particle_sliders.fibo_native import _identity
    _identity(options['manifest'])


def prepare(options, *, pipeline=None):
    from PIL import Image
    from particle_sliders.fibo_native import (
        staged_pipeline, target_modules, FiboParticleSlider, FiboEditResponseAdapter,
        FiboCachedRows, write_cache, build_fibo_game,
    )
    manifest = options['manifest']
    if options.get('dummy'):
        from .synthetic import Pipeline
        pipeline = Pipeline()
    if pipeline is None:
        pipeline = staged_pipeline(options['model_dir'], manifest, device=options.get('device', 'cpu'))
    restore = checkpoint_cleanup(pipeline.transformer)
    targets = [(module, module.forward) for module in target_modules(pipeline.transformer).values()]
    # Fibo projections replace target forwards; cleanup must undo attachment.
    def close():
        for module, original in targets:
            module.forward = original
        restore()
    try:
        slider = FiboParticleSlider(pipeline.transformer, seed=options.get('seed', 7))
        adapter = FiboEditResponseAdapter(pipeline, slider, model_identity=manifest)
        checksum = options.get('index_sha256')
        if checksum is None and options.get('dummy'):
            from pathlib import Path
            from hashlib import sha256
            index = Path(options['cache_dir']) / 'index.json'
            if index.exists():
                checksum = sha256(index.read_bytes()).hexdigest()
            else:
                images = [Image.new('RGB', (8, 8), (64, 20, 30))]
                rows = adapter.capture_pair('keep jacket', 'make jacket red', images)
                checksum = write_cache(options['cache_dir'], rows, model_identity=manifest)
        if checksum is None:
            rows = []
            for pair in options['pairs']:
                images = []
                for path in pair['images']:
                    with Image.open(path) as image:
                        images.append(image.copy())
                mask = None
                if pair.get('mask'):
                    with Image.open(pair['mask']) as image:
                        mask = image.copy()
                rows.extend(adapter.capture_pair(pair['neutral'], pair['positive'], images,
                    seed=pair.get('seed', 7), mask=mask))
            checksum = write_cache(options['cache_dir'], rows, model_identity=manifest)
        provider = FiboCachedRows(options['cache_dir'], index_sha256=checksum,
            model_identity=manifest, device=options.get('device', 'cpu'))
        game = build_fibo_game(adapter, provider, response_scale=options['response_scale'],
            total_steps=options.get('steps', 4), batch_size=options.get('batch', 2), seed=options.get('seed', 7))
        def render(*, prompt, strength, seed, images, **native):
            return adapter.render_edit(prompt, images, scale=strength, seed=seed, **native)
        return Binding(game, close, render)
    except BaseException:
        close()
        raise


def capabilities(options):
    return Capabilities('routed_particle_gan', render=True, reference_images=True,
        references_required=True, render_outputs=('latent',) if options.get('dummy') else ('pil', 'latent'),
        base_precisions=('float32',) if options.get('dummy') else ('bfloat16',))


BACKEND = Backend('fibo', capabilities({}), prepare, validate, capabilities)


def synthetic_config(cache_dir, *, steps=4, cohort='fibo-synthetic-v1'):
    """Config for the installed CPU example; the factory hides all construction."""
    from particle_sliders.lifecycle import SessionConfig
    return SessionConfig('fibo', cohort, dummy=True, cache_dir=str(cache_dir),
        manifest={'qualification': 'CPU fixture only', 'fixture': 'bounded Fibo EDIT protocol'},
        response_scale=1., steps=steps, ema_decay=.9)
