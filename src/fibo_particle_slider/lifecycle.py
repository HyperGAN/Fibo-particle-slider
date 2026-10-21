"""Local staged Fibo factory: load, attach, cache, train and render in one session."""
from particle_sliders.lifecycle import Backend, Binding, Capabilities, checkpoint_cleanup


def _local_path(value, name, *, directory=False):
    from pathlib import Path
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Fibo ' + name + ' requires a local path')
    path = Path(value)
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError('Fibo ' + name + ' does not exist locally')
    return path


def validate(options):
    allowed = {'model_dir', 'manifest', 'cache_dir', 'index_sha256', 'response_scale',
               'steps', 'batch', 'seed', 'device', 'base_precision', 'ema_decay', 'pairs', 'dummy'}
    if set(options) - allowed:
        raise ValueError('unknown Fibo session option')
    if type(options.get('dummy', False)) is not bool:
        raise ValueError('Fibo dummy must be a boolean')
    if (not isinstance(options.get('manifest'), dict) or
            not isinstance(options.get('cache_dir'), str) or not options['cache_dir'].strip()):
        raise ValueError('Fibo requires a local manifest and cache_dir')
    if not options.get('dummy') and not options.get('index_sha256') and not options.get('pairs'):
        raise ValueError('supply an existing cache index SHA256 or local capture pairs')
    if options.get('dummy') and (options['manifest'].get('qualification') != 'CPU fixture only' or
                                 options.get('device', 'cpu') != 'cpu'):
        raise ValueError('Fibo mock requires CPU and a CPU-fixture-only manifest')
    if not options.get('dummy'):
        if options['manifest'].get('qualification') == 'CPU fixture only':
            raise ValueError('Fibo native sessions require a verified staging receipt')
        _local_path(options.get('model_dir'), 'model_dir', directory=True)
    from math import isfinite
    scale = options.get('response_scale', 0)
    if type(scale) not in (float, int) or not isfinite(scale) or scale <= 0:
        raise ValueError('Fibo response_scale must be positive and finite')
    from particle_sliders.fibo_native import _identity
    _identity(options['manifest'])
    from pathlib import Path
    checksum = options.get('index_sha256')
    pairs = options.get('pairs')
    if checksum is not None:
        from hashlib import sha256
        import re
        if not isinstance(checksum, str) or re.fullmatch('[0-9a-f]{64}', checksum) is None:
            raise ValueError('Fibo index_sha256 requires a lowercase SHA256')
        if pairs is not None:
            raise ValueError('Fibo cache loading and capture pairs are mutually exclusive')
        index = _local_path(str(Path(options['cache_dir']) / 'index.json'), 'cache index')
        if sha256(index.read_bytes()).hexdigest() != checksum:
            raise ValueError('Fibo cache index checksum differs')
    elif pairs is not None:
        if options.get('dummy'):
            raise ValueError('Fibo mock uses its built-in capture pair')
        if not isinstance(pairs, list) or not pairs:
            raise ValueError('Fibo capture requires nonempty local pairs')
        if Path(options['cache_dir']).exists():
            raise ValueError('Fibo capture requires a new cache directory')
        for pair in pairs:
            if (not isinstance(pair, dict) or set(pair) - {'neutral', 'positive', 'images', 'mask', 'seed'} or
                    any(not isinstance(pair.get(key), str) or not pair[key].strip()
                        for key in ('neutral', 'positive'))):
                raise ValueError('Fibo pair requires neutral and positive edit instructions')
            images = pair.get('images')
            if not isinstance(images, list) or not 1 <= len(images) <= 4:
                raise ValueError('Fibo EDIT requires 1..4 ordered local reference images')
            for image in images:
                _local_path(image, 'reference image')
            if pair.get('mask') is not None:
                if len(images) != 1:
                    raise ValueError('Fibo mask requires exactly one reference image')
                _local_path(pair['mask'], 'mask')
            seed = pair.get('seed', 7)
            if type(seed) is not int or not 0 <= seed < 2**63 - 100:
                raise ValueError('Fibo capture seed is outside the public budget')


def prepare(options, *, pipeline=None):
    from PIL import Image
    from particle_sliders.fibo_native import (
        staged_pipeline, target_modules, FiboParticleSlider, FiboEditResponseAdapter,
        FiboCachedRows, write_cache, build_fibo_game,
    )
    manifest = options['manifest']
    if options.get('dummy') and pipeline is None:
        from .synthetic import Pipeline
        pipeline = Pipeline()
    if pipeline is None:
        pipeline = staged_pipeline(options['model_dir'], manifest, device=options.get('device', 'cpu'))
    restore = checkpoint_cleanup(pipeline.transformer)
    targets = [(module, module.forward) for module in target_modules(pipeline.transformer).values()]
    # A supplied pipeline remains caller-owned, including heterogeneous modes
    # and parameter flags changed when the native adapter freezes components.
    modules, parameters = {}, {}
    for name in ('transformer', 'text_encoder', 'vae'):
        component = getattr(pipeline, name, None)
        if component is not None:
            modules.update({id(module): (module, module.training) for module in component.modules()})
            parameters.update({id(param): (param, param.requires_grad) for param in component.parameters()})
    closed = False
    # Fibo projections replace target forwards; cleanup must undo attachment.
    def close():
        nonlocal closed
        if closed:
            return
        for module, original in targets:
            module.forward = original
        restore()
        for module, training in modules.values():
            module.training = training
        for param, requires_grad in parameters.values():
            param.requires_grad_(requires_grad)
        closed = True
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
                if pair.get('mask') is not None:
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
