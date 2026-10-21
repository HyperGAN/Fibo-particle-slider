"""Native EDIT-only factory; never constructs teachers, a cache or a PG game."""
from contextlib import contextmanager

from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import native_cleanup
from .lifecycle import _local_path, capabilities


def attach(options, *, pipeline=None):
    """Own adapter hooks while borrowing the staged native pipeline components."""
    from particle_sliders.fibo_native import (
        staged_pipeline, target_modules, FiboParticleSlider, FiboEditResponseAdapter,
    )
    if pipeline is None:
        if options.get('dummy'):
            from .synthetic import Pipeline
            pipeline = Pipeline()
        else:
            pipeline = staged_pipeline(options['model_dir'], options['manifest'], device=options.get('device', 'cpu'))
    targets = [(module, module.forward) for module in target_modules(pipeline.transformer).values()]
    restores = [native_cleanup(getattr(pipeline, name)) for name in ('transformer', 'text_encoder', 'vae')
                if getattr(pipeline, name, None) is not None]
    closed = False
    def close():
        nonlocal closed
        if not closed:
            for module, forward in targets:
                module.forward = forward
            for restore in restores:
                restore()
            closed = True
    try:
        slider = FiboParticleSlider(pipeline.transformer, seed=options.get('seed', 7))
        return FiboEditResponseAdapter(pipeline, slider, model_identity=options['manifest']), close
    except BaseException:
        close()
        raise


@contextmanager
def observe(adapter):
    """Restore native buffers, modes and pipeline sampler/fixture bookkeeping.

    Frozen multi-billion-parameter backbones are never cloned. Registered buffer
    references/values and shallow runtime attributes are the supported boundary;
    arbitrary replacement of native submodules is not supported.
    """
    import torch
    pipeline = adapter.pipeline
    roots = [getattr(pipeline, name) for name in ('transformer', 'text_encoder', 'vae')
             if getattr(pipeline, name, None) is not None]
    restores = [native_cleanup(root) for root in roots]
    modules = {id(module): module for root in roots for module in root.modules()}
    buffers = [(module, dict(module._buffers), set(module._non_persistent_buffers_set))
               for module in modules.values()]
    values = {id(value): (value, value.detach().clone()) for _, registered, _ in buffers
              for value in registered.values() if value is not None}
    attributes = dict(vars(pipeline))
    containers = [(value, value.copy()) for value in attributes.values() if type(value) in (list, dict)]
    # The CPU protocol fixture records calls on its transformer as well.
    seen = getattr(pipeline.transformer, 'seen', None)
    seen_values = list(seen) if isinstance(seen, list) else None
    try:
        for root in roots:
            root.eval()
        yield
    finally:
        with torch.no_grad():
            for original, value in values.values():
                if (original.shape, original.dtype, original.device) == (value.shape, value.dtype, value.device):
                    original.copy_(value)
                else:
                    original.data = value.clone()
        for module, registered, nonpersistent in buffers:
            module._buffers.clear()
            module._buffers.update(registered)
            module._non_persistent_buffers_set.clear()
            module._non_persistent_buffers_set.update(nonpersistent)
        vars(pipeline).clear()
        vars(pipeline).update(attributes)
        for original, value in containers:
            if isinstance(original, list):
                original[:] = value
            else:
                original.clear()
                original.update(value)
        if seen_values is not None:
            pipeline.transformer.seen = seen
            seen[:] = seen_values
        for restore in restores:
            restore()


def preflight_native(request):
    """Validate actual local native arguments before loading or sampling."""
    if request['operation'] != 'render':
        return
    import json
    from PIL import Image
    native = request['native']
    if set(native) - {'prompt', 'images', 'mask', 'height', 'width', 'output_type'}:
        raise ValueError('unsupported Fibo native argument; sampler remains four-step guidance1')
    prompt = native.get('prompt')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('nonempty native edit instruction required')
    if prompt.lstrip().startswith('{'):
        parsed = json.loads(prompt)
        if not isinstance(parsed.get('edit_instruction'), str) or not parsed['edit_instruction'].strip():
            raise ValueError('VGL edit JSON requires edit_instruction')
    images = native.get('images')
    if not isinstance(images, (list, tuple)) or not 1 <= len(images) <= 4:
        raise ValueError('Fibo EDIT requires 1..4 ordered local reference images')
    def size(value):
        if isinstance(value, Image.Image):
            return value.size
        path = _local_path(value, 'native image')
        with Image.open(path) as image:
            return image.size
    sizes = [size(image) for image in images]
    if native.get('mask') is not None and (len(images) != 1 or size(native['mask']) != sizes[0]):
        raise ValueError('Fibo mask requires one same-sized reference image')
    for key in ('height', 'width'):
        if native.get(key) is not None and (type(native[key]) is not int or native[key] <= 0):
            raise ValueError('Fibo native dimensions must be positive integers')


def renderer(adapter):
    def render(*, prompt, strength, seed, images, **native):
        preflight_native(dict(operation='render', native=dict(native, prompt=prompt, images=images)))
        # CLI preflight accepts paths; the sampler always receives image objects.
        from PIL import Image
        def load(value):
            if isinstance(value, Image.Image):
                return value
            with Image.open(value) as image:
                return image.copy()
        if native.get('mask') is not None:
            native['mask'] = load(native['mask'])
        return adapter.render_edit(prompt, [load(image) for image in images], scale=strength, seed=seed, **native)
    return render


def validate(options, spec):
    if set(spec.configuration) != {'dummy'} or type(spec.configuration['dummy']) is not bool:
        raise ValueError('unsupported Fibo native-only configuration')
    allowed = {'model_dir', 'manifest', 'device', 'base_precision'}
    if set(options) - allowed:
        raise ValueError('native Fibo inference accepts no training/cache options')
    fixture = spec.configuration['dummy']
    manifest = options.get('manifest', spec.identity['model'] if fixture else None)
    if manifest != spec.identity['model']:
        raise ValueError('native Fibo staging receipt differs from the exported base')
    from particle_sliders.fibo_native import _identity
    _identity(manifest)
    if fixture:
        if manifest.get('qualification') != 'CPU fixture only' or options.get('device', 'cpu') != 'cpu':
            raise ValueError('Fibo mock inference requires CPU fixture identity')
    else:
        if manifest.get('qualification') == 'CPU fixture only':
            raise ValueError('native Fibo requires the original verified staging receipt')
        _local_path(options.get('model_dir'), 'model_dir', directory=True)


def preflight(options, request, spec):
    preflight_native(request)


def prepare(options, spec):
    from particle_sliders.fibo_native import FORMAT
    runtime = dict(options, dummy=spec.configuration['dummy'], manifest=options.get('manifest', spec.identity['model']))
    adapter, close = attach(runtime)
    try:
        identity = native_identity(adapter, model_identity=adapter.model_identity, protocol=FORMAT)
        return InferenceBinding(adapter.network, identity, close, render=renderer(adapter),
                                observation=lambda: observe(adapter))
    except BaseException:
        close()
        raise


INFERENCE = InferenceBackend('fibo', capabilities({}), prepare, validate, response=False,
                             configure_capabilities=capabilities, preflight=preflight)
