# Fibo EDIT lifecycle

This candidate uses the exact modern cohort in [README.md](README.md).
`fibo_particle_slider.lifecycle.BACKEND` registers `fibo` with shared core;
`fibo_particle_slider.inference.INFERENCE` provides native-only render reload.
Shared core owns updates, EMA, checkpoints, exports and the CLI. Product code
owns staged loading, ordered references, masks, validation and cleanup.

`synthetic_config(cache_dir)` selects `hypergan-hndl-v2` and cohort
`fibo-hypergan-hndl-v2-cpu`. It constructs the actual HNDL routed projections
and HNDL spatial critic through HyperGAN's verified builder. Shared
`build_fibo_game` executes released PG0.9.4 `bispf_optimized` on the original
spatial velocity geometry. No flat-vector reshaping, width admission bypass,
new loss or new optimizer is introduced.

```python
from particle_sliders.lifecycle import open_session
from fibo_particle_slider.lifecycle import synthetic_config

with open_session(synthetic_config('new-cache')) as session:
    session.train()
    session.checkpoint('new-run-training.pt')
    session.export_inference('new-run-native.pt', selection='ema')
```

The one-command installed fixture is:

```bash
python -m fibo_particle_slider.example new-fibo-run
```

Native local options remain `model_dir`, verified `manifest`, `cache_dir`,
`index_sha256`, positive `response_scale`, `steps`, `batch`, `seed`, `device`,
`base_precision`, `ema_decay`, and `network_profile`. To capture a new cache,
replace the existing index checksum with nonempty `pairs` of `neutral`,
`positive`, local `images`, optional `mask` and `seed`. Native loading verifies
already staged content; it does not acquire weights or synthesize receipts.
A capture requires a new directory. The native guide describes the underlying
EDIT protocol; its old PG/network receipts remain historical S082 evidence.

CLI configuration uses `synthetic_config(...).as_dict()` as JSON:

```bash
particle-slider train --config config.json --out new-run --selection ema --export-format both
particle-slider infer --artifact new-run/adapter-v2.pt --native native.json --out latent.pt
```

`both` writes the strict v1 inference artifact and native-only v2 artifact with
hashes in the run manifest. Native-only reload does not construct training
rows, critic, game or optimizer. Its export specification explicitly binds
`dummy` and `network_profile`; reload cannot silently select a different graph.
For native model rendering, supply the original local staging receipt and
model directory through `--runtime`. Rendering requires references, accepts
strengths 0–1 and uses the native four-step guidance-1 sampler.

Close and setup failures restore borrowed pipeline forwards, modes, parameter
flags and checkpoint hooks. Observation restores native buffers and shallow
sampler state without cloning frozen backbone weights. Optional `before_update`
is a pure injected admission policy. Unsupported native arguments, reference
counts, masks and text-to-image requests fail before pipeline allocation.

The current installed CPU checks cover real HNDL/PG parameter updates, frozen
host ownership, nonzero adapter/bank gradients, exact zero-strength identity,
continuous spatial domain, full versus split resume, EMA, native-only reload
and rejected profile restoration without mutation. Pretrained/GPU sampling,
training quality and memory remain unqualified. Historical S082 checkpoints
require their original separate source environment; see `requirements-s082.txt`.
