# Fibo Edit Turbo common lifecycle

Proposed local shared pin: `fd89e1c780ab5210cd1be98ec3b70977e6349379` / core 0.7.0.
Unpublished; exact-head review is required before draft publication.

This derivative registers backend `fibo` for `particle_sliders.lifecycle`.
The core owns preparation dispatch, updates, EMA, checkpoints, export,
evaluation and cleanup. The product factory owns native loading/conditioning,
geometry and sampler differences. This is a new source cohort; keep original
pinned runs and receipts with their original loaders.

Run the complete local CPU mock example from any directory after installing
the reviewed product with its `mock` extra (Pillow; no native Diffusers import):

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m fibo_particle_slider.example new-fibo-run
```

It constructs the staged mock internally, trains, resumes, exports EMA and
reloads a response. No manual slider/adapter/provider/critic assembly is needed.
For Python, `fibo_particle_slider.lifecycle.synthetic_config(cache_dir)` returns
a configuration accepted by the same `open_session` API.

Native local options: `model_dir`, verified `manifest`, `cache_dir`,
`index_sha256`, positive `response_scale`, `steps`. To capture a new cache,
replace `index_sha256` with `pairs` containing `neutral`, `positive`, local
`images` paths, optional `mask` and `seed`. Fibo stays EDIT-only, uses native
four-step guidance 1 rendering and requires ordered reference images.
Strengths are [0,1]. Turbo training remains experimental.

The factory checks local paths, capture pair structure, reference counts, mask
arity, capture seeds and existing index checksums before model construction.
An existing cache checksum and new capture pairs are mutually exclusive. New
captures require a new cache directory. Native sessions reject CPU-fixture
receipts; no staging receipt is synthesized for pretrained weights.
The native callback retains mask-size validation and native preprocessing.

For explicitly supplied local test pipelines, close and setup failure restore
original projection forwards, checkpoint hooks, component training modes and
parameter requires-grad flags. The same mock pipeline is used when supplied;
it is never silently replaced with a newly allocated fixture.

Write `config.as_dict()` as JSON to use `particle-slider capabilities`,
`particle-slider train`, `particle-slider evaluate` and `particle-slider render`.
Run `particle-slider --help` for the common arguments. Use a fresh output
directory for each CLI continuation to preserve old metrics and receipts.
Training packets and inference artifacts have different versioned schemas;
exports cannot resume training. New source pins require a new cohort or a
reviewed explicit migration.

The core sdist includes IMAGE-LIFECYCLE.md with the public factory, capability,
packet and error contracts. Backend dependencies stay isolated: Qwen and Fibo
use different Diffusers Git revisions and are not a joint environment promise.
Use CPU thread limits and offline mode for these fixtures. They do not qualify
pretrained behavior, GPU memory, convergence or image quality.

## Fresh integration audit

The selected base preserves draft PR1 `10ee426c0c35c9f11e365b8af4b80c679f233bde`
and the reference draft `c68ab2189d1be389ad637d9096a706c35739413a`.
The observed main `09758f187b869c14027014e85752c9e76dd2950a` contains only the
initial repository payload. It is inventory evidence, not migration approval.

The public low-level re-exports and their safetensors/game loaders remain
available. The registered factory owns staged loading, native teacher capture,
cache/provider construction, the actual 128x4 routed bank, spatial FieldCritic,
released PG game, response evaluation, native sampling and cleanup. Session
train/resume/export and the installed CLI delegate to shared core. No GAN
mathematics or network definitions were copied into this product.

The original `validation.json` stays bound to its original cohort. Historical
milestone wheel evidence applies to `c68ab21`, not these edits. This source pass
adds preflight and supplied-pipeline cleanup regression cases, but their
execution, installed-wheel conformance and dependency guard are pending a
parent-scheduled CPU slot. Syntax/JSON inspection and diff checks are the only
fresh checks. Do not relabel original checkpoints with this factory identity.

Compact inference construction and matched-weight HyperGAN/HNDL definitions
remain shared-owner work. HyperGAN owns run presentation through the common
on_step, manifest and artifact seams. Native EDIT sampling remains the product
callback; this package adds no run dashboard or alternate sampler.
