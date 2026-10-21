# Fibo Edit Turbo common lifecycle

Proposed local shared pin: `f6807a91546e43e198645a3f8925d2dc7c9e03b4` / core 0.8.2.
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
milestone wheel evidence applies to `c68ab21`, not these edits. The subsequent
parent-scheduled CPU slot tested source commit
`458b25764359c6c5daeaa6c9f04df47ef43dbdb0`: all nine lifecycle cases passed
(seven new preflight/cleanup cases and two existing lifecycle cases), with no
skips, in 2.46 seconds of pytest time. End-to-end execution took 4.12 seconds;
sampled process peak was 364,474,368 bytes and host available memory stayed
above 50,385,260,544 bytes. Execution used one thread, offline mocks, CPU-only
torch and verified released PG sources. CUDA remained uninitialized and native
model imports were blocked. No build, dependency install or shared-core edit
was performed. Installed-wheel conformance and the dependency guard still
require a separate slot. Do not relabel original checkpoints with this factory
identity.

Executed evidence confirms that product-owned cleanup is sufficient with this
pinned core: supplied pipeline modes, parameter flags, forwards and checkpoint
attributes restore on both close and injected game-setup failure. The narrow
suite does not establish that central hooks are required. Atomic cache staging
and a native-request preflight hook remain source-review suggestions, not
failures demonstrated by these tests.

Matched-weight HyperGAN/HNDL definitions remain shared-owner work. HyperGAN owns run presentation through the common
on_step, manifest and artifact seams. Native EDIT sampling remains the product
callback; this package adds no run dashboard or alternate sampler.

## Core 0.8.1 local adoption proposal

Fibo retains PG 0.9.1 `bispf_optimized`; there is no PG 0.9.3 profile downgrade
or formulation change. The proposal consumes the owner's canonical source map
and artifact hashes in both lock copies. Native backend and provenance bytes
are unchanged. Prior core 0.7.0 commits, receipts and loaders remain separate;
the default new fixture cohort is `fibo-core081-synthetic-v1`. Existing packets
are never rebound to the new source identity.

The training factory now supplies an `InferenceSpec`. Export native-only v2
explicitly; `session.export()` retains the strict v1 compatibility path:

```python
from particle_sliders.inference import open_inference

# Within the newly prepared core 0.8.1 training session:
session.export_inference('new-run/native.pt', selection='ema')

# Afterwards: no training configuration, cache, captured rows, critic or game.
native = dict(prompt='keep jacket', images=[local_reference_image])
with open_inference('new-run/native.pt', options={
    'model_dir': staged_snapshot, 'manifest': original_verified_receipt,
    'device': admitted_device, 'base_precision': 'bfloat16',
}, references=True, native_request=native) as inference:
    output = inference.render(**native, strength=.5, seed=7)
```

For a mock v2 export, runtime options are empty and output is latent only.
Native-only Fibo is render-only: it accepts independent new edit instructions
and ordered references, but exposes no fabricated cached-response operation.
The common `particle-slider infer` CLI can consume the v2 artifact using
`--runtime` and `--native`; training CLI output is still the v1 artifact, so
create v2 with the explicit Python export above.

Both factories use shared `native_cleanup`, product-owned native observation
contexts and native-request preflight. Preflight checks prompt/VGL, local
images, reference count, mask dimensions and native option names before model
construction. Sampling repeats validation before invoking the native callback.
An optional staged `before_update` callable binds the common admission seam;
it is an injected pure policy, not permission to schedule resources. Observation
restores native buffer registrations/values, modes, checkpoint flags and shallow
pipeline runtime state without cloning frozen backbone parameters. Arbitrary
native submodule surgery is outside this boundary.

This adoption has source/syntax/hash inspection only. New native-only,
preflight, admission and observation regression cases are authored but unrun,
pending the next explicit CPU slot. The nine-case pass above belongs only to
the preserved core 0.7.0 proposal. No build or install has occurred for 0.8.1.

Remaining common seam: `new_directory` creates its destination on entry while
the immutable Fibo `write_cache` requires that destination not exist. It cannot
wrap the cache writer directly. Failed-cache rollback therefore remains a
source-review follow-up, not a demonstrated regression or a local core patch.

## Core 0.8.2 CLI qualification candidate

The preserved core 0.8.1 proposal at
`2911f71f8d3451c20c060d409a189f083548226a` passed all 17 installed lifecycle,
native-only inference and distribution tests in the next scheduled slot:
zero skips/failures, 4.67 seconds pytest time. It remains on local branch
`review/fibo-core081-qualified`. The tests imported an extracted product wheel
and the canonical core 0.8.1 wheel, with no checkout imports.

To close the CLI export seam, this subsequent proposal uses the explicitly
cleared core 0.8.2 commit
`f6807a91546e43e198645a3f8925d2dc7c9e03b4`, consuming its canonical 38-file
source/data map and official PG 0.9.1 profile. Native backend/provenance and
GAN implementation bytes remain unchanged. New fixture configurations use
cohort `fibo-core082-synthetic-v1`; earlier packets are not rebound.

The core 0.8.2 CLI provides both packet kinds directly:

```bash
particle-slider train --config config.json --out new-run --selection ema --export-format both
particle-slider infer --artifact new-run/adapter-v2.pt --native native.json --out latent.pt
```

The default training export remains v1. `both` writes `adapter.pt` and
`adapter-v2.pt` plus their hashes in the run manifest; `v2` selects only the
native-only inference artifact. Native-only inference needs local staged base
options for pretrained execution and never requires the training cache.
New CLI coverage includes a split resume, both export receipts, removal of the
mock training cache, and inference in a fresh process where game, critic,
optimizer and cache constructors are forbidden. Execution results for this
new proposal belong to its exact source/artifact receipt, not the earlier pass.

All evidence remains CPU mocks only. The official Turbo target is EDIT-only,
not intended for fine-tuning, gated under CC BY-NC 4.0, and still experimental
for training. No model access terms are accepted, no weights are fetched and
no pretrained gradients, memory fit, decode parity or edit quality are claimed.
