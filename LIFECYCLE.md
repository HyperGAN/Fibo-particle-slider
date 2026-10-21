# Fibo Edit Turbo common lifecycle

Proposed local shared pin: `cb0e6c39a51d451fa6ed5e72169b66e12ac247d1` / core 0.7.0.
Unpublished; exact-head review is required before draft publication.

This derivative registers backend `fibo` for `particle_sliders.lifecycle`.
The core owns preparation dispatch, updates, EMA, checkpoints, export,
evaluation and cleanup. The product factory owns native loading/conditioning,
geometry and sampler differences. This is a new source cohort; keep original
pinned runs and receipts with their original loaders.

Run the complete local CPU mock example from any directory after installation:

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
