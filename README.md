For the new common session/CLI interface, see [LIFECYCLE.md](LIFECYCLE.md).

# Fibo-particle-slider

Experimental standalone derivative for
[`briaai/Fibo-2.0-edit-turbo`](https://huggingface.co/briaai/Fibo-2.0-edit-turbo).
The public source repository is `HyperGAN/Fibo-particle-slider`; implementation
is proposed through a draft PR. The shared implementation remains isolated in
[particle-sliders draft PR 142](https://github.com/HyperGAN/particle-sliders/pull/142).

The implementation lives in the pinned shared package. This wrapper re-exports
the same classes, with no copied trainer, losses, regularizers or optimizer.
It retains native EDIT image conditioning and four-step guidance 1 sampling,
the actual routed particle bank, public PG 0.9.1 BISPF/DualNorm/SBNP, and strict
FP32 adapter export/reload. See [the native guide](NATIVE-GUIDE.md) and the
source receipt in `backend.lock.json`.

The [official model card](https://huggingface.co/briaai/Fibo-2.0-edit-turbo)
says Turbo is **not intended for fine-tuning** and recommends Dev. Turbo is
retained as specifically requested, with experimental training status and no
pretrained quality claim. No text-to-image substitute is supported.

The model is gated under **CC BY-NC 4.0**; commercial use requires a separate
Bria agreement. Its access form includes explicit Terms & Conditions and
Privacy Policy consent. No agreements were accepted. This wrapper's MIT code
license does not relicense the base model or derived weights.

After model access and a resource slot are explicitly resolved, install the
reviewed shared pin and native runtime in an isolated environment. The exact
official demo runtime and source hashes are in the guide. The backend's
`staged_pipeline` loads only a checksum-verified local snapshot with explicit
native component classes and safetensors. It never downloads a model, accepts
terms, calls a prompt service or creates credentials.

```python
from fibo_particle_slider import (
    FiboParticleSlider, FiboEditResponseAdapter, FiboCachedRows,
    build_fibo_game, staged_pipeline, write_cache,
)
```

`configs/jacket-edit.json` is a local preparation template with placeholders.
It does not launch training or imply GPU admission. No images, model weights
or adapter weights are bundled. CPU fixtures and actual released-PG tests are
maintained in the shared package; GPU memory, real native gradients,
checkpoint/reload sampling and held-out edit/preservation quality remain
unqualified. MOSS owns current GPU scheduling.

## Install and CPU checks

The proposed common lifecycle is pinned to commit
`f6807a91546e43e198645a3f8925d2dc7c9e03b4` (core 0.8.2), which is local and
unpublished. It requires parent review and a published, approved source pin
before the Git dependency can be installed from a public checkout.
Both lock copies consume the common owner's canonical 38-file source hash map.
This PG 0.9.1 proposal preserves the native backend/provenance bytes and adds
native-only inference plus preflight/admission/observation hooks. The prior
core 0.7.0 proposal remains a separate source cohort; its test results do not
qualify the new proposal until the next scheduled CPU slot.
The original PR1 receipts in `validation.json` describe the earlier
`f797be1ea891f6aeeeb56c02402667bcf2a640a4` cohort; they are retained unchanged.
The default installation does not include a pretrained model or diffusers.
Install in an isolated environment with your reviewed CPU PyTorch build:

```bash
python -m pip install '.[test]'
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
```

These commands are for a separately admitted verification slot. No suites,
dependency installation, model loading or builds were run during the fresh
source audit; current verification status is in [LIFECYCLE.md](LIFECYCLE.md).

The tests build from an extracted source distribution, verify wheel contents
and source identity, and import the installed wrapper with native model
packages blocked. Shared native contract tests and released-PG CPU updates
are recorded in [VALIDATION.md](VALIDATION.md). This is not a clean dependency
resolution or pretrained/GPU qualification claim.

The optional `[native]` extra records the inspected official demo runtime.
Installing it or acquiring model access does not grant GPU scheduling
approval. Nothing in this package accepts terms, creates credentials,
downloads weights, schedules a job, merges a PR or deploys a service.
