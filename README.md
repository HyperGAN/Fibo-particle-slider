# Fibo-particle-slider

Experimental routed particle slider for **Fibo 2.0 Edit Turbo**. This source
candidate connects the installed shared particle-sliders game to the real
HyperGAN develop/HNDL projection and spatial critic. It does not contain a
product-local GAN loss, optimizer, trainer or replacement sampler.

## Exact modern cohort

| Component | Required identity |
| --- | --- |
| particle-sliders-core | 0.12.0, commit `ddb41423bcefc1fe76cd94d0f9b367353b3ea137` |
| ParticleGAN | official 0.9.4, verified release source profile |
| HyperGAN | develop commit `5f368a3770cf5bdae0448995c9f76e29af8049f7`, clean build provenance |
| HNDL | 0.10.0, `hypergan-hndl-v2` |
| Native model | `briaai/Fibo-2.0-edit-turbo`, revision `177f2d4a06b6587144ef5c2b8452a4f1fe2ab4b7` |
| Native Diffusers | custom commit `60f95274bf07ed705d675eb09d99b6ae125b601e` |

The active modern candidate selects the exact shared source above, including
the merged unified training and isolated executor interface. Its
recursive 574-file source/data closure is checked against both built artifacts;
[the migration guide](MIGRATION-DDB.md) records reproducible checks and packet
identity changes. Prior `a794` and `4abc` preview packets remain bound to their original
product/core source environment. They are not relabeled or converted.
Existing coordinated Claude RETURN review remains required before landing this product pin.
The two `backend.lock.json` copies bind the exact installed core source/data
file set and hashes within `particle_sliders` and `concept_slider_core`,
including rejection of additional Python/extension modules there. Generated
`__pycache__` entries are excluded from inspection. This is an installation
consistency check, not a global tamper-proof security boundary; writing to those
cache directories already requires access to the installation. Core verifies
the released PG sources, HNDL closure, HyperGAN builder closure and clean build
provenance before the product allocates a pipeline.

Install in a fresh Python 3.11–3.14 environment with a reviewed CPU Torch build:

```bash
python -m pip install '.[test,mock]'
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python -m fibo_particle_slider.example new-fibo-cpu-run
```

This exercises a tiny synthetic EDIT host with real HNDL graphs and the real
released PG optimizer. The optional `[native]` extra is the separately inspected
model runtime; it is not needed for CPU fixtures.

| Runtime lane | Torch / torchvision | Evidence |
| --- | --- | --- |
| Current ddb software check | Torch 2.14.1+cu130, CPU execution only | 46 distinct installed/packaging/executor cases; existing read-only dependencies |
| Earlier clean supported CPU install | Official Torch 2.6.0+cpu; no torchvision execution required | Fresh dependency-resolving install and real HNDL/PG fixture tests |
| Earlier CPU fixture | Torch 2.14.1+cpu; no torchvision execution required | Real HNDL/PG tests with read-only dependency reuse |
| Optional `[native]` demo runtime | Torch 2.11.0 / torchvision 0.26.0 | Inspected upstream pair; not installed, resolved or pretrained-qualified here |

The historical CPU rows retain their original source receipts; see
[MIGRATION-DDB.md](MIGRATION-DDB.md) for this source pin's verification.
Use separate environments for these lanes: adding `[native]` selects Torch 2.11.0
and does not reproduce either CPU fixture receipt. These Torch versions satisfy
core's `>=2.6,<3` declaration; this is not evidence that pretrained execution works
in these fixture lanes. The inspected native pair is retained without an untested upgrade.
Native dependency resolution,
pretrained gradients, decoder parity, GPU memory and image quality are unrun.
No model weights are downloaded by the fixture or bundled in the package.

## EDIT and text-to-image are separate

The registered `fibo` backend requires 1–4 ordered reference images and supports
native masks with one same-sized source image. It retains the native four-step,
guidance-1 EDIT sampler and six layer-bundled Qwen3-VL conditioning tensors.
It does **not** implement text-to-image. The inspected product and shared Fibo
sources provide no text-to-image loader, conditioning adapter or lifecycle.
An import or a successful EDIT fixture cannot qualify a text-to-image model.
That path needs its own exact model/pipeline source, native response contract,
shared binding and tests; substituting empty references would break this model's
contract. The current capability gate continues to reject it before loading.

The [upstream Turbo model card](https://huggingface.co/briaai/Fibo-2.0-edit-turbo)
states that Turbo is not intended for fine-tuning and recommends Dev. The target
remains experimental. Its CC BY-NC 4.0 weights and access terms are separate from
this wrapper's MIT code. No terms were accepted and no weights published.

## Existing checkpoints

The known-good S082 product remains installable from its original exact source
in a **separate environment**:

```bash
python -m pip install -r requirements-s082.txt
```

This selects product `c45be09f847542ae17f03872dea84663f9f44bb5`, core 0.8.2
`f6807a91546e43e198645a3f8925d2dc7c9e03b4`, and PG 0.9.1. Its original lock is
preserved byte-for-byte at `cohorts/s082/backend.lock.json`. Preserve original
source/model/data receipts with those runs. The modern product uses a new
`fibo-hypergan-hndl-v2-cpu` fixture cohort. It refuses historical source/profile
checkpoints rather than relabeling or automatically converting them. The optional
`legacy` network choice in core 0.12 is a fresh comparison cohort, not an S082
checkpoint loader.

See [LIFECYCLE.md](LIFECYCLE.md) for session, CLI and native-only inference,
[NATIVE-GUIDE.md](NATIVE-GUIDE.md) for historical native EDIT protocol evidence,
and [PUBLICATION-STATUS.md](PUBLICATION-STATUS.md) for current scope and blockers.

The prior modern preview is preserved separately at product
`50a70253fd14930671ad4fc52cbca6d6b13ce6a6`; use
`cohorts/preview-core4abc/requirements.txt` in its original environment to
continue those runs. Current native-only artifacts explicitly bind the cleared
core commit and reject earlier preview specifications.
