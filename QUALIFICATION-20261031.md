# Installed CPU qualification — 2026-10-31

New runtime: Python3.12.15, Torch2.14.1+cpu, product0.2.0,
particle-sliders-core0.12.0 at `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`,
ParticleGAN0.9.4, HNDL0.10.0, clean HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, Pillow13.0.0,
safetensors0.8.0, setuptools84.2.1. The product wheel was installed into an
isolated task-local virtual environment. Tests ran from copies outside the
checkout. The environment reuses read-only CPU dependency libraries; this is
not a pristine dependency resolver or native runtime installation result.

| Executed check | Result |
| --- | --- |
| Installed product lifecycle, inference, CLI and modern HNDL cases | 22 passed, zero skipped |
| Extracted sdist → wheel packaging, installed offline import and provenance | 3 passed, zero skipped |
| Original S082 product wheel, separate core0.8.2/PG0.9.1 environment | 17 passed, zero skipped |
| Actual one-update S082 training checkpoint restore into modern product | Refused; modern checkpoint hash unchanged |
| Original S082 lock comparison against public source | Byte-for-byte identical |
| Selected mandatory distribution constraints | Passed; native extra excluded |
| Diff whitespace check | Passed |

The modern tests assert real HNDL projection and spatial critic classes,
particle-bank and critic updates through actual PG0.9.4, one owned particle
bank, nonzero finite bank/input gradients, frozen host parameters unchanged
with no gradients, exact projection identity at strength zero after training,
full/split resume equality, EMA, native-only reload in a fresh process without
training constructors, and profile refusal without live-state mutation.
Tampered HyperGAN builder closure and text-to-image requests are refused
before allocation. Exported inference explicitly binds the network profile.

These tests use a tiny synthetic EDIT host. They do not test pretrained
Fibo/Diffusers, GPU or native BF16 execution, decoder parity, memory fit,
convergence, held-out image quality or text-to-image. No models were loaded,
no weights downloaded, no GPU work run, and CUDA remained uninitialized.
The native `[native]` dependency set was not installed or resolved.

No repository workflow/automatic CI existed at the inspected public base.
The draft must remain blocked until its exact shared-core commit is publicly
fetchable and coordinated review is complete. Original `validation.json` and
native-guide receipts keep their earlier S082 scope and counts.
