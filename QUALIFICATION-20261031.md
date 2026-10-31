# Installed CPU qualification — 2026-10-31

Earlier fixture runtime: Python3.12.15, Torch2.14.1+cpu, product0.2.0,
particle-sliders-core0.12.0 at `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`,
ParticleGAN0.9.4, HNDL0.10.0, clean HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, Pillow13.0.0,
safetensors0.8.0, setuptools84.2.1. The product wheel was installed into an
isolated task-local virtual environment. Tests ran from copies outside the
checkout. The environment reuses read-only CPU dependency libraries; this is
not a pristine dependency resolver or native runtime installation result.

| Executed check | Result |
| --- | --- |
| Installed lifecycle, inference, CLI, modern HNDL and review regressions | 35 passed, zero skipped on the earlier Torch2.14.1+cpu fixture lane |
| Extracted sdist → wheel packaging, installed offline import and provenance | 3 passed, zero skipped |
| Original S082 product wheel, separate core0.8.2/PG0.9.1 environment | 17 passed, zero skipped |
| Actual one-update S082 training checkpoint restore into modern product (out-of-repository receipt) | Refused; modern checkpoint hash unchanged |
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
The exact shared-core commit is now publicly reachable and independently
verified; coordinated review remains required. Original `validation.json` and
native-guide receipts keep their earlier S082 scope and counts.

## Review findings and response

| Finding | Verified response and regression evidence |
| --- | --- |
| Stale current Shared URL pointed at PR142 | Package metadata now links the shared repository. Historical native-guide PR142 links retain their explicitly historical scope. Distribution metadata assertion passes. |
| Native extra uses Torch2.11 while CPU evidence uses2.14 | These are separate environments, not equivalent qualification. Retained the inspected Torch2.11.0/torchvision0.26.0 native pair and documented both lanes. Metadata checks confirm the native-only marker and that core accepts both versions; no native resolver or pretrained test is claimed. |
| Legacy network could use a modern reserved cohort name | Both synthetic helper and manual/JSON backend opening now reject cross-profile reserved names before allocation. Custom names and default profile/cohort selections remain unchanged. Four cross-profile/history cases and a custom-name case pass. |
| Nested PIL output could pass latent-only fixture preflight | Training and native-only preflight now check the selected runtime's output set. A latent top-level request with nested PIL output is rejected before allocation. PIL reference inputs with latent outputs remain accepted in both lifecycle paths. |
| Hash allowlist missed added core modules | Admission now compares the complete core Python/extension/data file map to the locked map, excluding generated `__pycache__`. Added `.py`, sourceless `.pyc`, and extension `.so` files plus missing/changed source are rejected before allocation. |

The follow-up installed run passed34 cases in14.32s; the updated distribution
run passed3 cases in2.54s. The original17-case S082 pass is historical evidence
for its unchanged source/environment, not a new execution of that lane.
Core deserialization and cross-repository pin/publication atomicity remain
central-owner review items. No shared-core source or historical lock was edited.

The genuine S082 checkpoint result is an **out-of-repository execution receipt**,
not an in-repository reusable binary fixture or pytest case. The original
one-step packet was produced with the original installed S082 product/core/PG
environment; its refusal and unchanged modern state were recorded separately.
The reusable in-repository tests cover network/source-profile refusal with
bounded generated fixtures and do not claim to regenerate that historical packet.

## Bounded final polish and clean installation

Renderer callbacks now receive the actual runtime output capabilities in both
training and native-only inference. A direct-callback regression forbids the
sampler and verifies that both callbacks reject PIL output for CPU fixtures,
even without the shared session wrapper's capability check.

Core commit `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` is publicly reachable on
`qualify/core012-cutpoints-20261031`; all51 locked source/data files match the
public Git tree's blob identities. The previous publication blocker is resolved.
A new virtual environment with no borrowed site-packages completed normal
pip dependency resolution/installation from that exact core VCS commit and
HyperGAN `5f368a3770cf5bdae0448995c9f76e29af8049f7`, plus PyPI dependencies.
It uses the official Torch2.6.0+cpu cp312 wheel, independently checked against
the public PyTorch index SHA256. `pip check` reports no broken requirements;
installed VCS `direct_url.json` receipts and the verified HyperGAN/HNDL builder
confirm the exact identities. No native extra or pretrained weights were fetched.

The README limits module inspection to `particle_sliders` and
`concept_slider_core`. Generated `__pycache__` entries are excluded; the check
is installation consistency evidence, not a global tamper-proof security boundary.
Writing to cache directories already requires installation access.

Clean Torch2.6.0+cpu execution: **35 installed cases passed in14.84s** and
**3 extracted-sdist/wheel/import/provenance cases passed in2.40s**, zero skipped.
The installed tests ran outside the source checkout with all product/core/HNDL/
HyperGAN/Torch modules imported from the new environment's own site-packages.
The preceding read-only-dependency Torch2.14.1+cpu lane passed the same35 cases
in14.71s. These are CPU software checks, not native model qualification.
