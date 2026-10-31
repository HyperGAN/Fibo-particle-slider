> Historical review receipts below retain their original product/core scope.
> Current active migration and fresh execution evidence are recorded in
> MIGRATION-A794.md and the migration section appended below.

# Installed CPU qualification — 2026-10-31

Earlier fixture runtime: Python 3.12.15, Torch 2.14.1+cpu, product 0.2.0,
particle-sliders-core 0.12.0 at `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`,
ParticleGAN 0.9.4, HNDL 0.10.0, clean HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, Pillow 13.0.0,
safetensors 0.8.0, setuptools 84.2.1. The product wheel was installed into an
isolated task-local virtual environment. Tests ran from copies outside the
checkout. The environment reuses read-only CPU dependency libraries; this is
not a pristine dependency resolver or native runtime installation result.

| Executed check | Result |
| --- | --- |
| Installed lifecycle, inference, CLI, modern HNDL and review regressions | 35 passed, zero skipped on the earlier Torch 2.14.1+cpu fixture lane |
| Extracted sdist → wheel packaging, installed offline import and provenance | 3 passed, zero skipped |
| Original S082 product wheel, separate core 0.8.2/PG 0.9.1 environment | 17 passed, zero skipped |
| Actual one-update S082 training checkpoint restore into modern product (out-of-repository receipt) | Refused; modern checkpoint hash unchanged |
| Original S082 lock comparison against public source | Byte-for-byte identical |
| Selected mandatory distribution constraints | Passed; native extra excluded |
| Diff whitespace check | Passed |

The modern tests assert real HNDL projection and spatial critic classes,
particle-bank and critic updates through actual PG 0.9.4, one owned particle
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
| Native extra uses Torch 2.11 while CPU evidence uses 2.14 | These are separate environments, not equivalent qualification. Retained the inspected Torch 2.11.0/torchvision 0.26.0 native pair and documented both lanes. Metadata checks confirm the native-only marker and that core accepts both versions; no native resolver or pretrained test is claimed. |
| Legacy network could use a modern reserved cohort name | Both synthetic helper and manual/JSON backend opening now reject cross-profile reserved names before allocation. Custom names and default profile/cohort selections remain unchanged. Four cross-profile/history cases and a custom-name case pass. |
| Nested PIL output could pass latent-only fixture preflight | Training and native-only preflight now check the selected runtime's output set. A latent top-level request with nested PIL output is rejected before allocation. PIL reference inputs with latent outputs remain accepted in both lifecycle paths. |
| Hash allowlist missed added core modules | Admission now compares the complete core Python/extension/data file map to the locked map, excluding generated `__pycache__`. Added `.py`, sourceless `.pyc`, and extension `.so` files plus missing/changed source are rejected before allocation. |

The follow-up installed run passed 34 cases in 14.32 seconds; the updated distribution
run passed 3 cases in 2.54 seconds. The original 17-case S082 pass is historical evidence
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
`qualify/core012-cutpoints-20261031`; all 51 locked source/data files match the
public Git tree's blob identities. The previous publication blocker is resolved.
A new virtual environment with no borrowed site-packages completed normal
pip dependency resolution/installation from that exact core VCS commit and
HyperGAN `5f368a3770cf5bdae0448995c9f76e29af8049f7`, plus PyPI dependencies.
It uses the official Torch 2.6.0+cpu cp312 wheel, independently checked against
the public PyTorch index SHA256. `pip check` reports no broken requirements;
installed VCS `direct_url.json` receipts and the verified HyperGAN/HNDL builder
confirm the exact identities. No native extra or pretrained weights were fetched.

The README limits module inspection to `particle_sliders` and
`concept_slider_core`. Generated `__pycache__` entries are excluded; the check
is installation consistency evidence, not a global tamper-proof security boundary.
Writing to cache directories already requires installation access.

Clean Torch 2.6.0+cpu execution: **35 installed cases passed in 14.84 seconds** and
**3 extracted-sdist/wheel/import/provenance cases passed in 2.40 seconds**, zero skipped.
The installed tests ran outside the source checkout with all product/core/HNDL/
HyperGAN/Torch modules imported from the new environment's own site-packages.
The preceding read-only-dependency Torch 2.14.1+cpu lane passed the same 35 cases
in 14.71 seconds. These are CPU software checks, not native model qualification.

## Bounded renderer compatibility audit

A read-only search covered 40 available inventory/source-cohort and recorded
original checkouts, with 17,055 tracked Python/Markdown/TOML files. Fibo import
references, direct calls, qualified calls and simple import aliases were
examined. The only identified call sites were two historical in-package
`renderer(adapter)` calls and the two current explicit-capability calls. No
cross-repository call was identified within this scope; untracked files,
arbitrary dynamic aliases and unknown external consumers were not covered.
The exact paths, heads and matching lines are retained in the separate audit
receipt; absence within this bounded search is not a claim that no external
users exist.

The positional helper call is preserved by deriving outputs from the validated
adapter model identity. Explicit capabilities cannot broaden that set; a CPU
fixture defaults to a latent output forwarded explicitly to sampling, and PIL
output is still refused. Regressions cover the original call shape, its default,
PIL refusal, attempted capability widening and missing identity. Shared-core
source, pins and historical/default cohort identities remain unchanged.

After this compatibility patch, the clean Torch 2.6.0+cpu environment passed
**37 installed cases in 14.52 seconds** and **3 distribution cases in 2.31 seconds**, zero skipped.
No dependency pins, shared-core source or default/historical cohort names changed.

## Active coordinated migration to a794

The active modern candidate now pins core
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`; its version remains 0.12.0, with
PG 0.9.4, HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7` and
HNDL 0.10.0 unchanged. Both recursive 51-file locks were regenerated once from
the exact clean source and verified against its wheel and source archive.
The previous preview lock and S082 lock remain byte-for-byte preserved.

A fresh Python 3.12 environment resolved the real VCS dependencies and verified
the imported core file bytes, module origins, exact VCS receipts and HNDL builder.
`pip check` reports no broken requirements. Tests imported the installed product
wheel built through the source archive, outside both product/core checkouts:
**42 cases passed in 16.35 seconds**, zero skipped. Extracted-sdist/wheel/import
and provenance checks: **3 passed in 2.41 seconds**, zero skipped.

Coverage includes the current product conformance suite, exact split resume,
raw and EMA native-only reload, exact OFF behavior, the EMA-disabled contract,
recursive extra-data/source tamper rejection, explicit native core identity,
cleanup failure retry, primary update error preservation and restored host flags.
Real training and raw/EMA inference packets saved from the original 4abc preview
were refused by the new source cohort without mutation; inference refusal preceded
pipeline allocation. The original S082 training packet was also refused without
changing live state. These genuine historical packet checks are separate
out-of-repository execution receipts, with packet checksums preserved.

Prior preview artifacts must continue with their original product/core source
environment. No packet metadata is relabeled. Shared-core mathematics, native
model gates, model licenses, routed-bank architecture and cohort names were not
changed here. The central core's synthetic CUDA qualification is not a native
Fibo, width, audio, text-to-image or quality qualification. This product migration
still requires actual-diff Claude RETURN review before landing.

Historical receipt correction: earlier direct `session.restore(path)` checks
used an unsupported argument and therefore did not establish compatibility of
the decoded historical payload. Those path-argument receipts are superseded by
the current out-of-repository check, which loads the genuine S082 and 4abc
training packets with `torch.load(..., weights_only=True)` and passes the decoded
packets to `session.restore(packet)`. Both are refused for manifest/source/cohort
mismatch, with live state and original packet checksums unchanged. Native-only
raw/EMA file loading used its supported path API and remains a valid check.
