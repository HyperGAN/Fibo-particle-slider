# Publication and qualification status

This is a source draft for the new core0.12/PG0.9.4/HyperGAN-develop/HNDL0.10
cohort. Earlier Fibo PR1 and security PR2 are merged; their receipts remain
bound to their original sources. `validation.json` and the historical guide
are unchanged evidence for those earlier cohorts, not modern qualification.

The new product explicitly selects `hypergan-hndl-v2`. It runs HNDL projections
and the HNDL spatial critic through the exact clean HyperGAN develop builder;
it is not an import-only adoption. CPU tests use a tiny synthetic EDIT host and
actual released ParticleGAN0.9.4. They do not qualify pretrained model weights,
GPU gradients, native BF16 behavior, memory, decoder parity or edit quality.
[The installed qualification receipt](QUALIFICATION-20261031.md) records
35 product/review cases and 3 distribution cases, all passing without skips.
Text-to-image is unavailable and remains refused before model construction.

The core owner's exact candidate is
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`. It is now publicly reachable on
`qualify/core012-cutpoints-20261031`. All51 locked source/data files match its
public Git tree. A clean environment installed that exact core and HyperGAN
VCS commit, official CPU Torch2.6.0, PG0.9.4 and HNDL0.10 with normal dependency
resolution; `pip check` passes. The earlier source-publication blocker is
resolved. Coordinated review and native qualification remain separate.

No GitHub workflow or release job exists in the inspected repository. This
change publishes source through a draft PR only; it does not publish packages,
model or adapter weights, deploy a service, request hardware or merge a PR.
The known-good S082 product/source lock remains separately installable through
`requirements-s082.txt`. New profile checkpoints are intentionally incompatible.
