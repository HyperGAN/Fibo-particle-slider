# Publication and qualification status

This is a source draft for the new core 0.12/PG 0.9.4/HyperGAN-develop/HNDL 0.10
cohort. Earlier Fibo PR1 and security PR2 are merged; their receipts remain
bound to their original sources. `validation.json` and the historical guide
are unchanged evidence for those earlier cohorts, not modern qualification.

The new product explicitly selects `hypergan-hndl-v2`. It runs HNDL projections
and the HNDL spatial critic through the exact clean HyperGAN develop builder;
it is not an import-only adoption. CPU tests use a tiny synthetic EDIT host and
actual released ParticleGAN 0.9.4. They do not qualify pretrained model weights,
GPU gradients, native BF16 behavior, memory, decoder parity or edit quality.
[The installed qualification receipt](QUALIFICATION-20261031.md) records
42 product/review/migration cases and 3 distribution cases, all passing without skips.
Text-to-image is unavailable and remains refused before model construction.

The active cleared core is
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`. Core 0.12.0, official PG 0.9.4,
HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7` and HNDL 0.10.0
remain explicit. This migration changes the recursive source closure and
native-only artifact identity. The earlier 4abc preview and original S082 source
locks remain separately preserved; no old packet metadata is rewritten.
MIGRATION-A794.md describes reproducible checks and the exact source boundary.
The central core's CPU/synthetic-CUDA evidence does not qualify native Fibo,
response widths, text-to-image, audio or model quality. This product still
requires renewed actual-diff Claude RETURN review before landing.

No GitHub workflow or release job exists in the inspected repository. This
change publishes source through a draft PR only; it does not publish packages,
model or adapter weights, deploy a service, request hardware or merge a PR.
The known-good S082 product/source lock remains separately installable through
`requirements-s082.txt`. New profile checkpoints are intentionally incompatible.
