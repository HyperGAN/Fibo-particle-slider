# Validation and limits

The pinned shared backend at `f797be1ea891f6aeeeb56c02402667bcf2a640a4`
passed 190 CPU regressions and a final 23-test Fibo/package check. These counts
overlap; 20 are Fibo contracts. They include actual released PG 0.9.1 updates,
DualNorm/BISPF/SBNP ownership, matched native mock edit responses, the real
routed bank, exact split resume, lazy cache checksums, and adapter/native mock
render reload parity. Fibo's pretrained pipeline and weights were mocked.

The standalone suite builds an extracted sdist into a wheel, checks required
config/provenance/docs, rejects bundled weights, and imports the installed
wrapper with diffusers, transformers, PEFT, Hub and repository imports blocked.
It verifies the exact shared backend hash and released PG sources and asserts
CUDA is uninitialized. It also checks public resources exclude local receipts
and private workspace paths. Results are recorded in `validation.json`.

Run the standalone tests from this repository after installing the pinned
shared package and existing CPU test dependencies:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
```

To repeat the shared native contracts, check out the exact shared commit and
run its `packages/particle-sliders-core/tests/test_fibo_native.py`. Use the
verified released PG 0.9.1 sources; the core rejects modified upstream sources.
No model, GPU, network fetch or prompt-service credential is needed by those
tests. These runs reused existing CPU dependencies, and do not claim a clean
new dependency resolution or parity with the official torch 2.11 GPU stack.

Pending: gated model access, native runtime/config loading, pretrained forward
and gradient checks, memory fit, GPU checkpoint/reload/resume and sampling parity,
held-out edit/preservation quality, and stability. GPU scheduling remains with
the parent/MOSS owner. The official Turbo card discourages fine-tuning; this
requested target remains experimental. No terms were accepted and no weights,
GPU jobs, merges or deployments are part of source publication.
