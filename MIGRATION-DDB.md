# Current Fibo shared-core migration

The active package selects core 0.12.0 at
`ddb41423bcefc1fe76cd94d0f9b367353b3ea137`, including the merged unified
training and isolated executor APIs. Official ParticleGAN remains 0.9.4
(release source `330eb94ee748cdcad131b0275d8de75fd5a7f96f`). The real routed
particle bank, HyperGAN develop builder, HNDL 0.10.0 projection/critic graphs,
native EDIT response geometry and public PG recipe remain in shared core.
Python 3.11–3.14 is required by this product's HNDL lane.

Both active locks cover the complete 574-file source/data closure in
`particle_sliders` and `concept_slider_core`, including the unified executor
and packaged legacy namespaces. The generator compares the clean exact Git
checkout against both the wheel and source archive:

```bash
git clone https://github.com/HyperGAN/particle-sliders.git core-source
git -C core-source checkout --detach ddb41423bcefc1fe76cd94d0f9b367353b3ea137
python -m build --outdir core-artifacts core-source/packages/particle-sliders-core
python tools/update_core_lock.py --core-source core-source \
  --wheel core-artifacts/particle_sliders_core-0.12.0-py3-none-any.whl \
  --sdist core-artifacts/particle_sliders_core-0.12.0.tar.gz
```

Append `--check` to verify the recorded build inputs. Archive timestamps can
change artifact digests; the source-map digest identifies source independently.

The previous a794 proposal remains installable in its separate original
environment through `cohorts/preview-corea794/requirements.txt`, pinned to
product `aab985662c5efe295dbd28523bf54b90b5e49e48`. Its lock is preserved
byte-for-byte alongside the older 4abc preview and S082 locks. See
[the historical migration receipt](MIGRATION-A794.md) for a794-specific checks.
New native-only packets bind the current exact core commit. Old packets and
checkpoints retain their source identities; this change neither rewrites them
nor retrains any previously produced weights.

The new executor works with an explicit fixture profile:

```bash
python - <<'PY'
import json
from pathlib import Path
from fibo_particle_slider.lifecycle import synthetic_config
Path('fixture.json').write_text(json.dumps(synthetic_config('new-cache', steps=2).as_dict()))
PY
particle-slider plan --config fixture.json --profile configs/training-fixture-profile.json --no-reporting
particle-slider train --config fixture.json --profile configs/training-fixture-profile.json --check --no-reporting
particle-slider train --config fixture.json --profile configs/training-fixture-profile.json \
  --out new-run --selection ema --export-format both --no-reporting
```

This profile declares `status=fixture` and requires explicit `dummy=true`;
it does not admit native model execution. Periodic checkpoints appear under
`new-run/artifacts/checkpoints/step-*`. Resume accepts a checkpoint directory
or its `training.pt`, with a fresh output directory. The existing session CLI
and native-only inference API remain available.

Verification on Python 3.11.17, Torch 2.14.1+cu130 with CPU execution only,
ParticleGAN 0.9.4, HyperGAN 2.0.0 at the pinned develop source and HNDL 0.10.0:
46 distinct installed-product cases passed with no skips across the full run
and targeted correction run. Coverage includes source/sdist/wheel closure,
offline import with optional native packages blocked, actual HNDL/PG gradients,
resume, EMA/raw export, native-only reload, cleanup retry and unified executor
plan/check/periodic checkpoints/resume. The executor rejects a fixture request
without `dummy=true` before allocation.

The isolated environment installed the product and core wheels, reusing existing
read-only dependency installations. This is not a clean dependency-resolution
receipt for ddb. An initial host `backports` namespace collision was resolved
inside the verification environment with `backports.tarfile==1.2.0`; the product
requires no such runtime dependency. The training test parser was corrected to
accept streamed progress followed by the final JSON receipt. All three initially
failing/erroring cases passed on the bounded rerun.

These checks use a tiny EDIT host with the actual HNDL/PG code.
Native model gradients, pretrained quality, BF16 behavior, GPU memory and
text-to-image remain unqualified. No weights are acquired or published.
