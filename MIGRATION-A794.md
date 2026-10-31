# Active modern Fibo source migration

Only the active modern candidate moves from core `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`
to the coordinated immutable core `a794e3b4a8813b012f2c4faa3da82b78d4a90214`.
Core version remains 0.12.0, PG remains 0.9.4, and HyperGAN/HNDL pins, model
identity, native EDIT gate, routed bank, objectives and cohort names are unchanged.
The new core's software qualification does not establish native model/GPU,
response-width, text-to-image or image-quality admission for this product.

Both locks are generated from the exact clean core source and checked against
its built wheel and source archive, recursively within `particle_sliders` and
`concept_slider_core`. Build the core artifacts in an isolated environment:

```bash
git clone --filter=blob:none https://github.com/HyperGAN/particle-sliders.git core-source
git -C core-source checkout --detach a794e3b4a8813b012f2c4faa3da82b78d4a90214
python -m build --outdir core-artifacts core-source/packages/particle-sliders-core
python tools/update_core_lock.py --core-source core-source \
  --wheel core-artifacts/particle_sliders_core-0.12.0-py3-none-any.whl \
  --sdist core-artifacts/particle_sliders_core-0.12.0.tar.gz --check
```

`--check` verifies the recorded artifact inputs; rebuilding archives can change
archive bytes despite identical source. A deliberate regeneration uses the same
command without `--check`. Source-map identity is independent of archive timestamps.
For the tested Python 3.12/Linux x86_64 CPU lane, install from this product source
in a fresh environment with the exact official CPU archive (other platforms need
their own reviewed CPU artifact):

```bash
python3.12 -m venv cpu-env
cpu-env/bin/python -m pip install \
  'https://download-r2.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp312-cp312-linux_x86_64.whl#sha256=59e78aa0c690f70734e42670036d6b541930b8eabbaa18d94e090abf14cc4d91' \
  '.[test,mock]'
cpu-env/bin/python -m pip check
```

The product's Git dependency selects the exact new core and existing HyperGAN commit.
The resolver receipt records selected transitive versions; this is not a promise
that unconstrained future dependency resolution reproduces every transitive version.
No native extra, model download or GPU execution is needed for CPU fixtures.

The earlier modern preview remains separate at product
`50a70253fd14930671ad4fc52cbca6d6b13ce6a6`. Its original lock is retained
byte-for-byte in `cohorts/preview-core4abc/backend.lock.json`; its separate
requirements select that original source environment. S082 files stay unchanged.
Do not edit prior packet metadata or transplant preview tensors into new packets.

New native-only specifications explicitly contain `core_commit`; old preview
specifications lacking the cleared commit are refused. Existing training manifests
also bind their original core/factory source identities. Current human-readable
cohort labels remain unchanged, while source identity distinguishes the runs.
There is no automatic checkpoint conversion or relabeling.

Executed product results for this source migration: 42 installed CPU cases and
3 distribution/provenance checks passed, with no skips. The checks include
EMA-disabled/raw behavior, raw/EMA reload, exact OFF rendering, resume and
failure/cleanup retry. Genuine old preview/S082 packet refusal receipts remain
separate out-of-repository evidence; the archived packets were not modified.

The current historical training check uses decoded packets with `session.restore`;
older receipts that passed a path directly to that method are superseded. Native
inference uses the supported artifact-path API. No prior packet is rewritten.

The final installed and extracted-source checks also passed while both canonical
product and core checkouts were temporarily absent: 42 installed tests and
3 distribution checks, with no skips. Both checkouts were restored afterward.
