"""Rebuild both modern locks from a clean exact source and its built artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tomllib
import zipfile

TARGET = 'df0ea1c33e9bca0180eb8137a2db4229bbb88cb8'
NAMESPACES = ('particle_sliders/', 'concept_slider_core/')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--core-source', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--sdist', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    source = args.core_source.resolve()
    def git(*arguments):
        return subprocess.check_output(['git', *arguments], cwd=source, text=True).strip()
    if git('rev-parse', 'HEAD') != TARGET or git('status', '--porcelain', '--untracked-files=normal'):
        raise ValueError('lock generation requires the exact clean cleared core checkout')
    package = source / 'packages/particle-sliders-core'
    sources = {}
    for namespace in NAMESPACES:
        for path in (package / 'src' / namespace).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                sources[path.relative_to(package / 'src').as_posix()] = digest(path.read_bytes())
    sources = dict(sorted(sources.items()))
    with zipfile.ZipFile(args.wheel) as wheel:
        delivered = {name: digest(wheel.read(name)) for name in wheel.namelist()
                     if name.startswith(NAMESPACES) and not name.endswith('/')}
    with tarfile.open(args.sdist) as sdist:
        archived = {name.split('/src/', 1)[1]: digest(sdist.extractfile(name).read())
                    for name in sdist.getnames() if '/src/' in name
                    and name.split('/src/', 1)[1].startswith(NAMESPACES)
                    and sdist.getmember(name).isfile()}
    if delivered != sources or archived != sources:
        raise ValueError('core source, wheel and sdist recursive closures differ')
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / 'backend.lock.json').read_text())
    lock.update(shared_commit=TARGET,
                shared_version=tomllib.loads((package / 'pyproject.toml').read_text())['project']['version'],
                shared_source_sha256=sources,
                shared_source_map_sha256=digest(json.dumps(sources, sort_keys=True, separators=(',', ':')).encode()),
                shared_backend_sha256=sources['particle_sliders/fibo_native.py'],
                shared_wheel_sha256=digest(args.wheel.read_bytes()),
                shared_sdist_sha256=digest(args.sdist.read_bytes()),
                previous_modern={'product_commit': 'c55f68fd005b2c6c965f50737a10e5c52b5d06c2',
                                 'shared_commit': 'ddb41423bcefc1fe76cd94d0f9b367353b3ea137',
                                 'lock': 'cohorts/pg094-ddb/backend.lock.json'},
                publication='Cleared immutable modern core; product qualification remains CPU software only')
    encoded = json.dumps(lock, indent=2) + '\n'
    for destination in (root / 'backend.lock.json', root / 'src/fibo_particle_slider/backend.lock.json'):
        if args.check:
            if destination.read_text() != encoded:
                raise ValueError('recorded modern lock differs from these source/artifact inputs')
        else:
            destination.write_text(encoded)
    print('Verified' if args.check else 'Wrote', 'matching modern locks for', TARGET, 'with', len(sources), 'source/data files')


if __name__ == '__main__':
    main()
