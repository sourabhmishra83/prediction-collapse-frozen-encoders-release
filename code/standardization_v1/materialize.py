"""Copy only hash-matched frozen feature caches to ephemeral local storage."""
from pathlib import Path
import argparse, shutil
from study_common import digest
from standardize import read_plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--dest', required=True)
    a = ap.parse_args()
    plan, _ = read_plan(a.suite)
    root, dest = Path(a.root), Path(a.dest)
    for c in plan['caches']:
        source = root / c['drive_relative']
        target = dest / 'cache' / c['encoder'] / c['source']
        target.mkdir(parents=True, exist_ok=True)
        for name, expected in [('vectors.pt', c['sha256']), ('metadata.json', c['metadata_sha256'])]:
            output = target / name
            if output.exists():
                assert digest(output) == expected, 'Different local cache: ' + str(output)
                continue
            original = source / name
            assert original.is_file(), 'Frozen cache missing from Drive: ' + str(original)
            assert digest(original) == expected, 'Frozen Drive cache hash mismatch: ' + str(original)
            temporary = target / (name + '.copying')
            shutil.copyfile(original, temporary)
            assert digest(temporary) == expected
            temporary.replace(output)
        print('Feature cache ready:', c['encoder'], c['source'], flush=True)
    print('No source text, encoder inference or model training was used.')


if __name__ == '__main__':
    main()
