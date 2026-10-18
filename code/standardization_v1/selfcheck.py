"""Preparation tests only: no model fitting, backward pass or optimizer steps."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import numpy as np
from controls import best_threshold
from study_common import read, write, digest
from standardize import fit_scaler, transform, authorize, Study


def main():
    tests = []
    x = np.array([[1., 7.], [3., 7.]])
    mean, scale, zero = fit_scaler(x)
    assert np.array_equal(mean, [2., 7.]) and np.array_equal(scale, [1., 1.])
    assert zero.tolist() == [False, True]
    z = transform(x, mean, scale).numpy()
    assert np.allclose(z.mean(axis=0), 0) and np.allclose(z[:, 0].std(ddof=0), 1)
    heldout = transform(np.array([[1000., 9.]]), mean, scale).numpy()
    assert heldout.tolist() == [[998., 2.]] and mean.tolist() == [2., 7.]
    tests.append('training-only population scaler, zero variance and unchanged held-out transform')
    threshold, f1 = best_threshold(np.array([0, 0, 1, 1]), np.array([-2., -1., 1., 2.]))
    assert threshold == 0 and f1 == 1
    tests.append('source-only threshold tie rule')
    with TemporaryDirectory() as temp:
        root = Path(temp)
        write(root / 'approval_record.json', {'status': 'pending', 'approved_plan_sha256': 'a', 'approval_text': None})
        for expected_hash, execute in [('a', True), ('a', False)]:
            try:
                authorize(root, expected_hash, execute)
            except PermissionError:
                pass
            else:
                raise AssertionError('Unauthorized execution accepted')
        write(root / 'approval_record.json', {'status': 'approved', 'approved_plan_sha256': 'a', 'approval_text': 'fixture only'})
        try:
            authorize(root, 'changed-plan', True)
        except PermissionError:
            pass
        else:
            raise AssertionError('Changed plan accepted')
        tests.append('authorization rejects pending approval and modified plan')
        job = {'job_id': 'fixture'}
        folder = root / 'results/fixture'
        folder.mkdir(parents=True)
        (folder / 'payload.txt').write_text('fixture', encoding='utf-8')
        write(folder / 'result.json', {'job': job, 'plan_sha256': 'fixture-plan',
                                      'files_sha256': {'payload.txt': digest(folder / 'payload.txt')}})
        # Completed-job branch returns before any model/optimizer creation.
        stub = SimpleNamespace(out=root, h='fixture-plan')
        Study.execute(stub, job)
        (folder / 'payload.txt').write_text('tampered', encoding='utf-8')
        try:
            Study.execute(stub, job)
        except AssertionError:
            pass
        else:
            raise AssertionError('Corrupt completed job was accepted')
        tests.append('resumption skips verified completion and rejects corrupt files')
    print(json.dumps({'status': 'PASS', 'tests': tests, 'model_fits': 0, 'optimizer_steps': 0, 'encoder_inference': 0}))


if __name__ == '__main__':
    main()
