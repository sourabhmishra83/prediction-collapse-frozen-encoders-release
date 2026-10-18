"""Frozen 72-job standardization control; importing never trains."""
from pathlib import Path
import argparse, hashlib, json, platform, time, uuid
import numpy as np
import pandas as pd
import torch
import sklearn
from study_common import read, write, digest, fingerprint, seed_everything, make_model, core_lr
from controls import metrics, best_threshold


def read_plan(suite):
    suite = Path(suite)
    summary = read(suite / 'plan_summary.json')
    assert digest(suite / 'prepared/plan.json') == summary['plan_sha256']
    plan = read(suite / 'prepared/plan.json')
    assert len(plan['jobs']) == 72
    assert {(j['encoder'], j['source'], j['ratio'], j['loss'], j['seed']) for j in plan['jobs']} == {
        (enc, src, ratio, loss, seed) for enc in ['RoBERTa', 'BERT']
        for src in ['coaid', 'fakenewsnet', 'liar'] for ratio in [.5, .8]
        for loss in ['unweighted_ce', 'weighted_ce'] for seed in range(3)}
    return plan, summary['plan_sha256']


def authorize(suite, plan_hash, execute):
    a = read(Path(suite) / 'approval_record.json')
    if not execute or a.get('status') != 'approved' or a.get('approved_plan_sha256') != plan_hash or not a.get('approval_text'):
        raise PermissionError('Execution requires the approved exact 72-job plan and --execute.')
    return a


def fit_scaler(training_features):
    x = np.asarray(training_features, dtype=np.float64)
    assert x.ndim == 2 and np.isfinite(x).all()
    mean, sd = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale = np.where(sd == 0, 1., sd)
    return mean, scale, sd == 0


def transform(x, mean, scale):
    z = ((np.asarray(x, dtype=np.float64) - mean) / scale).astype(np.float32)
    assert np.isfinite(z).all()
    return torch.from_numpy(z)


def state_hash(state):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in state.values())).hexdigest()


def preflight(suite, assets):
    suite, assets = Path(suite), Path(assets)
    plan, h = read_plan(suite)
    for name, sha in plan['frozen_files'].items():
        assert digest(suite / name) == sha, name
    splits = {s: pd.read_csv(suite / f'prepared/splits/{s}_s42.csv') for s in plan['spec']['sources']}
    for s, split in splits.items():
        assert split.row_index.tolist() == list(range(len(split)))
        assert set(split.label) == {0, 1}
        assert split.groupby('text_group').split.nunique().max() == 1
        assert set(split.split) == {'train', 'val', 'test'}
    for j in plan['jobs']:
        assert j['job_sha256'] == fingerprint({k: v for k, v in j.items() if k != 'job_sha256'})
        ids = pd.read_csv(suite / 'prepared' / j['cohort']).row_index
        split = splits[j['source']]
        chosen = split.set_index('row_index').loc[ids]
        assert ids.is_unique and len(chosen) == j['fixed_n'] and set(chosen.split) == {'train'}
        assert int(chosen.label.sum()) == round(j['ratio'] * j['fixed_n'])
        assert digest(suite / 'prepared' / j['cohort']) == j['cohort_sha256']
        assert digest(suite / 'prepared' / j['split']) == j['split_sha256']
    for c in plan['caches']:
        folder = assets / 'cache' / c['encoder'] / c['source']
        assert (folder / 'vectors.pt').stat().st_size == c['bytes']
        assert digest(folder / 'vectors.pt') == c['sha256']
        assert digest(folder / 'metadata.json') == c['metadata_sha256']
        x = torch.load(folder / 'vectors.pt', map_location='cpu', weights_only=True)
        assert tuple(x.shape) == (len(splits[c['source']]), 768)
        assert x.dtype == torch.float32 and torch.isfinite(x).all()
    baseline = read(suite / 'prepared/baseline_metadata.json')
    assert len(baseline) == 72
    # Forward-only initialization check: no fitting or optimizer is involved.
    initial = {}
    for seed in range(3):
        model = make_model('tbase', 768, 256, .3)
        state = torch.load(suite / f'prepared/initial_states/seed{seed}.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        initial[seed] = state_hash(state)
    for j in plan['jobs']:
        assert initial[j['seed']] == baseline[j['baseline_job_id']]['initial_parameter_sha256']
    return {'status': 'PASS', 'plan_sha256': h, 'planned_fits': 72,
            'cached_matrices': 6, 'training_executed': False}


class Study:
    def __init__(self, suite, assets, out, device):
        self.suite, self.assets, self.out = Path(suite), Path(assets), Path(out)
        self.plan, self.h = read_plan(suite)
        self.device = torch.device(device)
        self.splits = {s: pd.read_csv(self.suite / f'prepared/splits/{s}_s42.csv') for s in self.plan['spec']['sources']}
        self.raw = {(c['encoder'], c['source']): torch.load(
            self.assets / 'cache' / c['encoder'] / c['source'] / 'vectors.pt',
            map_location='cpu', weights_only=True) for c in self.plan['caches']}
        self.baseline = read(self.suite / 'prepared/baseline_metadata.json')
        self.out.mkdir(parents=True, exist_ok=True)
        env = {'plan_sha256': self.h, 'python': platform.python_version(),
               'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
               'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': str(self.device),
               'gpu': torch.cuda.get_device_name(0) if self.device.type == 'cuda' else None,
               'threads': torch.get_num_threads(), 'float32_matmul_precision': torch.get_float32_matmul_precision(),
               'cuda_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
               'code_sha256': {p.name: digest(p) for p in sorted((self.suite / 'code').glob('*.py'))}}
        ep = self.out / 'execution_environment.json'
        if ep.exists():
            assert read(ep) == env, 'Runtime/code changed. Stop; do not mix execution environments in this result folder.'
        else:
            write(ep, env)

    def rows(self, job, target, part):
        split = self.splits[target]
        q = split[split.split == part]
        if part == 'test':
            groups = set(self.splits[job['source']].query("split == 'train'").text_group)
            q = q[~q.text_group.isin(groups)]
        return q.row_index.to_numpy(), q.label.to_numpy(int)

    def score(self, model, x):
        with torch.no_grad():
            z = model(x.to(self.device)).detach().cpu().numpy()
        margin = z[:, 1].astype(np.float64) - z[:, 0].astype(np.float64)
        assert np.isfinite(margin).all()
        return margin

    def execute(self, job):
        start = time.perf_counter()
        dest = self.out / 'results' / job['job_id']
        if (dest / 'result.json').exists():
            saved = read(dest / 'result.json')
            assert saved['job'] == job and saved['plan_sha256'] == self.h
            for name, sha in saved['files_sha256'].items():
                assert digest(dest / name) == sha
            print('Already complete and hash-checked:', job['job_id'], flush=True)
            return
        if dest.exists():
            raise FileExistsError('Unexpected incomplete final directory; stop and inspect ' + str(dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        pending = dest.parent / (job['job_id'] + '.pending')
        marker = {'job_sha256': job['job_sha256'], 'plan_sha256': self.h}
        if pending.exists():
            assert read(pending / 'attempt.json') == marker, 'Unknown partial attempt; stop.'
            archive = self.out / 'incomplete_attempts'
            archive.mkdir(exist_ok=True)
            pending.rename(archive / (job['job_id'] + '_' + uuid.uuid4().hex))
            print('Preserved unfinished attempt; restarting this job from its seed.', flush=True)
        pending.mkdir()
        write(pending / 'attempt.json', marker)
        cohort = pd.read_csv(self.suite / 'prepared' / job['cohort']).row_index.to_numpy()
        source_raw = self.raw[job['encoder'], job['source']]
        mean, scale, zero = fit_scaler(source_raw[cohort].numpy())
        np.savez(pending / 'scaler.npz', mean=mean, scale=scale, zero_variance=zero)
        scaler_record = {'population': 'sampled source training cohort only',
                         'training_row_indices_sha256': fingerprint(cohort.tolist()),
                         'training_rows': len(cohort), 'dimensions': 768, 'ddof': 0,
                         'zero_variance_coordinates': int(zero.sum()), 'zero_sd_replacement': 1.,
                         'statistics_dtype': 'float64', 'transformed_dtype': 'float32'}
        write(pending / 'scaler_record.json', scaler_record)
        # The same source-cohort transform is used unchanged on every target.
        features = {s: transform(self.raw[job['encoder'], s].numpy(), mean, scale)
                    for s in self.plan['spec']['sources']}
        y = self.splits[job['source']].set_index('row_index').label.loc[cohort].to_numpy(int)
        xt = features[job['source']][cohort].to(self.device)
        yt = torch.as_tensor(y, dtype=torch.long, device=self.device)
        vi, vy = self.rows(job, job['source'], 'val')
        vx = features[job['source']][vi]
        tests = {s: self.rows(job, s, 'test') for s in self.plan['spec']['sources']}
        seed_everything(job['seed'])
        model = make_model('tbase', 768, 256, .3).to(self.device).eval()
        model.load_state_dict(torch.load(self.suite / f"prepared/initial_states/seed{job['seed']}.pt",
                                         map_location=self.device, weights_only=True))
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        initial_hash = state_hash(initial)
        assert initial_hash == self.baseline[job['baseline_job_id']]['initial_parameter_sha256'], 'Paired initialization changed'
        torch.save(initial, pending / 'initial_model.pt')
        initial_val_margin = self.score(model, vx)
        initial_val = metrics(vy, initial_val_margin, (initial_val_margin > 0).astype(int))
        initial_scores = {s: self.score(model, features[s][ids]) for s, (ids, _) in tests.items()}
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        weights = None if job['loss'] == 'unweighted_ce' else torch.as_tensor(
            len(y) / (2 * np.bincount(y, minlength=2)), dtype=torch.float32, device=self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        trajectory, best, selected_update, selected_state = [], -1., None, None
        for u in range(30):
            lr = .001 * core_lr(u, 30)
            for group in opt.param_groups:
                group['lr'] = lr
            model.train()
            opt.zero_grad()
            loss = criterion(model(xt), yt)
            assert torch.isfinite(loss), 'Non-finite loss; stop without changing hyperparameters'
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(gradient_norm), 'Non-finite gradient; stop'
            opt.step()
            model.eval()
            margin = self.score(model, vx)
            val = metrics(vy, margin, (margin > 0).astype(int))
            trajectory.append({'optimizer_update': u + 1, 'learning_rate': lr,
                               'training_loss': float(loss.detach().cpu()), 'source_validation': val})
            if val['macro_f1'] > best:
                best, selected_update = val['macro_f1'], u + 1
                selected_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(final_state, pending / 'final_model.pt')
        fm = self.score(model, vx)
        final_val = metrics(vy, fm, (fm > 0).astype(int))
        final_scores = {s: self.score(model, features[s][ids]) for s, (ids, _) in tests.items()}
        model.load_state_dict(selected_state)
        model.eval()
        torch.save(selected_state, pending / 'selected_model.pt')
        vm = self.score(model, vx)
        selected_val = metrics(vy, vm, (vm > 0).astype(int))
        pd.DataFrame({'row_index': vi, 'label': vy, 'initial_margin': initial_val_margin,
                      'selected_margin': vm, 'final_margin': fm}).to_csv(pending / 'validation_scores.csv', index=False)
        threshold, valf = best_threshold(vy, vm)
        effective = .5 if job['loss'] == 'weighted_ce' else float(y.mean())
        prior = float(vy.mean())
        offset = float(np.log(prior / (1 - prior)) - np.log(effective / (1 - effective)))
        decisions, endpoints = [], []
        for target, (ids, labels) in tests.items():
            ss = self.score(model, features[target][ids])
            preds = {'raw': (ss > 0).astype(int), 'source_prior': ((ss + offset) > 0).astype(int),
                     'source_val_threshold': (ss >= threshold).astype(int)}
            for rule, pred in preds.items():
                decisions.append({'job_id': job['job_id'], 'baseline_job_id': job['baseline_job_id'],
                                  **{k: job[k] for k in ['encoder', 'source', 'ratio', 'loss', 'seed']},
                                  'family': 'standardized_mlp', 'endpoint': 'selected', 'target': target,
                                  'rule': rule, **metrics(labels, ss, pred)})
            row = {'row_index': ids, 'label': labels}
            for endpoint, scores in [('initial', initial_scores[target]), ('selected', ss), ('final', final_scores[target])]:
                row[endpoint + '_margin'] = scores
                endpoints.append({'target': target, 'endpoint': endpoint, **metrics(labels, scores, (scores > 0).astype(int))})
            row.update({rule + '_prediction': pred for rule, pred in preds.items()})
            pd.DataFrame(row).to_csv(pending / f'test_{target}_scores.csv', index=False)
        write(pending / 'trajectory.json', trajectory)
        result = {'job': job, 'plan_sha256': self.h, 'new_training_executed': True,
                  'initial_parameter_sha256': initial_hash, 'scaler_record': scaler_record,
                  'initialization_validation': initial_val, 'selected_validation': selected_val,
                  'final_validation': final_val, 'selected_optimizer_update': selected_update,
                  'training_ever_validation_collapsed': any(t['source_validation']['collapsed'] for t in trajectory),
                  'first_collapsed_post_update': next((t['optimizer_update'] for t in trajectory if t['source_validation']['collapsed']), None),
                  'checkpoint_test_metrics': endpoints, 'evaluations': decisions,
                  'decision_record': {'source_validation_threshold': threshold, 'source_validation_macro_f1': valf,
                                      'source_validation_prior': prior, 'effective_training_prior': effective, 'offset': offset},
                  'elapsed_seconds': time.perf_counter() - start,
                  'files_sha256': {p.name: digest(p) for p in pending.iterdir()}}
        write(pending / 'result.json', result)
        pending.rename(dest)
        print(f"Completed {job['job_id']} | selected update {selected_update} | {result['elapsed_seconds']:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', required=True)
    ap.add_argument('--assets', required=True)
    ap.add_argument('--out')
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(2)
    print(preflight(a.suite, a.assets), flush=True)
    if not a.execute:
        return
    plan, h = read_plan(a.suite)
    approval = authorize(a.suite, h, a.execute)
    assert a.out, '--out is required for execution'
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; select a GPU runtime or explicitly use CPU before starting')
    print('Training device:', device,
          torch.cuda.get_device_name(0) if device == 'cuda' else '(CPU; GPU is optional)', flush=True)
    study = Study(a.suite, a.assets, a.out, device)
    write(Path(a.out) / 'approval_record.json', approval)
    for n, job in enumerate(plan['jobs'], 1):
        print(f'Job {n}/72', flush=True)
        study.execute(job)
    write(Path(a.out) / 'execution_finished.json', {'plan_sha256': h, 'planned_jobs': 72,
                                                   'execution_loop_finished': True, 'independent_verification_required': True})


if __name__ == '__main__':
    main()
