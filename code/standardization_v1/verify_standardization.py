"""Verify saved scaler, source-only rules, checkpoint replay and paired aggregates. Never trains."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from study_common import read, write, digest, fingerprint, make_model, core_lr
from controls import best_threshold, metrics
from standardize import read_plan, preflight, fit_scaler, transform, state_hash, Study


def same(saved, computed):
    for k, v in computed.items():
        if isinstance(v, (float, int, bool, np.number)):
            assert abs(float(saved[k]) - float(v)) < 1e-10, (k, saved[k], v)
        else:
            assert saved[k] == v, (k, saved[k], v)


def verify(suite, assets, out, replay=True):
    suite, assets, out = Path(suite), Path(assets), Path(out)
    preflight(suite, assets)
    plan, h = read_plan(suite)
    env = read(out / 'execution_environment.json')
    assert env['plan_sha256'] == h
    for name, expected in env['code_sha256'].items():
        assert digest(suite / 'code' / name) == expected
    splits = {s: pd.read_csv(suite / f'prepared/splits/{s}_s42.csv') for s in plan['spec']['sources']}
    baseline = read(suite / 'prepared/baseline_metadata.json')
    raw = {(c['encoder'], c['source']): torch.load(assets / 'cache' / c['encoder'] / c['source'] / 'vectors.pt',
           map_location='cpu', weights_only=True) for c in plan['caches']}
    device = torch.device(env['device'])
    if replay and device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Use the original GPU runtime for numerical checkpoint replay.')
    decisions, endpoints, diagnostics = [], [], []
    for index, job in enumerate(plan['jobs'], 1):
        folder = out / 'results' / job['job_id']
        assert (folder / 'result.json').is_file(), 'Missing job: ' + job['job_id']
        r = read(folder / 'result.json')
        assert r['job'] == job and r['plan_sha256'] == h and r['new_training_executed'] is True
        for name, expected in r['files_sha256'].items():
            assert digest(folder / name) == expected
        cohort = pd.read_csv(suite / 'prepared' / job['cohort']).row_index.to_numpy()
        mean, scale, zero = fit_scaler(raw[job['encoder'], job['source']][cohort].numpy())
        scaler = np.load(folder / 'scaler.npz')
        assert np.array_equal(scaler['mean'], mean) and np.array_equal(scaler['scale'], scale)
        assert np.array_equal(scaler['zero_variance'], zero)
        assert r['scaler_record']['training_row_indices_sha256'] == fingerprint(cohort.tolist())
        assert r['scaler_record']['population'] == 'sampled source training cohort only'
        trajectory = read(folder / 'trajectory.json')
        assert len(trajectory) == 30 and [t['optimizer_update'] for t in trajectory] == list(range(1, 31))
        assert all(np.isfinite(t['training_loss']) and t['learning_rate'] == .001 * core_lr(i, 30) for i, t in enumerate(trajectory))
        best = max(trajectory, key=lambda t: t['source_validation']['macro_f1'])
        assert r['selected_optimizer_update'] == best['optimizer_update']
        same(r['selected_validation'], best['source_validation'])
        same(r['final_validation'], trajectory[-1]['source_validation'])
        assert r['training_ever_validation_collapsed'] == any(t['source_validation']['collapsed'] for t in trajectory)
        assert r['first_collapsed_post_update'] == next((t['optimizer_update'] for t in trajectory if t['source_validation']['collapsed']), None)
        initial = torch.load(folder / 'initial_model.pt', map_location='cpu', weights_only=True)
        assert state_hash(initial) == r['initial_parameter_sha256'] == baseline[job['baseline_job_id']]['initial_parameter_sha256']
        vp = pd.read_csv(folder / 'validation_scores.csv')
        vs = splits[job['source']].query("split == 'val'")
        assert vp.row_index.tolist() == vs.row_index.tolist() and vp.label.tolist() == vs.label.tolist()
        threshold, valf = best_threshold(vp.label.to_numpy(), vp.selected_margin.to_numpy())
        cy = splits[job['source']].set_index('row_index').label.loc[cohort].to_numpy(int)
        prior = float(vp.label.mean())
        effective = .5 if job['loss'] == 'weighted_ce' else float(cy.mean())
        offset = float(np.log(prior / (1 - prior)) - np.log(effective / (1 - effective)))
        dr = r['decision_record']
        assert abs(threshold - dr['source_validation_threshold']) < 1e-12
        assert abs(valf - dr['source_validation_macro_f1']) < 1e-12
        assert abs(offset - dr['offset']) < 1e-12
        assert prior == dr['source_validation_prior'] and effective == dr['effective_training_prior']
        groups = set(splits[job['source']].query("split == 'train'").text_group)
        targets = {}
        for target in plan['spec']['sources']:
            ts = splits[target]
            ts = ts[(ts.split == 'test') & ~ts.text_group.isin(groups)]
            p = pd.read_csv(folder / f'test_{target}_scores.csv')
            assert p.row_index.tolist() == ts.row_index.tolist() and p.label.tolist() == ts.label.tolist()
            targets[target] = p
            for endpoint in ['initial', 'selected', 'final']:
                margin = p[endpoint + '_margin'].to_numpy()
                computed = metrics(p.label, margin, (margin > 0).astype(int))
                saved = next(t for t in r['checkpoint_test_metrics'] if t['target'] == target and t['endpoint'] == endpoint)
                same(saved, computed)
                endpoints.append({**{k: job[k] for k in ['job_id', 'baseline_job_id', 'encoder', 'source', 'ratio', 'loss', 'seed']}, **saved})
            ss = p.selected_margin.to_numpy()
            for rule in ['raw', 'source_prior', 'source_val_threshold']:
                pred = (ss >= threshold).astype(int) if rule == 'source_val_threshold' else ((ss + offset) > 0).astype(int) if rule == 'source_prior' else (ss > 0).astype(int)
                assert np.array_equal(pred, p[rule + '_prediction'])
                saved = next(t for t in r['evaluations'] if t['target'] == target and t['rule'] == rule)
                same(saved, metrics(p.label, ss, pred))
                decisions.append(saved)
        for endpoint, key in [('initial', 'initialization_validation'), ('selected', 'selected_validation'), ('final', 'final_validation')]:
            ss = vp[endpoint + '_margin'].to_numpy()
            same(r[key], metrics(vp.label, ss, (ss > 0).astype(int)))
        if replay:
            model = make_model('tbase', 768, 256, .3).to(device).eval()
            transformed = {s: transform(raw[job['encoder'], s].numpy(), mean, scale) for s in plan['spec']['sources']}
            for endpoint in ['initial', 'selected', 'final']:
                state = torch.load(folder / (endpoint + '_model.pt'), map_location=device, weights_only=True)
                model.load_state_dict(state)
                for target, p in targets.items():
                    scores = Study.score(type('Replay', (), {'device': device})(), model, transformed[target][p.row_index.to_numpy()])
                    recorded = p[endpoint + '_margin'].to_numpy()
                    assert np.allclose(scores, recorded, atol=1e-6, rtol=1e-5)
                    assert np.array_equal(scores > 0, recorded > 0)
                scores = Study.score(type('Replay', (), {'device': device})(), model, transformed[job['source']][vp.row_index.to_numpy()])
                recorded = vp[endpoint + '_margin'].to_numpy()
                assert np.allclose(scores, recorded, atol=1e-6, rtol=1e-5) and np.array_equal(scores > 0, recorded > 0)
        diagnostics.append({**{k: job[k] for k in ['job_id', 'baseline_job_id', 'encoder', 'source', 'ratio', 'loss', 'seed']},
                            'selected_update': r['selected_optimizer_update'],
                            'initial_validation_collapse': r['initialization_validation']['collapsed'],
                            'training_ever_validation_collapse': r['training_ever_validation_collapsed'],
                            'selected_validation_collapse': r['selected_validation']['collapsed'],
                            'final_validation_collapse': r['final_validation']['collapsed'], 'elapsed_seconds': r['elapsed_seconds']})
        print(f'Verified job {index}/72', flush=True)
    frame = pd.DataFrame(decisions)
    assert len(frame) == 648 and not frame.duplicated(['job_id', 'target', 'rule']).any()
    analysis = out / 'analysis'
    analysis.mkdir(exist_ok=True)
    frame.to_csv(analysis / 'decision_evaluations.csv', index=False)
    pd.DataFrame(endpoints).to_csv(analysis / 'endpoint_evaluations.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(analysis / 'job_diagnostics.csv', index=False)
    base = pd.read_csv(suite / 'prepared/baseline_decisions.csv')
    pairs = frame.merge(base, left_on=['baseline_job_id', 'target', 'rule'],
                        right_on=['job_id', 'target', 'rule'], suffixes=('_standardized', '_raw_features'), validate='one_to_one')
    assert len(pairs) == 648
    for field in ['macro_f1', 'auroc', 'average_precision', 'mcc', 'pred_class1_fraction']:
        pairs['delta_' + field] = pairs[field + '_standardized'] - pairs[field + '_raw_features']
    pairs.to_csv(analysis / 'paired_comparison.csv', index=False)
    frame.groupby(['encoder', 'source', 'target', 'ratio', 'loss', 'rule']).agg(
        evaluations=('macro_f1', 'size'), collapse_count=('collapsed', 'sum'),
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_sd=('macro_f1', 'std'), auroc_mean=('auroc', 'mean'),
        mcc_mean=('mcc', 'mean')).reset_index().to_csv(analysis / 'condition_summary.csv', index=False)
    report = {'status': 'PASS', 'complete': True, 'plan_sha256': h, 'planned_new_fits': 72,
              'verified_complete_job_records': 72, 'verified_post_update_trajectory_records': 2160,
              'verified_decision_rule_target_records': 648, 'checkpoint_weight_replay': replay,
              'new_training_executed_by_verifier': False,
              'interpretation': 'Three paired seeds on one fixed split; records reuse heads across rules/targets and are not independent replications.'}
    write(analysis / 'completion.json', report)
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', required=True)
    ap.add_argument('--assets', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    print(verify(a.suite, a.assets, a.out))
