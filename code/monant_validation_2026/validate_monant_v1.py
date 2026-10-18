from pathlib import Path
import json, hashlib, collections, zipfile
import pandas as pd, numpy as np
ROOT = Path(str(Path.home() / 'Documents/Codex/2026-10-11/referenced-chatgpt-conversation-this-is-an'))
OUT = ROOT / 'outputs'
PRIVATE = ROOT / 'work/monant_validation_private'
PRIVATE.mkdir(exist_ok=True)
RAW = Path(str(Path.home() / 'Documents/Sourabh/UT/Fall2025/Project/raw/FakeHealth'))
SAMPLE = ROOT / 'work/monant_official_sample/sample_data'
with zipfile.ZipFile(str(Path.home() / 'Desktop/Paper/CSML-20260720T013115Z-1-002.zip')) as z:
    b = z.read('CSML/data/clean/fakehealth_clean.csv')
    import io
    clean = pd.read_csv(io.BytesIO(b))
assert clean.id.is_unique
annotations = pd.read_csv(RAW / 'entity_annotations.csv')
annotations = annotations[(annotations.annotation_type_id == 6) & (annotations.entity_type == 'articles')]
assert annotations.entity_id.is_unique
records = {int(r.entity_id): json.loads(r.value).get('claims', []) for r in annotations.itertuples()}
claims = pd.read_csv(RAW / 'claims.csv', keep_default_na=False)
claim_lookup = claims.set_index('id').to_dict('index')
binary = {'true': 0, 'mostly-true': 0, 'false': 1, 'mostly-false': 1}

def vote(cs, field):
    vals = [binary[c[field]] for c in cs if c.get(field) in binary]
    if not vals:
        return (None, False)
    n1 = sum(vals)
    return (int(n1 >= len(vals) - n1), n1 == len(vals) - n1)
rows = []
combinations = collections.Counter()
claimref_bad = 0
rating_bad = 0
for r in clean.itertuples():
    cs = records[int(r.id)]
    old, tie = vote(cs, 'rating')
    new, ntie = vote(cs, 'partial_veracity')
    rows.append({'article_id': int(r.id), 'saved_label': int(r.label), 'rating_vote': old, 'partial_vote': new, 'rating_tie': tie, 'partial_tie': ntie, 'stratum': f'class{r.label}_' + ('unresolved' if new is None else 'agree' if new == r.label else 'disagree')})
    for c in cs:
        combinations[c.get('presence'), c.get('stance')] += 1
        if c['claim_id'] not in claim_lookup:
            claimref_bad += 1
        elif c.get('rating') != claim_lookup[c['claim_id']]['rating']:
            rating_bad += 1
frame = pd.DataFrame(rows)
assert (frame.saved_label == frame.rating_vote).all()
h = pd.concat([chunk[(chunk.annotation_category == 'label') & chunk.annotation_type_id.isin([2, 3])] for chunk in pd.read_csv(RAW / 'relation_annotations.csv', chunksize=100000)], ignore_index=True)
h['decoded'] = h.value.map(lambda x: json.loads(x).get('value'))
placeholder = int((h.decoded == 'not-determined-yet').sum())
complete = h[h.decoded != 'not-determined-yet']
index = {}
conflicts = 0
for (aid, cid, kind), g in complete.groupby(['source_entity_id', 'target_entity_id', 'annotation_type_id']):
    values = set(g.decoded)
    if len(values) != 1:
        conflicts += 1
        continue
    index[int(aid), int(cid), int(kind)] = next(iter(values))
rating_mismatch_types = collections.Counter()
for aid in clean.id:
    for c in records[int(aid)]:
        if c['claim_id'] in claim_lookup and c.get('rating') != claim_lookup[c['claim_id']]['rating']:
            rating_mismatch_types[str(c.get('rating')), str(claim_lookup[c['claim_id']]['rating'])] += 1
comp = []
human_votes = collections.defaultdict(list)
for aid in clean.id:
    for c in records[int(aid)]:
        pair = (int(aid), int(c['claim_id']))
        presence = index.get((*pair, 2))
        stance = index.get((*pair, 3))
        if presence is not None:
            comp.append({'article_id': int(aid), 'claim_id': pair[1], 'kind': 'presence', 'manual': presence, 'automatic': {'yes': 'present', 'no': 'not-present'}.get(c.get('presence'), c.get('presence'))})
        if stance is not None:
            comp.append({'article_id': int(aid), 'claim_id': pair[1], 'kind': 'stance', 'manual': stance, 'automatic': c.get('stance')})
        if presence == 'present' and stance in ['supporting', 'contradicting'] and (c.get('rating') in binary):
            human_votes[int(aid)].append(binary[c['rating']] ^ (stance == 'contradicting'))
comparison = pd.DataFrame(comp)
comparison['agreement'] = comparison.manual == comparison.automatic
comparison.to_csv(PRIVATE / 'manual_pair_comparisons.csv', index=False)
manual_summary = comparison.groupby('kind').agg(compared_pairs=('agreement', 'size'), agreeing=('agreement', 'sum'), articles=('article_id', 'nunique')).to_dict('index') if len(comparison) else {}
sample_report = {}
for name in ['articles', 'claims', 'entity_annotations', 'relation_annotations']:
    d = pd.read_csv(SAMPLE / (name + '.csv'))
    sample_report[name] = {'rows': len(d), 'columns': d.columns.tolist()}
    if 'annotation_category' in d:
        sample_report[name]['annotation_counts'] = {str(k): int(v) for k, v in d.groupby(['annotation_type_id', 'annotation_category']).size().items()}
sa = pd.read_csv(SAMPLE / 'entity_annotations.csv')
joined = sa.merge(annotations, on='id', suffixes=('_sample', '_full'))
sample_report['entity_annotations']['matched_full_ids'] = len(joined)
sample_report['entity_annotations']['identical_values'] = sum((json.loads(x) == json.loads(y) for x, y in zip(joined.value_sample, joined.value_full)))
sc = pd.read_csv(SAMPLE / 'claims.csv')
scj = sc.merge(claims, on='id', suffixes=('_sample', '_full'))
sample_report['claims']['matched_full_ids'] = len(scj)
sample_report['claims']['rating_matches'] = int((scj.rating_sample.fillna('') == scj.rating_full.fillna('')).sum())
selected = []
strata = []
for label, g in frame.groupby('stratum'):
    n = min(20, len(g))
    chosen = g.sample(n=n, random_state=20261017).copy()
    chosen['inclusion_probability'] = n / len(g)
    selected.append(chosen)
    strata.append({'stratum': label, 'population': len(g), 'sample': n, 'inclusion_probability': n / len(g)})
selection = pd.concat(selected).sort_values('article_id')
selection.to_csv(PRIVATE / 'sampling_frame_key.csv', index=False)
lookup = clean.set_index('id')
blinded = []
casebook = []
for i, r in enumerate(selection.itertuples(), 1):
    review_id = f'MON-{i:03d}'
    cs = records[r.article_id]
    blinded.append({'review_id': review_id, 'article_id': r.article_id, 'text': lookup.loc[r.article_id, 'text'], 'claims': [{'claim_id': c['claim_id'], 'statement': claim_lookup.get(c['claim_id'], {}).get('statement', '')} for c in cs]})
    casebook.append({'review_id': review_id, 'article_id': r.article_id, 'saved_label': r.saved_label, 'stratum': r.stratum, 'source_annotations': cs})
(PRIVATE / 'blinded_review_articles.jsonl').write_text('\n'.join((json.dumps(v, ensure_ascii=False) for v in blinded)) + '\n', encoding='utf-8')
(PRIVATE / 'reviewer_key.jsonl').write_text('\n'.join((json.dumps(v) for v in casebook)) + '\n')
pd.DataFrame([{'review_id': r['review_id'], 'reviewer_id': '', 'claim_presence': '', 'article_stance': '', 'article_assessable': '', 'article_label': '', 'evidence_reference': '', 'reason': '', 'confidence': ''} for r in blinded]).to_csv(PRIVATE / 'reviewer_ratings_blank.csv', index=False)
frame.to_csv(PRIVATE / 'all_derivative_label_diagnostics.csv', index=False)
summary = {'clean_rows': len(clean), 'clean_sha256': hashlib.sha256(b).hexdigest(), 'saved_labels_match_rating_vote': int((frame.saved_label == frame.rating_vote).sum()), 'partial_resolved': int(frame.partial_vote.notna().sum()), 'partial_disagreements': int(((frame.partial_vote != frame.saved_label) & frame.partial_vote.notna()).sum()), 'partial_unresolved': int(frame.partial_vote.isna().sum()), 'rating_ties': int(frame.rating_tie.sum()), 'partial_ties': int(frame.partial_tie.sum()), 'all_clean_annotation_categories': annotations[annotations.entity_id.isin(clean.id)].annotation_category.value_counts().to_dict(), 'claim_references_missing': claimref_bad, 'claim_rating_raw_mismatches': rating_bad, 'claim_rating_mismatch_types': {str(k): v for k, v in rating_mismatch_types.items()}, 'completed_manual_annotation_rows': len(complete), 'placeholder_manual_rows_excluded': placeholder, 'conflicting_completed_pair_types_excluded': conflicts, 'manual_comparison_on_derivative_pairs': manual_summary, 'articles_with_present_and_directional_manual_pair_evidence': len(human_votes), 'official_sample': sample_report, 'review_sample': strata, 'review_sample_size': len(selection), 'review_seed': 20261017, 'independent_human_article_truth_review_completed': False, 'raw_texts_uploaded': False, 'official_commit': json.loads((ROOT / 'work/monant_official_sample/download_manifest.json').read_text())['commit']}
(OUT / 'Discover_Monant_validation_v1.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'official_sample'}, indent=2))
