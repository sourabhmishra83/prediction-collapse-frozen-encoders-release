from pathlib import Path
import pandas as pd, json, hashlib, zipfile, collections, numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRIVATE = ROOT / 'private'
PRIVATE.mkdir(exist_ok=True)
RAW = Path(str(Path.home() / 'Documents/Sourabh/UT/Fall2025/Project/raw/FakeHealth'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def key(x):
    return hashlib.sha256(' '.join(str(x).casefold().split()).encode()).hexdigest()
if (PRIVATE / 'input_manifest.json').exists():
    raise SystemExit('Frozen inputs exist; use a new version directory to rebuild')
claims = pd.read_csv(RAW / 'claims.csv', keep_default_na=False).set_index('id')
gold = []
for chunk in pd.read_csv(RAW / 'relation_annotations.csv', chunksize=100000, keep_default_na=False):
    h = chunk[(chunk.annotation_type_id == 3) & (chunk.annotation_category == 'label')]
    for r in h.itertuples():
        value = json.loads(r.value).get('value')
        if value in ['supporting', 'contradicting', 'neutral']:
            gold.append({'article_id': int(r.source_entity_id), 'claim_id': int(r.target_entity_id), 'label': value})
gold = pd.DataFrame(gold).drop_duplicates()
assert gold.groupby(['article_id', 'claim_id']).label.nunique().max() == 1
assert len(gold) == 308
articles = set(gold.article_id)
testclaims = set(gold.claim_id)
testclaimkeys = {key(claims.loc[c, 'statement']) for c in testclaims}
auto = []
a = pd.read_csv(RAW / 'entity_annotations.csv', keep_default_na=False)
a = a[(a.annotation_type_id == 6) & (a.entity_type == 'articles') & (a.annotation_category == 'prediction')]
for r in a.itertuples():
    for c in json.loads(r.value).get('claims', []):
        cid = int(c['claim_id'])
        if c.get('presence') == 'yes' and c.get('stance') in ['supporting', 'contradicting', 'neutral'] and (int(r.entity_id) not in articles) and (cid not in testclaims) and (cid in claims.index) and (key(claims.loc[cid, 'statement']) not in testclaimkeys):
            auto.append({'article_id': int(r.entity_id), 'claim_id': cid, 'label': c['stance']})
auto = pd.DataFrame(auto).drop_duplicates()
assert auto.groupby(['article_id', 'claim_id']).label.nunique().max() == 1
wanted = set(auto.article_id) | articles
texts = {}
for chunk in pd.read_csv(RAW / 'articles.csv', usecols=['id', 'title', 'body'], chunksize=6000, keep_default_na=False):
    for r in chunk[chunk.id.isin(wanted)].itertuples():
        texts[int(r.id)] = (str(r.title).strip() + '. ' + str(r.body).strip()).strip()
    if len(texts) == len(wanted):
        break
assert set(texts) == wanted and all((texts[i].strip('. ') for i in wanted))
for d in [gold, auto]:
    d['article_text'] = d.article_id.map(texts)
    d['article_group'] = d.article_text.map(key)
    d['claim_text'] = d.claim_id.map(claims.statement)
    assert d.claim_text.str.strip().str.len().min() > 0
    d['claim_group'] = d.claim_text.map(key)
    d['row_id'] = [f'a{a}_c{c}' for a, c in zip(d.article_id, d.claim_id)]
initial = len(auto)
auto = auto[~auto.article_group.isin(set(gold.article_group))].copy()
parent = {}

def find(x):
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    parent[find(a)] = find(b)
for r in auto.itertuples():
    union('a' + r.article_group, 'c' + r.claim_group)
auto['component'] = [find('a' + x) for x in auto.article_group]
sizes = auto.groupby('component').size()
largest = sizes.idxmax()
candidates = sorted(set(sizes.index) - {largest})
rng = np.random.default_rng(20261017)
rng.shuffle(candidates)
chosen = []
counts = collections.Counter()
n = 0
for component in candidates:
    part = auto[auto.component == component]
    chosen.append(component)
    counts.update(part.label)
    n += len(part)
    if n >= 600 and min((counts.get(k, 0) for k in ['supporting', 'contradicting', 'neutral'])) >= 40:
        break
if n < 600 or min(counts.values()) < 40:
    raise ValueError('Validation class/size thresholds infeasible; design decision needed')
auto['split'] = np.where(auto.component.isin(chosen), 'validation', 'train')
gold['split'] = 'test'
gold['component'] = 'expert_test'
for field in ['article_id', 'claim_id', 'article_group', 'claim_group']:
    train = set(auto[auto.split == 'train'][field])
    val = set(auto[auto.split == 'validation'][field])
    test = set(gold[field])
    assert not train & val and (not train & test) and (not val & test)
frame = pd.concat([auto, gold], ignore_index=True).sort_values(['split', 'row_id']).reset_index(drop=True)
frame['label_id'] = frame.label.map({'supporting': 0, 'contradicting': 1, 'neutral': 2})
assert frame.row_id.is_unique
path = PRIVATE / 'monant_stance_pairs_v1.csv.gz'
frame.to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0})
manifest = {'schema_version': 1, 'file': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size, 'labels': {'supporting': 0, 'contradicting': 1, 'neutral': 2}, 'rows': len(frame), 'split_seed': 20261017, 'split_counts': frame.groupby(['split', 'label']).size().to_dict(), 'removed_by_normalized_article_overlap': initial - len(auto), 'largest_training_component_pairs': int(sizes.max()), 'validation_policy': 'Non-largest components shuffled with seed 20261017 until >=600 pairs and >=40/class; largest component retained for training. Topology-biased weak-label validation, not representative of expert-test performance.', 'source_sha256': {n: sha(RAW / n) for n in ['claims.csv', 'entity_annotations.csv', 'relation_annotations.csv']}, 'article_text_policy': 'stripped title plus dot-space plus stripped body, no raw HTML fallback', 'weak_supervision': 'predicted presence=yes and predicted three-class stance; expert test labels excluded from all fits', 'training_executed': False}
manifest['split_counts'] = {a + '_' + b: int(v) for (a, b), v in manifest['split_counts'].items()}
(PRIVATE / 'input_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
with zipfile.ZipFile(PRIVATE / 'Monant_stance_private_inputs_v1.zip', 'w', zipfile.ZIP_STORED) as z:
    z.write(path, path.name)
    z.write(PRIVATE / 'input_manifest.json', 'input_manifest.json')
public = {k: v for k, v in manifest.items() if k not in ['source_sha256']}
(ROOT / 'input_preparation_summary.json').write_text(json.dumps(public, indent=2) + '\n')
print(json.dumps(public, indent=2))
