import pandas as pd, json, sys, re, collections
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
s = pd.read_csv('work/monant_official_sample/sample_data/articles.csv', keep_default_na=False).set_index('id')
ids = set(s.index)
found = set()
counts = collections.Counter()
examples = []
norm = lambda x: re.sub('\\s+', ' ', str(x)).strip()
for chunk in pd.read_csv(str(Path.home() / 'Documents/Sourabh/UT/Fall2025/Project/raw/FakeHealth/articles.csv'), usecols=['id', 'title', 'body'], keep_default_na=False, chunksize=6000):
    for r in chunk[chunk.id.isin(ids)].itertuples():
        found.add(r.id)
        t = s.loc[r.id, 'title']
        b = s.loc[r.id, 'body']
        counts['title_exact'] += r.title == t
        counts['body_exact'] += r.body == b
        counts['body_whitespace_equivalent'] += norm(r.body) == norm(b)
        counts['both_whitespace_equivalent'] += norm(r.title) == norm(t) and norm(r.body) == norm(b)
        if len(examples) < 2:
            examples.append({'id': r.id, 'sample_title': t[:90], 'full_title': r.title[:90], 'sample_body_len': len(b), 'full_body_len': len(r.body), 'sample_body_prefix': b[:90], 'full_body_prefix': r.body[:90]})
    if len(found) == len(ids):
        break
p = Path('outputs/Discover_Monant_validation_v1.json')
v = json.loads(p.read_text())
v['official_sample']['articles'].update(matched_full_ids=len(found), comparisons=dict(counts))
p.write_text(json.dumps(v, indent=2) + '\n')
print(dict(counts))
print(json.dumps(examples, ensure_ascii=False))
