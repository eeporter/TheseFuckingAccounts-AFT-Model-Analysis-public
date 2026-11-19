#!/usr/bin/env python3
"""Sample page of matched pairs for Emily to read, 2026-11-17. PRIVATE: the page holds real comment text.
It must stay a local file (Hyak, then Emily's Mac). Never publish it, never commit it, never put it in shared folders.

Reads the full-run pairs (full_out_v1), looks up a few archive files to find the text of sampled archive items, and
writes ONE html file with --n pairs per length band:
    50-99, 100-139, 140+     (band = the SHORTER of the two visible lengths, so the bands do not overlap;
                              the 50 version = all three bands, the 100 version = the last two, the 140 version = the last)
Sampling: a few archive files are picked at random (fixed seed); among the pairs whose archive item is in those files,
texts are picked at random within each band (each accused text at most once on the page), then one of its earlier
copies at random. No usernames are written: the page only says whether the earlier copy is by the same account,
by a deleted author, or by someone else. Prints counts only.

Inputs: --pairs (full-run output folder, pair files), --lsh (candidates_v1.parquet: the accused texts),
--arch (archive root; older-batch files only, R.NEWER skipped). Options: --n per band, --seed, --bands, --max-files.
Outputs (in --out): sample_matches_v1.html (PRIVATE, holds texts; the job script sets it to mode 600) and
sample_summary_v1.json (counts only).
Job: run_sample_page_v1.sh, job 43331457 (full_out_v1, lsh/, review_v1/).
Steps: (1) read the pairs (ids, times, lengths, author, Jaccard; no text) and put each pair in a band;
(2) read only the id column of randomly ordered archive files (two comment files, then one post file, repeating)
until every band has enough distinct accused texts or --max-files files were read (at least 3 files);
(3) sample texts per band; (4) read the texts: accused texts from candidates_v1, archive texts from the exact row
group and row of the archive file (the id is asserted to match); (5) write the HTML page, each text HTML-escaped and
cut at 6,000 characters. The author name is used only to classify same / deleted / different and is not written."""
import argparse, glob, html, json, os, sys, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules2 as R

ap = argparse.ArgumentParser()
ap.add_argument('--pairs', required=True); ap.add_argument('--lsh', required=True); ap.add_argument('--arch', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=20); ap.add_argument('--seed', type=int, default=17)
ap.add_argument('--bands', default='50,100,140'); ap.add_argument('--max-files', type=int, default=16)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)
B = [int(x) for x in a.bands.split(',')]
names = [f'{B[i]} to {B[i + 1] - 1} characters' for i in range(len(B) - 1)] + [f'{B[-1]} or more characters']
T = {'bands': names, 'n_per_band': a.n, 'seed': a.seed}; t0 = time.time()

cols = ['ckey', 'aid', 'akind', 'a_author', 'a_subreddit', 'a_created_utc', 'q_created_utc', 'q_acc_utc', 'q_vis_len', 'a_vis_len', 'jaccard']
P = pa.concat_tables([pq.read_table(f, columns=cols) for f in sorted(glob.glob(f'{a.pairs}/pairs_task*_p*.parquet'))]).to_pandas()
T['pairs'] = len(P)
P['short'] = np.minimum(P.q_vis_len, P.a_vis_len)
# band index b: B[b] <= shorter visible length < B[b + 1] (last band open-ended)
P['band'] = np.searchsorted(np.array(B), P.short.values, side='right') - 1
T['pairs_by_band'] = {names[b]: int((P.band == b).sum()) for b in range(len(B))}
T['texts_by_band'] = {names[b]: int(P[P.band == b].ckey.nunique()) for b in range(len(B))}
ncop = P.groupby('ckey', sort=False).size()

# ---- find archive items of pairs in a few archive files (id column only) ----
files = {k: sorted(f for f in glob.glob(f'{a.arch}/{d}/*.parquet') if not any(u in f for u in R.NEWER))
         for k, d in (('comment', 'reddit_comments_by_author.parquet'), ('post', 'reddit_submissions_by_author.parquet'))}
for k in files: files[k] = [files[k][i] for i in rng.permutation(len(files[k]))]
want = {k: pa.array(P[P.akind == k].aid.unique(), type=pa.string()) for k in files}
found = []; used = 0; pos = {'comment': 0, 'post': 0}


def enough():
    """True when the archive ids found so far cover, in every band, at least min(--n, texts in that band) distinct
    accused texts."""
    if not found: return False
    F = pd.concat(found); m = P.merge(F, on=['akind', 'aid'])
    return all(m[m.band == b].ckey.nunique() >= min(a.n, T['texts_by_band'][names[b]]) for b in range(len(B)))


order = ['comment', 'comment', 'post']
while used < a.max_files:
    k = order[used % 3]
    if pos[k] >= len(files[k]): k = 'comment' if k == 'post' else 'post'
    if pos[k] >= len(files[k]): break
    f = files[k][pos[k]]; pos[k] += 1; used += 1
    # whole-file id column; 'row' = position in the file (all row groups together)
    ids = pq.read_table(f, columns=['id'])['id'].combine_chunks()
    hit = np.flatnonzero(pc.is_in(ids, value_set=want[k]).to_numpy(zero_copy_only=False))
    found.append(pd.DataFrame({'akind': k, 'aid': ids.take(pa.array(hit)).to_pylist(), 'file': f, 'row': hit}))
    if used >= 3 and enough(): break
T['archive_files_looked_up'] = used
F = pd.concat(found).drop_duplicates(['akind', 'aid'])
M = P.merge(F, on=['akind', 'aid'])
T['pairs_found_in_those_files'] = len(M)

# ---- sample: texts at random within a band, one earlier copy each, each text once on the page ----
# highest band first, so a text whose copies fall in several bands is shown in the higher band
pick = []; seen = set()
for b in range(len(B) - 1, -1, -1):
    D = M[(M.band == b) & ~M.ckey.isin(seen)]
    ck = D.ckey.unique(); ck = ck[rng.permutation(len(ck))][:a.n]
    for c in ck:
        d = D[D.ckey == c]; pick.append(d.iloc[int(rng.integers(len(d)))]); seen.add(c)
S = pd.DataFrame(pick).reset_index(drop=True)
T['sampled_by_band'] = {names[b]: int((S.band == b).sum()) for b in range(len(B))}

# ---- texts ----
c = pq.read_table(f'{a.lsh}/candidates_v1.parquet', columns=['kind', 'id', 'outer_key', 'text']).to_pandas()
c['ckey'] = c.kind.astype(str) + ':' + c.id.astype(str)
c = c.drop_duplicates('ckey').set_index('ckey')
S['q_text'] = c.text.reindex(S.ckey).fillna('').values
S['outer_key'] = c.outer_key.reindex(S.ckey).values
au = S.a_author.fillna('[deleted]').astype(str).str.lower()
# who wrote the earlier copy, relative to the accused account (only this label is written, never a name)
S['who'] = np.where(au.isin(['[deleted]', '[removed]', '']), 'deleted author',
                    np.where(au == S.outer_key.astype(str).str.lower(), 'SAME account as the accused text', 'a different account'))
S['a_text'] = ''
for f, d in S.groupby('file'):
    pf = pq.ParquetFile(f); kind = R._kind(f)
    # map file-wide row positions to (row group, row within it) using cumulative row-group sizes
    ends = np.cumsum([pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])
    rg = np.searchsorted(ends, d.row.values, side='right')
    for g in np.unique(rg):
        tab = pf.read_row_group(int(g), columns=['id'] + (['body'] if kind == 'comment' else ['title', 'selftext']))
        start = int(ends[g - 1]) if g else 0
        dd = d[rg == g]
        sub = tab.take(pa.array(dd.row.values - start))
        assert sub['id'].to_pylist() == dd.aid.tolist(), 'archive row lookup mismatch'
        S.loc[dd.index, 'a_text'] = R._text(sub, kind).to_pylist()
T['texts_missing'] = int((S.q_text == '').sum() + (S.a_text == '').sum())

# ---- html ----
# ---- html: page header, then one section per band with one card per sampled pair ----
day = lambda s: time.strftime('%Y-%m-%d', time.gmtime(int(s)))
esc = lambda s: html.escape(s if len(s) <= 6000 else s[:6000] + '\n[... cut at 6,000 characters ...]')
H = ['<!doctype html><html><head><meta charset="utf-8"><title>Copy search: sample matches</title><style>',
     'body{font:15px/1.45 -apple-system,Helvetica,Arial,sans-serif;margin:24px auto;max-width:1200px;padding:0 16px;color:#222}',
     'h2{margin-top:40px;border-bottom:2px solid #444;padding-bottom:4px}.warn{background:#fde8e8;border:1px solid #d33;padding:10px 14px;border-radius:6px}',
     '.card{border:1px solid #bbb;border-radius:6px;margin:18px 0;padding:10px 14px}.meta{color:#444;font-size:13px;margin-bottom:8px}',
     '.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}.t{white-space:pre-wrap;word-break:break-word;background:#f6f6f6;padding:8px 10px;border-radius:4px}',
     '.lab{font-weight:600;font-size:13px;margin-bottom:4px}.same{color:#b00;font-weight:600}nav a{margin-right:14px}',
     '</style></head><body><h1>Copy search: sample of matched pairs</h1>',
     '<p class="warn"><b>Private.</b> This page contains real Reddit text. Keep it on this computer. Do not share, upload or publish it.</p>',
     f'<p>Each card is one accused text (left) and ONE earlier item from the archive that matches it (right), word 3-gram Jaccard at least 0.7. '
     f'Up to {a.n} texts per length group, picked at random; a text appears once on the page. The group is set by the SHORTER of the two texts, '
     f'not counting links. The {B[0]} version of the search includes every group below; each higher version drops the shorter groups. '
     f'No usernames are shown.</p><nav>' + ''.join(f'<a href="#b{b}">{names[b]}</a>' for b in range(len(B))) + '</nav>']
for b in range(len(B)):
    D = S[S.band == b]
    H.append(f'<h2 id="b{b}">Shorter text has {names[b]}</h2><p>{T["pairs_by_band"][names[b]]:,} matched pairs from {T["texts_by_band"][names[b]]:,} accused texts '
             f'are in this group in the full results; {len(D)} texts shown.</p>')
    for i, r in enumerate(D.itertuples(), 1):
        who = f'<span class="same">{r.who}</span>' if r.who.startswith('SAME') else r.who
        pre = 'before' if r.q_created_utc < r.q_acc_utc else 'AFTER'
        H.append(f'<div class="card"><div class="meta">#{i} &nbsp; Jaccard {r.jaccard:.2f} &nbsp;|&nbsp; this accused text has {int(ncop[r.ckey]):,} earlier copies in total '
                 f'&nbsp;|&nbsp; accused text posted {pre} the account was accused</div><div class="two">'
                 f'<div><div class="lab">Accused text ({r.ckey.split(":")[0]}, {day(r.q_created_utc)}, {int(r.q_vis_len)} characters)</div><div class="t">{esc(r.q_text)}</div></div>'
                 f'<div><div class="lab">Earlier copy ({r.akind} in r/{html.escape(str(r.a_subreddit))}, {day(r.a_created_utc)}, {int(r.a_vis_len)} characters, by {who})</div>'
                 f'<div class="t">{esc(r.a_text)}</div></div></div></div>')
H.append('</body></html>')
open(f'{a.out}/sample_matches_v1.html', 'w', encoding='utf-8').write('\n'.join(H))
T['who_counts'] = {k: int(v) for k, v in S.who.value_counts().items()}
T['html_bytes'] = os.path.getsize(f'{a.out}/sample_matches_v1.html'); T['total_s'] = round(time.time() - t0, 1)
json.dump(T, open(f'{a.out}/sample_summary_v1.json', 'w'), indent=1)
print(json.dumps(T, indent=1))
