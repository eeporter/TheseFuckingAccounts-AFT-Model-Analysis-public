"""rules2.py - the current copy-search rules for fastscan2.

The rule code (GAME, MDLINK, URL, clean_and_keep, to_epoch_s) is COPIED VERBATIM from fastscan's fast_full_v1.py
(md5 f2d6e34e7ab342440e9427649594a3f8, 2026-11-17, written by the "Find fastest practical copy search" thread), so both
versions apply the same definition. Only `minchar` became an argument. Do not edit the rules here; change them there
and copy again.

Rules (both sides): game-post notice stripped first; markdown links reduced to their visible words and bare URLs removed;
  >= minchar characters must remain and links may be at most half of the characters; then Spark cleanup + word 3-grams
  of the notice-stripped text. Dupe = exact Jaccard >= 0.7.
Archive side: CreatedAt + 8 h = UTC; rows on/after 30 Nov 2022 dropped before any text work.
A pair counts only if the archive item is strictly earlier than the query text.

How fastscan2 applies the same rules with less work (scan_group_fast):
  * date mask first (same as fastscan);
  * the game-notice regex can only match text containing "old reddit" (ASCII letters, any case). Those rows, and rows
    that are not valid UTF-8, are sent through fastscan's exact pipeline (clean_and_keep + norm). They are rare;
  * for every other row the notice-stripped text IS the raw text, so the C scanner works on the raw buffer. Rows whose
    raw length is under minchar cannot pass the length rule (links and trimming only shorten), so they are skipped;
  * the length + link rule itself (clean_and_keep) is then evaluated only for rows that produced a match.
  The result is the same set of pairs with the same columns; scan_group_ref is fastscan's pipeline, kept for checking.

rules3.py = rules2.py with one change, in scan_group_fast: the hits table of a row group is built in slices of at most
  `chunk` (default 2M) pairs. A single Arrow string column cannot hold more than 2 GB (32-bit offsets); a row group
  of a mass-copied text could produce enough pairs to pass that and crash with "Negative offsets". When a sink is
  given and the row group has more than `chunk` pairs, each slice is written by the worker (sink) instead of being
  returned, and the stats carry sunk_q (the query numbers matched) so the caller can still count matched queries.
  All other functions are identical to rules2.py.

Inputs: LSH_DIR/candidates_v1.parquet (query texts: kind, id, outer_key, created_utc, text) and
  LSH_DIR/sample_accounts_dates_v1.parquet (accusation date per account), read by Queries; archive parquet files
  (one row group at a time) read by sample_shingles / scan_group_fast / scan_group_ref.
Outputs: nothing on disk by itself; returns per-row-group stats dicts and pyarrow hits tables (one row per pair:
  query key, archive id/author/subreddit/time, visible lengths, shingle counts, intersection, Jaccard).
Used by: fast2_full_v3.py only (rerun of tasks 1, 3, 22 of the whole-pool run = array 43360110).
"""
import numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
import fastscan2 as fs

# CUTOFF: archive rows created on/after 30 Nov 2022 00:00 UTC are dropped (after the +8 h fix below). The same value
# is used by --query pre2022 to keep only query texts created before that date.
CUTOFF = int(np.datetime64('2022-11-30T00:00:00', 's').astype(np.int64))
# Archive file names containing these uuids belong to the newer batch; the full-run scripts leave them out so only the
# older batch (through 2024) is scanned.
NEWER = ('9c8fe3bb', '0f597d44')                      # uuids of the 2025-26 batch (comments, submissions)

# ---- verbatim from fast_full_v1.py ----
# GAME: the game-post notice ("content not supported on old reddit" plus an optional view-full-post link), removed first.
# MDLINK: a markdown link [visible words](target) -> its visible words. URL: a bare http(s):// or www. address.
GAME = r'(?i)this post contains content not supported on old reddit\.?\s*(\[click here to view the full post\]\([^)]*\))?'
MDLINK = r'\[([^\]]*)\]\([^)\s]*\)'
URL = r'(?i)(https?://|www\.)[^ \t\n\x0B\f\r]+'


def clean_and_keep(text, minchar):
    """text: string array. Returns (text with game notice removed, keep mask by length + link rule).

    Also returns n_vis, the visible length (characters after links/URLs are removed and spaces trimmed), as a third
    element. Rule: keep a row when n_vis >= minchar AND the characters removed as link/URL text are at most half of
    the notice-stripped text (2 * (n_raw - len(nolink)) <= n_raw). The returned text is only notice-stripped: links
    stay in it, so shingles are made from the notice-stripped text, as in the earlier Spark job."""
    t = pc.replace_substring_regex(pc.fill_null(text, ''), GAME, ' ')
    # nolink: markdown links reduced to their visible words, then bare URLs replaced by a space
    nolink = pc.replace_substring_regex(pc.replace_substring_regex(t, MDLINK, r'\1'), URL, ' ')
    n_all = pc.utf8_length(pc.utf8_trim_whitespace(t))
    n_vis = pc.utf8_length(pc.utf8_trim_whitespace(nolink))
    n_raw = pc.utf8_length(t)
    keep = pc.and_(pc.greater_equal(n_vis, minchar),
                   pc.less_equal(pc.multiply(pc.subtract(n_raw, pc.utf8_length(nolink)), 2), n_raw))
    return t, keep, n_vis


def to_epoch_s(col):
    """Timestamp or numeric column -> numpy int64 seconds since 1970. Timestamps are cast to seconds; numbers are
    taken as seconds and truncated. No time-zone shift is applied here (callers add +8 h for the archive)."""
    a = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    if pa.types.is_timestamp(a.type):
        return pc.cast(pc.cast(a, pa.timestamp('s')), pa.int64()).to_numpy(zero_copy_only=False)
    return np.asarray(pc.cast(a, pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64).astype(np.int64)
# ---- end verbatim ----


class Queries:
    """query side, same selection as fast_full_v1.py. Holds the cleaned query texts and their metadata (no index)."""
    def __init__(self, lsh_dir, query, minchar):
        """Load candidates + accusation dates, select query texts, apply the length + link rule.

        query: 'pre2022' = texts created before CUTOFF; 'preacc' = texts created before their account's accusation
        date; 'all' = every text. minchar: the visible-length threshold of clean_and_keep.
        Attributes: text (notice-stripped query texts), vis (visible length), created, acc (accusation time),
        ckey ('kind:id' key), n, accounts, total (rows before selection), limit, minchar."""
        c = pq.read_table(f'{lsh_dir}/candidates_v1.parquet', columns=['kind', 'id', 'outer_key', 'created_utc', 'text'])
        acc = pq.read_table(f'{lsh_dir}/sample_accounts_dates_v1.parquet', columns=['username_key', 'first_mention_adj_utc']).to_pandas()
        accmap = dict(zip(acc.username_key, acc.first_mention_adj_utc.astype(np.int64)))
        self.total = c.num_rows
        q_created = to_epoch_s(c['created_utc'])
        q_acc = np.array([accmap.get(k, -1) for k in c['outer_key'].to_pylist()], dtype=np.int64)
        assert (q_acc > 0).all(), 'missing accusation date'
        # query texts are selected first by date (per --query), then by the same length + link rule as the archive
        sel = q_created < CUTOFF if query == 'pre2022' else (q_created < q_acc if query == 'preacc' else np.ones(c.num_rows, bool))
        qt, qkeep, qvis = clean_and_keep(c['text'], minchar)
        sel &= qkeep.to_numpy(zero_copy_only=False)
        c = c.filter(pa.array(sel))
        self.text = qt.filter(pa.array(sel)); self.vis = qvis.filter(pa.array(sel)).to_numpy(zero_copy_only=False)
        self.created = q_created[sel]; self.acc = q_acc[sel]
        self.n = c.num_rows; self.accounts = len(set(c['outer_key'].to_pylist()))
        self.ckey = np.array([f'{k}:{i}' for k, i in zip(c['kind'].to_pylist(), c['id'].to_pylist())], dtype=object)
        # Q.limit: archive rows are kept only if created < limit. It combines the CUTOFF and the earlier-only rule: an
        # archive row created at/after the newest query text cannot be strictly earlier than any query text.
        self.limit = min(CUTOFF, int(self.created.max()))   # no archive row at/after this can be earlier than any query
        self.minchar = minchar


def _kind(path):
    """'comment' for a file of the comments archive, else 'post' (submissions)."""
    return 'comment' if 'comments' in path else 'post'


def _text(tab, kind):
    """Archive text of each row: the comment body, or for a post the title and selftext joined by a newline."""
    return tab['body'] if kind == 'comment' else pc.binary_join_element_wise(
        pc.fill_null(tab['title'], ''), pc.fill_null(tab['selftext'], ''), '\n')


def _hits_table(Q, qlen, kind, sub, created_r, q, inter, m, avis_r):
    """Build the output pairs table, one row per (query text, archive row) pair.

    sub: archive rows of the pairs (columns id, author, subreddit), in pair order; created_r: their UTC times;
    q: query numbers; inter: shingle-set intersection size; m: archive shingle count; avis_r: archive visible length.
    jaccard = inter / (|query shingles| + |archive shingles| - inter)."""
    j = inter / (qlen[q] + m - inter)
    return pa.table({'ckey': pa.array(Q.ckey[q].astype(str), type=pa.string()), 'aid': sub['id'], 'akind': pa.array([kind] * len(q), type=pa.string()),
                     'a_author': sub['author'], 'a_subreddit': sub['subreddit'],
                     'a_created_utc': pa.array(created_r, type=pa.int64()), 'q_created_utc': pa.array(Q.created[q], type=pa.int64()),
                     'q_acc_utc': pa.array(Q.acc[q], type=pa.int64()), 'q_vis_len': pa.array(Q.vis[q]), 'a_vis_len': pa.array(avis_r),
                     'inter': pa.array(inter), 'a_nsh': pa.array(m), 'q_nsh': pa.array(qlen[q]), 'jaccard': pa.array(j)})


def sample_shingles(path, rg, minchar):
    """shingles of one archive row group, used only to order the index (any order gives the same result).

    The full-run scripts call this with row group 0 of the task's first file. Shingles are counted with repeats so
    the Index can rank shingles by how often they occur in this archive sample (rarest first in the prefix filter).
    No date or link rule is applied here; that is fine because this only affects speed, not which pairs are found."""
    kind = _kind(path)
    tab = pq.ParquetFile(path).read_row_group(rg, columns=['body'] if kind == 'comment' else ['title', 'selftext'])
    return fs.raw_shingles(_text(tab, kind), minchar)


def scan_group_fast(idx, Q, path, rg, sink=None, chunk=2_000_000):
    """fastscan2 path for one row group. Returns (stats dict, hits table).
    rules3 change (2026-11-18), output unchanged: the hits table is built in slices of at most `chunk` pairs, so no
    string column passes Arrow's 2 GB limit, and when `sink` is given and there are more than `chunk` pairs, each slice
    is handed to sink(table) (the worker writes it to disk) instead of being returned; hits is then None."""
    kind = _kind(path); f = pq.ParquetFile(path)
    tab = f.read_row_group(rg, columns=['CreatedAt'] + (['body'] if kind == 'comment' else ['title', 'selftext']))
    n = tab.num_rows
    # +8 h fix: archive CreatedAt is stored 8 h behind UTC, so it is shifted to UTC before any comparison
    created = to_epoch_s(tab['CreatedAt']) + 8 * 3600     # archive CreatedAt is Pacific (8 h behind UTC)
    m_date = created < Q.limit                            # CUTOFF and newest-query limit (see Queries.limit)
    text = _text(tab, kind)
    idx.stats[:] = 0
    # One C pass over the raw Arrow buffer, only rows with m_date set: rows whose raw length is under minchar are
    # skipped before any cleanup (removing links/URLs and trimming can only shorten a text, so such a row can never
    # reach minchar visible characters). Remaining rows are cleaned, shingled, Bloom-filtered and probed against the
    # inverted index (prefix filter), and candidates are verified by exact Jaccard >= 0.7. Rows containing
    # "old reddit" or invalid UTF-8 are not scanned here; their positions come back in fl.
    # stats afterwards: [0] rows passing the raw length test, [1] rows verified, [2] pairs verified.
    r, q, inter, m, fl = idx.scan_raw_flagged(text, keep=m_date, minchar=Q.minchar, flag_game=True)
    r_all, q_all, i_all, m_all, vis_all = [], [], [], [], []
    if len(r):                                            # length + link rule, only for rows that matched
        # these rows hold no game notice, so the full clean_and_keep on the raw text gives the exact keep decision
        ur, inv = np.unique(r, return_inverse=True)
        _, keep, vis = clean_and_keep(text.take(pa.array(ur)), Q.minchar)
        keep = keep.to_numpy(zero_copy_only=False)[inv]; vis = vis.to_numpy(zero_copy_only=False)[inv]
        r_all.append(r[keep]); q_all.append(q[keep]); i_all.append(inter[keep]); m_all.append(m[keep]); vis_all.append(vis[keep])
    n_flag = len(fl)
    if n_flag:                                            # possible game notice / invalid UTF-8: fastscan's exact pipeline
        # notice stripped + length/link rule, then fastscan's pyarrow cleanup (norm) and the same index (scan_norm)
        t2, keep2, vis2 = clean_and_keep(text.take(pa.array(fl)), Q.minchar)
        k2 = keep2.to_numpy(zero_copy_only=False)
        if k2.any():
            rows2 = fl[k2]; vis2 = vis2.filter(pa.array(k2)).to_numpy(zero_copy_only=False)
            r2, q2, i2, m2 = idx.scan_norm(fs.norm(t2.filter(pa.array(k2))))
            r_all.append(rows2[r2]); q_all.append(q2); i_all.append(i2); m_all.append(m2); vis_all.append(vis2[r2])
    st = [int(x) for x in idx.stats]
    if r_all:
        r = np.concatenate(r_all); q = np.concatenate(q_all); inter = np.concatenate(i_all); m = np.concatenate(m_all); vis = np.concatenate(vis_all)
        # earlier-only rule: archive item strictly earlier (UTC) than the query text. A query text found in the archive
        # has the same time, so it is not strictly earlier: this also removes self-matches.
        early = created[r] < Q.created[q]                 # earlier-only rule (also removes self-matches)
        r, q, inter, m, vis = r[early], q[early], inter[early], m[early], vis[early]
    else:
        r = np.empty(0, np.int64); q = inter = m = np.empty(0, np.int32); vis = np.empty(0, np.int64)
    if len(r):                                            # ids etc. are read only when there is something to write
        sub_all = f.read_row_group(rg, columns=['id', 'author', 'subreddit'])
        # chunked hit tables: each slice has at most `chunk` pairs, so no string column (ckey, author, ...) of one
        # slice can pass Arrow's 2 GB limit. Big row groups (> chunk pairs, sink given) go to disk slice by slice;
        # otherwise the slices are concatenated (concat_tables keeps them as separate chunks, no copy into one array).
        pieces = []
        for a in range(0, len(r), chunk):
            b = slice(a, a + chunk); rb = r[b]
            hb = _hits_table(Q, idx.qlen, kind, sub_all.take(pa.array(rb)), created[rb], q[b], inter[b], m[b], vis[b])
            if sink is not None and len(r) > chunk: sink(hb)
            else: pieces.append(hb)
        hits = pa.concat_tables(pieces) if pieces else None
        # query numbers of pairs written by the sink; the caller adds them to its count of matched queries
        sunk_q = np.unique(q) if (sink is not None and len(r) > chunk) else None
    else:
        hits = None; sunk_q = None
    return dict(file=path.rsplit('/', 1)[-1], rg=rg, kind=kind, rows=n, rows_date=int(m_date.sum()), rows_cleaned=st[0],
                rows_flagged=n_flag, rows_verified=st[1], verified=st[2], pairs=len(r), sunk_q=sunk_q), hits


def scan_group_ref(fs1, idx1, Q, path, rg):
    """fastscan's pipeline for one row group (the body of work() in fast_full_v1.py), using fastscan's own module fs1
    and index idx1. For equality checks and timing only.

    Order of work (reference pipeline): +8 h fix and date mask, then clean_and_keep on every remaining row, then
    fastscan's norm() cleanup and idx1.scan (exact prefix-filter search, no Bloom filter, no early skip), then the
    earlier-only rule. Used by the --selfcheck of the full-run scripts and by rules_check_v1.py."""
    kind = _kind(path)
    cols = ['id', 'author', 'subreddit', 'CreatedAt'] + (['body'] if kind == 'comment' else ['title', 'selftext'])
    tab = pq.ParquetFile(path).read_row_group(rg, columns=cols)
    n = tab.num_rows
    created = to_epoch_s(tab['CreatedAt']) + 8 * 3600     # +8 h fix (archive time -> UTC)
    m_date = created < Q.limit                            # CUTOFF / newest-query limit
    tab = tab.filter(pa.array(m_date)); created = created[m_date]
    n_date = tab.num_rows
    text, keep, avis = clean_and_keep(_text(tab, kind), Q.minchar)
    keep = keep.to_numpy(zero_copy_only=False)
    tab = tab.filter(pa.array(keep)); text = text.filter(pa.array(keep)); created = created[keep]
    avis = avis.filter(pa.array(keep)).to_numpy(zero_copy_only=False)
    idx1.stats[:] = 0
    r, q, inter, m, j = idx1.scan(fs1.norm(text))
    early = created[r] < Q.created[q]                     # earlier-only rule (also removes self-matches)
    r, q, inter, m = r[early], q[early], inter[early], m[early]
    hits = _hits_table(Q, idx1.qlen, kind, tab.take(pa.array(r)), created[r], q, inter, m, avis[r]) if len(r) else None
    return dict(file=path.rsplit('/', 1)[-1], rg=rg, kind=kind, rows=n, rows_date=n_date, rows_len=int(keep.sum()),
                verified=int(idx1.stats[1]), pairs=len(r)), hits
