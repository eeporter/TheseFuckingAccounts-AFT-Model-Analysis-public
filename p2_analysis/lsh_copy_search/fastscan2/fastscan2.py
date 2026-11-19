"""fastscan2.py - Python side of the faster exact copy search (see fastscan2.c).

Same rules and same results as fastscan (lsh_copy_search/fastscan/): Spark cleanup, word 3-grams hashed with
xxhash64 seed 42, exact prefix-filter index over the query texts, exact Jaccard >= t verification.
Difference: archive texts are cleaned, length-filtered and probed in one C pass over the raw Arrow buffer.

The per-code-point tables that drive the C cleanup are computed here FROM PYARROW ITSELF (utf8_lower and the same
RE2 character classes fastscan.norm uses), so the C cleanup follows whatever pyarrow does in this environment.
norm() below is fastscan's cleanup, kept for the query side, for invalid-UTF-8 rows and for testing.
No text is ever written out; results are row numbers, query numbers and counts.

Method (exact, not approximate): a pair (archive text, query text) counts when the Jaccard similarity of their SETS
of word-3-gram shingles is >= t = tnum/tden (7/10 in every run). Exact prefix-filter set-similarity join
(Chaudhuri, Ganti & Kaushik, ICDE 2006; length filter as in Bayardo, Ma & Srikant, WWW 2007; Xiao et al., WWW 2008):
each query is put in an inverted index under only its first n - ceil(t*n) + 1 shingles in one fixed global order;
any text with Jaccard >= t must share at least one of those, so only rows hitting the index need exact checking.
In fastscan2.c a one-hash Bloom filter sits in front of the inverted index.

Inputs : query texts (cleaned here with norm()), raw archive texts as pyarrow string / large_string arrays, and
         optionally the raw shingles of an archive sample (raw_shingles) to choose the index order.
Outputs: numpy arrays of (archive row, query number, intersection size, archive set size[, Jaccard]) and
         flagged-row positions; counters in Index.stats.
Needs  : the compiled C library libfastscan2.so next to this file, or at $FASTSCAN2_LIB.
Used by: pilot_fast2_v1.py, rules_check_v1.py, rules2.py / rules3.py, fast2_full_v1/v2/v3.py, test_clean_v1.py.
Jobs (Hyak Slurm): pilot 43324646 and 43328586; rules check 43328587; first full run array 43329112; 4-vs-12 worker
A/B 43332090; second full run array 43332591; whole-pool run array 43340006; rerun of tasks 1, 3, 22 array 43360110.
"""
import ctypes, os
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

HERE = os.path.dirname(os.path.abspath(__file__))
# Load the C library and declare every exported function's argument and return types (ctypes would otherwise
# assume int arguments/returns and truncate 64-bit values and pointers). P = any pointer (passed as an int address
# or None for NULL), I64 = int64.
_lib = ctypes.CDLL(os.environ.get('FASTSCAN2_LIB', os.path.join(HERE, 'libfastscan2.so')))
P, I64 = ctypes.c_void_p, ctypes.c_int64
_lib.init_tables.restype = ctypes.c_int; _lib.init_tables.argtypes = [P, P]
_lib.init_index.restype = ctypes.c_int
_lib.init_index.argtypes = [P, P, I64, P, P, P, I64, ctypes.c_int, ctypes.c_int]
_lib.shingle_batch.restype = I64; _lib.shingle_batch.argtypes = [P, P, I64, P, P, P]
_lib.clean_batch.restype = I64; _lib.clean_batch.argtypes = [P, P, I64, P, P, P]
_lib.scan_raw.restype = I64
_lib.scan_raw.argtypes = [P, P, P, I64, P, I64, P, I64, ctypes.c_int, P, P, P, P, P, I64, P]
_lib.shingle_raw.restype = I64; _lib.shingle_raw.argtypes = [P, P, P, I64, P, I64, I64, P, I64]
_lib.scan_norm.restype = I64; _lib.scan_norm.argtypes = [P, P, I64, P, P, P, P, I64, P]

# identical to fastscan.py
# WS: the whitespace class of the cleanup (Java's Unicode \s set, written out for RE2). WORD: word characters
# (Java's Unicode \w set, written out: letters, letter numbers, marks, decimal digits, connector punctuation,
# zero-width joiners, and the circled / squared Latin letters that Java counts as alphabetic). The apostrophe is
# added to WORD wherever WORD is used, so it is kept inside words.
WS = r'\t\n\x{0B}\f\r \x{85}\x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}'
WORD = (r'\p{L}\p{Nl}\p{Mn}\p{Mc}\p{Me}\p{Nd}\p{Pc}\x{200C}\x{200D}\x{345}\x{24B6}-\x{24E9}'
        r'\x{1F130}-\x{1F149}\x{1F150}-\x{1F169}\x{1F170}-\x{1F189}')


def norm(arr):
    """fastscan's cleanup (pyarrow regex passes). Used for queries, fallback rows and tests.

    Steps (each one a whole-column pyarrow pass): lower case; URLs (http:// or https:// up to ASCII whitespace)
    -> space; every character that is not a word character, whitespace or apostrophe -> space; runs of whitespace
    -> one space; trim spaces. Nulls become ''. Returns a large_string array (the C side reads int64 offsets).
    fastscan2.c's clean_one() does the same in one pass and must give identical output."""
    t = pc.utf8_lower(arr)
    t = pc.replace_substring_regex(t, r'https?://[^ \t\n\x0B\f\r]+', ' ')
    t = pc.replace_substring_regex(t, "[^" + WORD + WS + "']", ' ')
    t = pc.replace_substring_regex(t, "[" + WS + "]+", ' ')
    t = pc.utf8_trim(t, ' ')
    return pc.cast(pc.fill_null(t, ''), pa.large_string())


_TABLES = None


def build_tables():
    """lower-case code point and word/separator class for every Unicode code point, taken from pyarrow.

    Builds LOW[cp] (lower-case code point) and CLS[cp] (1 = word character or apostrophe, 0 = separator) for all
    0x110000 code points by running pyarrow's own utf8_lower and the same RE2 classes norm() uses over one string
    per code point, so the single-pass C cleanup follows exactly what the pyarrow regex cleanup does in this
    environment. Checks first that lower-casing maps every code point to exactly one code point (else the table
    approach cannot reproduce utf8_lower) and that no word character is also whitespace. Passes the tables to C
    (init_tables) and caches them in _TABLES so the arrays stay alive for the C pointers. Returns (LOW, CLS)."""
    global _TABLES
    if _TABLES is not None:
        return _TABLES
    # every code point except the UTF-16 surrogates (which cannot appear in valid UTF-8); those keep LOW = identity
    # and CLS = 0 below
    cps = np.array([c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF], dtype=np.uint32)
    # one single-character string per code point. The branch before "if False" is never run (dead code); the
    # list-of-chr branch after "else" is the one used.
    chars = pa.array(cps.astype('<u4').tobytes().decode('utf-32-le'), type=pa.string()) if False else \
        pa.array([chr(c) for c in cps.tolist()], type=pa.string())
    low = pc.utf8_lower(chars)
    if pc.min(pc.utf8_length(low)).as_py() != 1 or pc.max(pc.utf8_length(low)).as_py() != 1:
        raise RuntimeError('utf8_lower is not one code point to one code point here; fastscan2 tables do not apply')
    # each lower-cased string is one code point (checked above), so joining them and decoding as UTF-32 gives
    # exactly one lower-case code point per entry of cps
    LOW = np.arange(0x110000, dtype=np.uint32)
    LOW[cps] = np.frombuffer(''.join(low.to_pylist()).encode('utf-32-le'), dtype='<u4')
    CLS = np.zeros(0x110000, dtype=np.uint8)
    CLS[cps] = pc.match_substring_regex(chars, "^[" + WORD + "']$").to_numpy(zero_copy_only=False).astype(np.uint8)
    # a word character must not also be whitespace under the cleanup's whitespace class
    ws = pc.match_substring_regex(chars, "^[" + WS + "]$").to_numpy(zero_copy_only=False)
    if (CLS[cps][ws] != 0).any():
        raise RuntimeError('word and whitespace classes overlap')
    if _lib.init_tables(LOW.ctypes.data, CLS.ctypes.data) != 0:
        raise RuntimeError('unexpected ASCII lower-case mapping')
    _TABLES = (LOW, CLS)
    return _TABLES


def _large(a):
    """Arrow string array -> (large_string array, int64 offsets as numpy, address of the value buffer).

    Combines chunks and casts to large_string. If the array is a slice (offset != 0) or has nulls, it is rebuilt
    from Python objects so the offsets start at slot 0 (a rebuilt null slot is empty). The C functions that take
    these buffers (shingle_batch, clean_batch, scan_norm) ignore the validity bitmap."""
    a = a.combine_chunks() if isinstance(a, pa.ChunkedArray) else a
    a = pc.cast(a, pa.large_string())
    if a.offset != 0 or a.null_count:
        a = pa.array(a.to_pylist(), type=pa.large_string())
    off = np.frombuffer(a.buffers()[1], dtype=np.int64, count=len(a) + 1)
    data = a.buffers()[2]
    return a, off, (data.address if data is not None else 0)


def shingles(norm_arr):
    """Shingle SETS of already-cleaned texts (norm() output), via C shingle_batch.

    Returns (out, ptr, cnt): text i's sorted, de-duplicated xxhash64 seed-42 word-3-gram hashes are
    out[ptr[i]:ptr[i+1]]; cnt[i] = set size, or -1 for a text with fewer than 3 words (it has no shingles)."""
    a, off, dptr = _large(norm_arr)
    n = len(a)
    # room for all shingles: a text of L bytes has at most L//2 + 1 words, so fewer shingles than that
    room = int(((off[1:] - off[:-1]) // 2 + 1).sum()) + 1
    out = np.empty(room, np.uint64); ptr = np.empty(n + 1, np.int64); cnt = np.empty(n, np.int64)
    _lib.shingle_batch(dptr, off.ctypes.data, n, out.ctypes.data, ptr.ctypes.data, cnt.ctypes.data)
    return out[:ptr[-1]].copy(), ptr, cnt


def clean_c(arr):
    """C cleanup of raw texts -> (list of cleaned strings or None for invalid UTF-8). For tests only.

    The cleaned strings are returned as bytes (UTF-8). Used by test_clean_v1.py to compare against norm()."""
    build_tables()
    a, off, dptr = _large(arr)
    n = len(a)
    out = np.empty(2 * int(off[-1] - off[0]) + 16, np.uint8); oo = np.empty(n + 1, np.int64); bad = np.zeros(n, np.uint8)
    _lib.clean_batch(dptr, off.ctypes.data, n, out.ctypes.data, oo.ctypes.data, bad.ctypes.data)
    b = out.tobytes()
    return [None if bad[i] else b[oo[i]:oo[i + 1]] for i in range(n)]


def raw_shingles(arr, minchar=140):
    """all shingle hashes (repeats kept) of the RAW texts in arr that pass the length rule. For archive samples.

    The full runs pass row group 0 of the task's first archive file (rules2/rules3 sample_shingles). The result is
    only used to ORDER the prefix-filter index (Index(sample_shingles=...)), so it affects speed, not results.
    Works chunk by chunk directly on the Arrow buffers (no copy); rows with invalid UTF-8 are skipped by the C code.
    Returns one uint64 numpy array."""
    build_tables()
    out = []
    for a in (arr.chunks if isinstance(arr, pa.ChunkedArray) else [arr]):
        n = len(a)
        if n == 0: continue
        bufs = a.buffers(); wide = pa.types.is_large_string(a.type)
        # offsets of this chunk's slots (string -> int32, large_string -> int64), starting at the chunk's own offset
        offs = np.frombuffer(bufs[1], dtype=np.int64 if wide else np.int32, count=a.offset + n + 1)[a.offset:]
        # a row of L bytes has fewer than L shingles, so total bytes + 8 is always enough room
        cap = int(offs[-1] - offs[0]) + 8
        o = np.empty(cap, np.uint64)
        k = _lib.shingle_raw(bufs[2].address if bufs[2] is not None else 0, None if wide else offs.ctypes.data,
                             offs.ctypes.data if wide else None, n,
                             bufs[0].address if (bufs[0] is not None and a.null_count) else None, a.offset, minchar,
                             o.ctypes.data, cap)
        if k < 0: raise RuntimeError('shingle_raw buffer too small')
        out.append(o[:k].copy())
    return np.concatenate(out) if out else np.empty(0, np.uint64)


class Index:
    """Prefix-filter index over query texts (already cleaned with norm()). t = tnum/tden.
    sample_shingles = None: same index as fastscan (shingles ordered by how rare they are among the queries).
    sample_shingles = raw_shingles(<a sample of archive rows>): shingles are ordered by how rare they are in that
    archive sample first, so each query is indexed under shingles that few archive rows contain. The prefix filter
    is exact under ANY fixed order, so this changes speed only, never the result.

    Prefix filter: a query with n distinct shingles is indexed only under its first n - ceil(t*n) + 1 shingles in
    the chosen global order. A text with Jaccard >= t shares at least ceil(t*n) of the query's shingles, so it must
    contain at least one of those prefix shingles. The C side holds the inverted index (prefix shingle -> list of
    query ids) in an open-addressing hash table behind a Bloom filter, plus every query's full shingle set for the
    exact check. The C index is a single global per process: building a second Index replaces the first.

    Attributes: ok (query has >= 1 shingle), qsh / qptr / qlen (query shingle sets), nq, n_prefix (postings),
    n_keys (distinct prefix shingles), n_shingles, stats (counters summed over scans)."""
    def __init__(self, qnorm, tnum=7, tden=10, sample_shingles=None):
        """Build the index for cleaned query texts qnorm; t = tnum/tden; see the class docstring for sample_shingles."""
        build_tables()
        sh, ptr, cnt = shingles(qnorm)
        self.ok = cnt > 0
        self.qsh, self.qptr, self.qlen = sh, ptr, np.diff(ptr)
        self.nq = len(cnt)
        # keys = distinct shingle hashes over all queries; inv maps each entry of sh to its key; df = number of
        # queries containing the key (each query's shingles are already a set)
        keys, inv, df = np.unique(sh, return_inverse=True, return_counts=True)
        # rank = keys in the global order (np.lexsort sorts by its LAST key first): fewest queries first, ties by
        # hash value
        if sample_shingles is None:
            rank = np.lexsort((keys, df))
        else:
            # af = how often each query shingle occurs in the archive sample (repeats counted); sample shingles that
            # no query has are ignored
            pos = np.searchsorted(keys, sample_shingles); pos[pos >= len(keys)] = 0
            hit = keys[pos] == sample_shingles
            af = np.bincount(pos[hit], minlength=len(keys))
            rank = np.lexsort((keys, df, af))  # rarest in the archive sample first, then rarest among queries
        # order_of_key[k] = position of key k in the global order; o = that position for every entry of sh
        order_of_key = np.empty(len(keys), np.int64); order_of_key[rank] = np.arange(len(keys))
        qid = np.repeat(np.arange(self.nq), self.qlen)
        o = order_of_key[inv]
        # srt = indices into sh sorted by query, then by global order within each query. Each query's block in srt
        # covers the same index range ptr[q]..ptr[q+1] as in sh, so pos_in_q (position within the query, computed
        # from plain indices) is also the rank-position within the query for srt's entries
        srt = np.lexsort((o, qid))
        n = self.qlen
        # alpha = ceil(t*n) in integers; prefix length plen = n - alpha + 1 (0 for queries without shingles)
        alpha = (tnum * n + tden - 1) // tden
        plen = np.where(n > 0, n - alpha + 1, 0)
        pos_in_q = np.arange(len(sh)) - np.repeat(ptr[:-1], n)
        # take = the first plen shingles (in the global order) of every query: the prefix-filter entries
        take = srt[pos_in_q < np.repeat(plen, n)]
        pk, pq_ = sh[take], qid[take]
        # group the (shingle, query) entries by shingle (queries ascending inside a group): ukeys = distinct prefix
        # shingles, postings of ukeys[i] = pq_[kptr[i]:kptr[i+1]]. This is the inverted index handed to C.
        s2 = np.lexsort((pq_, pk)); pk, pq_ = pk[s2], pq_[s2].astype(np.int32)
        ukeys, start = np.unique(pk, return_index=True)
        kptr = np.append(start, len(pk)).astype(np.int64)
        # C keeps pointers to pq_ (postings), qsh and qptr, so these arrays must stay alive as long as the index
        self._keep = (ukeys, kptr, pq_)
        self.n_prefix = len(pk); self.n_keys = len(ukeys); self.n_shingles = len(sh)
        _lib.init_index(ukeys.ctypes.data, kptr.ctypes.data, len(ukeys), pq_.ctypes.data,
                        self.qsh.ctypes.data, self.qptr.ctypes.data, self.nq, tnum, tden)
        self.stats = np.zeros(5, np.int64)   # rows passing length rule, rows verified, pairs verified, invalid rows, game-flagged rows

    def _scan_chunk(self, a, keep, minchar, flag_game):
        """one Arrow chunk. Returns hits and the positions of flagged rows (invalid UTF-8 or possible game notice).

        Passes the chunk's raw Arrow buffers (values, offsets of either width, validity bitmap only if there are
        nulls) straight to C scan_raw; no copy, no pyarrow cleanup. Returns (rows, queries, intersections, archive
        set sizes, flagged row positions), all relative to this chunk. Flag values in C: 1 = invalid UTF-8,
        2 = contains "old reddit" (possible game notice); only the positions are returned."""
        n = len(a)
        z = np.empty(0, np.int64)
        if n == 0:
            return z, z.astype(np.int32), z.astype(np.int32), z.astype(np.int32), z
        bufs = a.buffers()
        wide = pa.types.is_large_string(a.type)
        offs = np.frombuffer(bufs[1], dtype=np.int64 if wide else np.int32, count=a.offset + n + 1)[a.offset:]
        dptr = bufs[2].address if bufs[2] is not None else 0
        vptr = bufs[0].address if (bufs[0] is not None and a.null_count) else None
        kp = None if keep is None else np.ascontiguousarray(keep, dtype=np.uint8)
        flag = np.zeros(n, np.uint8)
        # hit buffer: start with room for 4096 hits. If C finds more, it returns -(hits needed); grow to that (+16)
        # and rerun the whole chunk with fresh buffers, zeroed stats and cleared flags, so nothing is counted twice.
        cap = 4096
        while True:
            hr = np.empty(cap, np.int64); hq = np.empty(cap, np.int32); hi = np.empty(cap, np.int32); hm = np.empty(cap, np.int32)
            st = np.zeros(5, np.int64); flag[:] = 0
            r = _lib.scan_raw(dptr, None if wide else offs.ctypes.data, offs.ctypes.data if wide else None, n,
                              vptr, a.offset, None if kp is None else kp.ctypes.data, minchar, int(flag_game), flag.ctypes.data,
                              hr.ctypes.data, hq.ctypes.data, hi.ctypes.data, hm.ctypes.data, cap, st.ctypes.data)
            if r >= 0: break
            cap = -r + 16
        self.stats += st
        return hr[:r], hq[:r], hi[:r], hm[:r], (np.flatnonzero(flag) if (st[3] or st[4]) else z)

    def scan_norm(self, norm_arr):
        """already-cleaned texts (norm() output) -> (row, query, inter, m). Slow-path companion of scan_raw.

        Used for flagged rows after the caller's own cleanup. No length rule here. Same grow-and-retry hit buffer
        as _scan_chunk (starting at max(1024, n)). Rows are positions in norm_arr."""
        b, off, dp = _large(norm_arr)
        n = len(b)
        z = np.empty(0, np.int64)
        if n == 0:
            return z, z.astype(np.int32), z.astype(np.int32), z.astype(np.int32)
        c2 = max(1024, n)
        while True:
            r2 = np.empty(c2, np.int64); q2 = np.empty(c2, np.int32); i2 = np.empty(c2, np.int32); m2 = np.empty(c2, np.int32)
            s2 = np.zeros(5, np.int64)
            k = _lib.scan_norm(dp, off.ctypes.data, n, r2.ctypes.data, q2.ctypes.data, i2.ctypes.data, m2.ctypes.data, c2, s2.ctypes.data)
            if k >= 0: break
            c2 = -k + 16
        s2[3] = 0; self.stats += s2
        return r2[:k], q2[:k], i2[:k], m2[:k]

    def scan_raw_flagged(self, arr, keep=None, minchar=140, flag_game=False):
        """Like scan_raw but does NOT process flagged rows: returns (row, query, inter, m, flagged_rows).
        The caller must handle flagged_rows itself (used by the rules pipeline).

        arr: raw texts (Array or ChunkedArray); keep: optional bool mask over all rows (rules: the date limit);
        flag_game: also flag rows containing "old reddit" (rules2/rules3 pass True). Row numbers are converted
        from per-chunk to positions in arr by adding each chunk's start (base)."""
        chunks = arr.chunks if isinstance(arr, pa.ChunkedArray) else [arr]
        out = [[], [], [], [], []]; base = 0
        for a in chunks:
            k = None if keep is None else keep[base:base + len(a)]
            hr, hq, hi, hm, fl = self._scan_chunk(a, k, minchar, flag_game)
            out[0].append(hr + base); out[1].append(hq); out[2].append(hi); out[3].append(hm); out[4].append(fl + base)
            base += len(a)
        return tuple(np.concatenate(x) if x else np.empty(0, np.int64) for x in out)

    def scan_raw(self, arr, keep=None, minchar=140):
        """arr: RAW archive texts (pyarrow string/large_string Array or ChunkedArray); keep: optional bool mask.
        Pilot rules only: the length rule (>= minchar characters after trimming spaces) is applied inside.
        Returns (row, query, inter, m, jaccard): row = position in arr.

        Used by the pilot (pilot_fast2_v1.py); the rules pipeline uses scan_raw_flagged instead. Here only invalid
        UTF-8 rows are flagged (flag_game False); they are re-checked with the pyarrow length rule and cleaned with
        norm() and scanned by scan_norm. Hits from those rows are appended at the end (not in row order).
        jaccard = inter / (|query set| + m - inter), as a float."""
        hr, hq, hi, hm, fl = self.scan_raw_flagged(arr, keep, minchar, False)
        if len(fl):   # invalid UTF-8 rows: old path (pyarrow cleanup), same length rule
            sub = (arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr).take(pa.array(fl))
            ok = pc.greater_equal(pc.utf8_length(pc.utf8_trim(pc.fill_null(sub, ''), ' ')), minchar).to_numpy(zero_copy_only=False)
            fl = fl[ok]
            if len(fl):
                r2, q2, i2, m2 = self.scan_norm(norm(sub.filter(pa.array(ok))))
                hr = np.concatenate([hr, fl[r2]]); hq = np.concatenate([hq, q2]); hi = np.concatenate([hi, i2]); hm = np.concatenate([hm, m2])
        na = self.qlen[hq]
        return hr, hq, hi, hm, hi / (na + hm - hi)
