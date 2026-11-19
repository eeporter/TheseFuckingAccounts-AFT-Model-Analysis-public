"""fastscan.py - Python side of the exact copy search (prefix-filter index; replaces Spark MinHash LSH).

norm()   : the Spark cleanup from lsh_pilot_v1.py, done with pyarrow.compute (vectorised C++):
           lower -> drop https?://\\S+ -> (?U)[^\\w\\s'] to space -> (?U)\\s+ to one space -> trim spaces.
           Java's (?U) classes are spelled out for RE2 (Unicode letters, marks, digits, connector punct, etc.).
Index    : built from the query texts only; see fastscan.c for the prefix-filter rule.
No text is ever written out; results are ids + counts.

Added notes (comment-only pass):
  Role: Python half of the ORIGINAL exact scanner (fastscan/, "v1"); the C half is fastscan.c, loaded here with
    ctypes from libfastscan.so (or from the path in $FASTSCAN_LIB). The run_*.sh scripts compile it with gcc first.
  Contents: norm() text cleanup; shingles() word-3-gram hashes per row; class Index (prefix-filter inverted index
    over the query texts, plus scan() of archive texts); jaccard_bruteforce() a slow all-pairs checker.
  Inputs: pyarrow string arrays (query texts, archive texts). Outputs: numpy arrays of (archive row, query id,
    intersection size, archive shingle count, Jaccard). Exact method: prefix filter + length filter + exact
    intersection (no LSH, no sampling). There is NO Bloom filter in this version (it came in fastscan2).
  Used by: toy_fast_v1.py (jobs 43301873 failed - container could not see /gscratch - and 43301955 PASS, 33 pairs),
    pilot_fast_v1.py (job 43321861, all 427 Spark matches, no extras) and fast_full_v1.py (no recorded job).
    fastscan2/ref/fastscan.py is a copy of this file used by fast2_full --selfcheck.
"""
import ctypes, os
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

# Load the C library and declare the C signatures (all array arguments are passed as raw addresses, c_void_p).
HERE = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.environ.get('FASTSCAN_LIB', os.path.join(HERE, 'libfastscan.so')))
P = ctypes.c_void_p
_lib.shingle_batch.restype = ctypes.c_int64
_lib.shingle_batch.argtypes = [P, P, ctypes.c_int64, P, P, P]
_lib.init_index.restype = ctypes.c_int
_lib.init_index.argtypes = [P, P, ctypes.c_int64, P, P, P, ctypes.c_int64, ctypes.c_int, ctypes.c_int]
_lib.scan_batch.restype = ctypes.c_int64
_lib.scan_batch.argtypes = [P, P, ctypes.c_int64, P, P, P, P, P, ctypes.c_int64, P]
_lib.xxh64.restype = ctypes.c_uint64
_lib.xxh64.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64]

# The Spark job ran Java regexes with the (?U) flag (UNICODE_CHARACTER_CLASS); pyarrow uses RE2, which has no such
# flag, so the two classes are spelled out by hand:
#   WS   = all Unicode White_Space code points (tab..CR, space, NEL, NBSP, Ogham space, U+2000-200A, line/para
#          separators, narrow NBSP, medium math space, ideographic space).
#   WORD = letters (\p{L}) and letter numbers (\p{Nl}) plus the Other_Alphabetic symbols listed (U+0345, circled
#          and squared/negative-circled Latin letters), marks Mn/Mc/Me, decimal digits Nd, connector punctuation Pc,
#          and the two Join_Control characters U+200C/U+200D.
# Java (?U)\s = Unicode White_Space; Java (?U)\w = Alphabetic + Mn + Me + Mc + Nd + Pc + Join_Control
WS = r'\t\n\x{0B}\f\r \x{85}\x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}'
WORD = (r'\p{L}\p{Nl}\p{Mn}\p{Mc}\p{Me}\p{Nd}\p{Pc}\x{200C}\x{200D}\x{345}\x{24B6}-\x{24E9}'
        r'\x{1F130}-\x{1F149}\x{1F150}-\x{1F169}\x{1F170}-\x{1F189}')


def norm(arr):
    """arr: pyarrow string array (raw text). Returns large_string array of normalised text.

    Spark-equivalent cleanup, in the same order as the Spark job:
      1. lower case;
      2. URLs removed: 'http://' or 'https://' plus everything up to the next ASCII whitespace (Spark used \\S
         WITHOUT (?U), i.e. ASCII whitespace only), replaced by a space;
      3. every character that is not a word character, not whitespace and not an apostrophe becomes a space
         (so apostrophes stay inside words);
      4. any run of whitespace becomes one ASCII space;
      5. leading/trailing spaces trimmed; nulls become ''.
    The result has tokens separated by single spaces, which is what fastscan.c's shingle_one() assumes.
    """
    t = pc.utf8_lower(arr)
    t = pc.replace_substring_regex(t, r'https?://[^ \t\n\x0B\f\r]+', ' ')   # Java \S without (?U) = ASCII
    t = pc.replace_substring_regex(t, "[^" + WORD + WS + "']", ' ')
    t = pc.replace_substring_regex(t, "[" + WS + "]+", ' ')
    t = pc.utf8_trim(t, ' ')
    return pc.cast(pc.fill_null(t, ''), pa.large_string())


def _buffers(a):
    """Return (array, int64 offsets as numpy, address of the data buffer) for passing a string array to C.

    The array is combined into one chunk and cast to large_string (int64 offsets). If it is a slice (offset != 0)
    or has nulls, it is rebuilt from Python values so the offsets start at row 0 and null rows take zero bytes
    (the C code ignores the validity bitmap). The data buffer can be None when every string is empty; then 0 is
    passed. The returned array must stay alive while C uses the address.
    """
    a = a.combine_chunks() if isinstance(a, pa.ChunkedArray) else a
    a = pc.cast(a, pa.large_string())
    if a.offset != 0 or a.null_count:
        a = pa.array(a.to_pylist(), type=pa.large_string())
    off = np.frombuffer(a.buffers()[1], dtype=np.int64, count=len(a) + 1)
    data = a.buffers()[2]
    return a, off, (data.address if data is not None else 0)


def shingles(norm_arr):
    """distinct sorted shingle hashes per row: (flat uint64, ptr int64[n+1], counts int64[n] (-1 = under 3 tokens)).

    Shingle = word 3-gram "w1 w2 w3", hashed with xxhash64 seed 42 (as in Spark); row i's set is
    flat[ptr[i]:ptr[i+1]]. Input must already be normalised with norm().
    """
    a, off, dptr = _buffers(norm_arr)
    n = len(a)
    # upper bound on output size: a row of len bytes has at most len//2 + 1 tokens, hence fewer shingles
    room = int(((off[1:] - off[:-1]) // 2 + 1).sum()) + 1
    out = np.empty(room, np.uint64); ptr = np.empty(n + 1, np.int64); cnt = np.empty(n, np.int64)
    _lib.shingle_batch(dptr, off.ctypes.data, n, out.ctypes.data, ptr.ctypes.data, cnt.ctypes.data)
    return out[:ptr[-1]].copy(), ptr, cnt


class Index:
    """Prefix-filter index over query texts (already normalised). t = tnum/tden.

    Only one Index can be live per process: the C side keeps the index in global variables.
    Attributes: qsh/qptr/qlen = each query's full shingle set and its size; ok = query has >= 1 shingle;
    n_shingles / n_prefix / n_keys = sizes for the logs; stats = [rows shingled, candidate pairs verified].
    """
    def __init__(self, qnorm, tnum=7, tden=10):
        """Shingle the queries, choose each query's prefix and hand the inverted index to C (init_index).

        Prefix filter: a query with n shingles indexes only its first n - ceil(t*n) + 1 shingles in one fixed
        global order (rarest first = lowest document frequency among the QUERIES, ties broken by hash value).
        Any archive text with Jaccard >= t shares at least ceil(t*n) of the query's n shingles, so it must hit at
        least one of the indexed ones. This is exact for any fixed order; the order only changes speed.
        """
        sh, ptr, cnt = shingles(qnorm)
        self.ok = cnt > 0                      # queries with at least one shingle
        self.qsh, self.qptr, self.qlen = sh, ptr, np.diff(ptr)
        self.nq = len(cnt)
        # global order: rarest first by document frequency within the queries, ties by hash value
        # each query's set is distinct, so the count of a hash in the flat array = number of queries containing it
        keys, inv, df = np.unique(sh, return_inverse=True, return_counts=True)
        rank = np.lexsort((keys, df))          # positions in keys, rarest first
        # order_of_key[k] = global rank of distinct key k (0 = rarest)
        order_of_key = np.empty(len(keys), np.int64); order_of_key[rank] = np.arange(len(keys))
        qid = np.repeat(np.arange(self.nq), self.qlen)
        o = order_of_key[inv]
        srt = np.lexsort((o, qid))             # within each query, rarest first
        # prefix length per query: n - ceil(t n) + 1 (0 for queries with no shingles); ceil done in integers
        n = self.qlen
        alpha = (tnum * n + tden - 1) // tden  # ceil(t * n)
        plen = np.where(n > 0, n - alpha + 1, 0)
        # srt keeps the rows grouped by query in the same blocks as sh, so slot k of srt belongs to the query of
        # slot k; pos_in_q is the rank of that slot inside its query (0 = rarest). Keep the first plen of each query.
        pos_in_q = np.arange(len(sh)) - np.repeat(ptr[:-1], n)
        take = srt[pos_in_q < np.repeat(plen, n)]
        pk, pq_ = sh[take], qid[take]
        # build the inverted index in CSR form: sort (shingle, query) pairs by shingle, then ukeys[i] -> its
        # postings pq_[kptr[i]:kptr[i+1]] (query ids, int32)
        s2 = np.lexsort((pq_, pk)); pk, pq_ = pk[s2], pq_[s2].astype(np.int32)
        ukeys, start = np.unique(pk, return_index=True)
        kptr = np.append(start, len(pk)).astype(np.int64)
        self._keep = (ukeys, kptr, pq_)        # keep buffers alive for the C side
        self.n_prefix = len(pk); self.n_keys = len(ukeys); self.n_shingles = len(sh)
        _lib.init_index(ukeys.ctypes.data, kptr.ctypes.data, len(ukeys), pq_.ctypes.data,
                        self.qsh.ctypes.data, self.qptr.ctypes.data, self.nq, tnum, tden)
        self.stats = np.zeros(2, np.int64)

    def scan(self, anorm, keep=None):
        """anorm: normalised archive texts; keep: bool mask. Returns (row, query, inter, m, jaccard) arrays.

        row = index into anorm, query = index into the query array given to Index, inter = |A n B|, m = |B|
        (archive distinct shingles); jaccard = inter / (|A| + m - inter) as float, for pairs that passed the
        exact integer test 10*inter >= 7*(|A| + m - inter) in C. self.stats accumulates
        [rows shingled, candidate pairs verified].
        Hit-buffer retry: start with room for max(1024, n) hits; if C reports more (negative return = -needed),
        allocate needed + 16 and rescan the whole batch. Only the successful attempt's stats are added.
        """
        a, off, dptr = _buffers(anorm)
        n = len(a)
        k = np.ones(n, np.uint8) if keep is None else np.ascontiguousarray(keep, dtype=np.uint8)
        cap = max(1024, n)
        while True:
            hr = np.empty(cap, np.int64); hq = np.empty(cap, np.int32); hi = np.empty(cap, np.int32); hm = np.empty(cap, np.int32)
            st = np.zeros(2, np.int64)
            r = _lib.scan_batch(dptr, off.ctypes.data, n, k.ctypes.data, hr.ctypes.data, hq.ctypes.data,
                                hi.ctypes.data, hm.ctypes.data, cap, st.ctypes.data)
            if r >= 0: break
            cap = -r + 16
        self.stats += st
        hr, hq, hi, hm = hr[:r], hq[:r], hi[:r], hm[:r]
        na = self.qlen[hq]
        return hr, hq, hi, hm, hi / (na + hm - hi)


def jaccard_bruteforce(qsh, qptr, ash, aptr):
    """exact Jaccard of every query x archive pair (tiny inputs only). returns dense matrix (nan if either has no shingles).

    Brute-force checker that uses Python sets and no index or filters, as an independent check of the index.
    Inputs are CSR shingle arrays as returned by shingles(). Not called by the scripts in fastscan/ (the toy test,
    toy_fast_v1.py, writes its own equivalent loop with the integer test).
    """
    nq, na = len(qptr) - 1, len(aptr) - 1
    J = np.full((nq, na), np.nan)
    A = [set(ash[aptr[j]:aptr[j + 1]].tolist()) for j in range(na)]
    for i in range(nq):
        q = set(qsh[qptr[i]:qptr[i + 1]].tolist())
        if not q: continue
        for j in range(na):
            if A[j]:
                c = len(q & A[j]); J[i, j] = c / (len(q) + len(A[j]) - c)
    return J
