/* fastscan.c - exact Jaccard >= t search of a small query set against a stream of texts.
 * COPY of fastscan/fastscan.c, kept here only for fast2_full --selfcheck (the fastscan reference pipeline).
 * Texts arrive ALREADY NORMALISED (Spark rules applied in Python/pyarrow): tokens separated by single ASCII spaces.
 * Shingle = word 3-gram, hashed exactly like Spark: xxhash64(seed 42) of the UTF-8 bytes "w1 w2 w3".
 * Prefix filter: each query indexes only its rarest |A| - ceil(t|A|) + 1 shingles; any pair with J >= t must
 * share one of them (holds for any fixed global order). Hits are verified by an exact sorted-set intersection.
 * Build: gcc -O3 -march=x86-64-v2 -shared -fPIC -o libfastscan.so fastscan.c
 *
 * ---- Added notes (comment-only pass) ----
 * Role: the C core of the ORIGINAL exact scanner (fastscan/, "v1"). It is loaded through ctypes by fastscan.py
 * (which does the text cleanup, chooses the prefix of each query and owns all the arrays passed in here).
 * Exported functions:
 *   xxh64         - xxhash64 of a byte string (also exported so Python can call it).
 *   shingle_batch - word-3-gram shingle hashes (sorted, distinct) for every row of an Arrow string array.
 *   init_index    - builds the inverted index (open-addressing hash table: shingle hash -> list of query ids).
 *   scan_batch    - for every archive row: shingle it, look each shingle up in the inverted index, apply the
 *                   length filter, count the exact intersection with each candidate query, and keep the pair if
 *                   Jaccard >= tnum/tden (tested in integers). Hits go into caller-supplied buffers.
 * Method references: prefix filter (Chaudhuri, Ganti & Kaushik, ICDE 2006), length filter (Bayardo, Ma & Srikant,
 * WWW 2007), see also Xiao et al., WWW 2008. There is NO Bloom filter in this version (that came in fastscan2.c).
 * Input: Arrow large_string buffers (data bytes + int64 offsets) of already-normalised text. Output: hit arrays of
 * (archive row, query id, intersection size, archive shingle count). No text is written anywhere.
 * The run scripts build it with: gcc -O3 -Wall -shared -fPIC -o libfastscan.so fastscan.c
 * Jobs that used it (Hyak Slurm): toy test 43301873 (failed before reading data: container could not see /gscratch)
 * and 43301955 (PASS, 33 pairs, same as brute force and Spark); pilot comment file 43321861 (8 cores; all 427
 * Spark matches found, no extras). fastscan2/ref/fastscan.c is a copy of this file used by fast2_full --selfcheck.
 * State: the index lives in file-scope static variables, so there is one index per process. Forked worker
 * processes each get their own copy of it (and of the STAMP array and ROWCTR counter).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---------- xxhash64 (reference algorithm, as in Spark's XXH64.hashUnsafeBytes) ---------- */
/* P1..P5 are the five standard xxhash64 prime constants. rotl = 64-bit rotate left; rd64/rd32 = unaligned
 * little-endian loads (memcpy, so no alignment assumptions; byte order is the host's, little-endian on x86-64);
 * rnd = one xxhash64 accumulator round; mrg = merge one accumulator into the hash (used for inputs >= 32 bytes). */
#define P1 11400714785074694791ULL
#define P2 14029467366897019727ULL
#define P3 1609587929392839161ULL
#define P4 9650029242287828579ULL
#define P5 2870177450012600261ULL
static inline uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
static inline uint64_t rd64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint64_t rnd(uint64_t acc, uint64_t in) { acc += in * P2; acc = rotl(acc, 31); return acc * P1; }
static inline uint64_t mrg(uint64_t acc, uint64_t v) { v = rnd(0, v); acc ^= v; return acc * P1 + P4; }
/* xxh64: standard xxhash64 of len bytes at p with the given seed (this code always uses seed 42, as Spark does).
 * Steps: 32-byte stripes through four accumulators (only if len >= 32), add len, then the 8-byte, 4-byte and
 * 1-byte tails, then the final avalanche. Spark's hashUnsafeBytes processes bytes the same way, so the shingle
 * hashes here equal the Spark job's hashes. */
uint64_t xxh64(const uint8_t *p, size_t len, uint64_t seed) {
    const uint8_t *end = p + len; uint64_t h;
    if (len >= 32) {
        const uint8_t *lim = end - 32;
        uint64_t v1 = seed + P1 + P2, v2 = seed + P2, v3 = seed, v4 = seed - P1;
        do { v1 = rnd(v1, rd64(p)); v2 = rnd(v2, rd64(p + 8)); v3 = rnd(v3, rd64(p + 16)); v4 = rnd(v4, rd64(p + 24)); p += 32; } while (p <= lim);
        h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
        h = mrg(h, v1); h = mrg(h, v2); h = mrg(h, v3); h = mrg(h, v4);
    } else h = seed + P5;
    h += (uint64_t)len;
    while (p + 8 <= end) { h ^= rnd(0, rd64(p)); h = rotl(h, 27) * P1 + P4; p += 8; }
    if (p + 4 <= end) { h ^= (uint64_t)rd32(p) * P1; h = rotl(h, 23) * P2 + P3; p += 4; }
    while (p < end) { h ^= (*p) * P5; h = rotl(h, 11) * P1; p++; }
    h ^= h >> 33; h *= P2; h ^= h >> 29; h *= P3; h ^= h >> 32;
    return h;
}

/* cmp_u64: qsort comparator for unsigned 64-bit hashes (ascending, unsigned order). */
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }

/* shingles of one normalised text into out (sorted, distinct). returns count, or -1 if fewer than 3 tokens.
 * tok is scratch of size >= len/2+2. */
/* Added notes: because the text is normalised (single spaces, no leading/trailing space), token k starts at
 * tok[k] and token k+2 ends one byte before the start of token k+3 (or at len for the last token). The shingle is
 * therefore the exact byte range "w1 w2 w3" including its two separating spaces, hashed with xxh64 seed 42.
 * The hashes are sorted and duplicates dropped, so each text becomes a SET of shingle hashes (needed for Jaccard
 * and for the merge-style intersection in scan_batch). A text of n tokens has at most n-2 shingles, and n is at
 * most len/2+1, which is why len/2-sized scratch buffers are enough. */
static int64_t shingle_one(const uint8_t *s, int64_t len, int64_t *tok, uint64_t *out) {
    int64_t nt = 0, i = 0;
    while (i < len) { /* token starts */
        tok[nt++] = i;
        while (i < len && s[i] != ' ') i++;
        i++; /* skip single space */
    }
    if (len == 0) nt = 0;
    if (nt < 3) return -1;
    int64_t n = 0;
    for (int64_t k = 0; k + 2 < nt; k++) {
        int64_t a = tok[k], b; /* end of token k+2 */
        b = (k + 3 < nt) ? tok[k + 3] - 1 : len;
        out[n++] = xxh64(s + a, (size_t)(b - a), 42);
    }
    qsort(out, (size_t)n, sizeof(uint64_t), cmp_u64);
    int64_t m = 0;
    for (int64_t k = 0; k < n; k++) if (m == 0 || out[k] != out[m - 1]) out[m++] = out[k];
    return m;
}

/* batch shingling (used for the query side). data/offsets = Arrow string buffers (int64 offsets).
 * counts[i] = number of distinct shingles (or -1). out must hold sum of (len/2+1) per row; out_ptr[i] = start. */
/* Added notes: output is CSR-style - row i's sorted distinct hashes are out[out_ptr[i] .. out_ptr[i+1]); rows
 * with fewer than 3 tokens get counts[i] = -1 and an empty range. Returns the total number of hashes written.
 * Used by fastscan.shingles() for the queries (index build) and for the toy brute-force check. */
int64_t shingle_batch(const uint8_t *data, const int64_t *off, int64_t n, uint64_t *out, int64_t *out_ptr, int64_t *counts) {
    int64_t maxlen = 0; for (int64_t i = 0; i < n; i++) if (off[i + 1] - off[i] > maxlen) maxlen = off[i + 1] - off[i];
    int64_t *tok = malloc(sizeof(int64_t) * (maxlen / 2 + 4));
    int64_t pos = 0;
    for (int64_t i = 0; i < n; i++) {
        out_ptr[i] = pos;
        int64_t c = shingle_one(data + off[i], off[i + 1] - off[i], tok, out + pos);
        counts[i] = c; if (c > 0) pos += c;
    }
    out_ptr[n] = pos; free(tok); return pos;
}

/* ---------- index: open-addressing table key -> [start,end) into postings ---------- */
/* Inverted index state (file-scope, one index per process):
 *   T_key/T_lo/T_hi - open-addressing hash table with linear probing; slot h holds a shingle hash T_key[h] and
 *                     its postings range POST[T_lo[h] .. T_hi[h]). T_hi[h] = -1 marks an empty slot.
 *   T_mask          - table size - 1 (size is a power of two).
 *   POST            - postings: query ids, grouped by shingle (CSR from Python).
 *   QSH/QPTR/NQ     - every query's FULL sorted distinct shingle set (not only its prefix), for exact verification.
 *   STAMP/ROWCTR    - de-duplication: STAMP[q] is the number of the last archive row that already verified query q,
 *                     so a query reached through several of its prefix shingles is verified only once per row.
 *                     ROWCTR is a running row counter that keeps growing across scan_batch calls.
 *   TNUM/TDEN       - the threshold t = TNUM/TDEN (7/10), kept as integers so all tests are exact.
 * mix(): a 64-bit bit-mixing step applied to the shingle hash before masking, so the slot index does not depend
 * only on the low bits of the xxhash value. */
static uint64_t *T_key; static int64_t *T_lo, *T_hi; static uint64_t T_mask;
static const int32_t *POST; static const uint64_t *QSH; static const int64_t *QPTR; static int64_t NQ;
static int64_t *STAMP; static int64_t ROWCTR = 0; /* per query: last archive row (global counter) that touched it */
static int TNUM, TDEN;
static inline uint64_t mix(uint64_t k) { k ^= k >> 31; k *= 0x7fb5d329728ea185ULL; k ^= k >> 27; return k; }

/* keys sorted distinct, kptr CSR into postings (query ids). qsh/qptr = each query's full sorted shingle set. */
/* Added notes: builds the inverted index. keys[i] is one prefix shingle hash; its postings (the ids of queries
 * that put this shingle in their prefix) are post[kptr[i] .. kptr[i+1]). The table size is the smallest power of
 * two >= 2*nkeys + 16, so it is at most about half full and linear probing stays short. The pointers post, qsh and
 * qptr are NOT copied: Python must keep those arrays alive (fastscan.Index does). The tables are malloc'd and never
 * freed; calling init_index twice in one process would replace (and leak) the first index. Always returns 0. */
int init_index(const uint64_t *keys, const int64_t *kptr, int64_t nkeys, const int32_t *post,
               const uint64_t *qsh, const int64_t *qptr, int64_t nq, int tnum, int tden) {
    uint64_t cap = 1; while (cap < (uint64_t)nkeys * 2 + 16) cap <<= 1;
    T_key = malloc(cap * 8); T_lo = malloc(cap * 8); T_hi = malloc(cap * 8); T_mask = cap - 1;
    for (uint64_t i = 0; i < cap; i++) T_hi[i] = -1;
    for (int64_t i = 0; i < nkeys; i++) {
        /* insert: start at the mixed hash's slot, step forward to the next empty slot (keys are distinct) */
        uint64_t h = mix(keys[i]) & T_mask;
        while (T_hi[h] >= 0) h = (h + 1) & T_mask;
        T_key[h] = keys[i]; T_lo[h] = kptr[i]; T_hi[h] = kptr[i + 1];
    }
    POST = post; QSH = qsh; QPTR = qptr; NQ = nq; TNUM = tnum; TDEN = tden;
    STAMP = malloc(sizeof(int64_t) * (size_t)nq); for (int64_t i = 0; i < nq; i++) STAMP[i] = -1;
    return 0;
}

/* scan a batch of normalised archive texts. keep[i]=0 skips the row. Writes up to cap hits:
 * hit_row, hit_q, hit_inter, hit_m (archive distinct shingles). returns number of hits (or -(needed) if cap exceeded).
 * stats[0] += rows shingled, stats[1] += candidate pairs verified. */
/* Added notes on the steps for each archive row B (m = |B| distinct shingles):
 *   1. shingle it (rows with < 3 tokens are skipped);
 *   2. probe the inverted index with EVERY shingle of B; a query A can be a match only if B contains one of
 *      A's prefix shingles (prefix filter), so only queries reached this way are candidates;
 *   3. skip a query already handled for this row (STAMP);
 *   4. length filter: J >= t requires t*|A| <= |B| <= |A|/t, tested as TDEN*m >= TNUM*na and TNUM*m <= TDEN*na;
 *   5. exact verification: count c = |A n B| by merging the two sorted hash arrays, and keep the pair if
 *      c / (|A|+|B|-c) >= TNUM/TDEN, tested in integers as TDEN*c >= TNUM*(na + m - c) (no rounding error).
 * Hit-buffer protocol: hits are written only while nh < cap, but nh keeps counting. If more than cap hits occur
 * the function returns -nh (the total needed); the caller (fastscan.Index.scan) then allocates bigger buffers and
 * rescans the whole batch. stats are accumulated into the caller's array, which the caller zeroes per attempt.
 * Hit rows are indices into this batch (0..n-1), query ids are indices into the query array given to Index. */
int64_t scan_batch(const uint8_t *data, const int64_t *off, int64_t n, const uint8_t *keep,
                   int64_t *hit_row, int32_t *hit_q, int32_t *hit_inter, int32_t *hit_m, int64_t cap, int64_t *stats) {
    int64_t maxlen = 0; for (int64_t i = 0; i < n; i++) if (off[i + 1] - off[i] > maxlen) maxlen = off[i + 1] - off[i];
    int64_t *tok = malloc(sizeof(int64_t) * (maxlen / 2 + 4));
    uint64_t *sh = malloc(sizeof(uint64_t) * (maxlen / 2 + 4));
    int64_t nh = 0;
    for (int64_t r = 0; r < n; r++) {
        if (!keep[r]) continue;
        int64_t m = shingle_one(data + off[r], off[r + 1] - off[r], tok, sh);
        if (m <= 0) continue;
        stats[0]++;
        int64_t stamp = ++ROWCTR;
        for (int64_t j = 0; j < m; j++) {
            /* inverted-index lookup: linear probing until the key or an empty slot (= not indexed) is found */
            uint64_t h = mix(sh[j]) & T_mask;
            while (T_hi[h] >= 0 && T_key[h] != sh[j]) h = (h + 1) & T_mask;
            if (T_hi[h] < 0) continue;
            for (int64_t p = T_lo[h]; p < T_hi[h]; p++) {
                int32_t q = POST[p];
                if (STAMP[q] == stamp) continue;
                STAMP[q] = stamp;
                const uint64_t *a = QSH + QPTR[q]; int64_t na = QPTR[q + 1] - QPTR[q];
                /* size filter: J >= t needs t*na <= m <= na/t */
                if ((int64_t)TDEN * m < (int64_t)TNUM * na || (int64_t)TNUM * m > (int64_t)TDEN * na) continue;
                stats[1]++;
                /* exact intersection of two sorted distinct arrays (merge walk) */
                int64_t x = 0, y = 0, c = 0;
                while (x < na && y < m) { if (a[x] < sh[y]) x++; else if (a[x] > sh[y]) y++; else { c++; x++; y++; } }
                /* integer Jaccard test: c/(na+m-c) >= TNUM/TDEN  <=>  TDEN*c >= TNUM*(na+m-c) */
                if ((int64_t)TDEN * c >= (int64_t)TNUM * (na + m - c)) {
                    if (nh < cap) { hit_row[nh] = r; hit_q[nh] = q; hit_inter[nh] = (int32_t)c; hit_m[nh] = (int32_t)m; }
                    nh++;
                }
            }
        }
    }
    free(tok); free(sh);
    return nh <= cap ? nh : -nh;
}
