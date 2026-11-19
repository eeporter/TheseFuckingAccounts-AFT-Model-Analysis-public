/* fastscan2.c - faster exact Jaccard >= t search of a small query set against a stream of RAW texts.
 * Same results as fastscan.c (same cleanup rules, same xxhash64 seed-42 word 3-gram shingles, same prefix-filter
 * index, same exact verification). What is different is only how much work is done per archive row:
 *   1. Cleanup (lower case, URL removal, non-word -> space, whitespace collapse) is ONE pass in C over the raw
 *      Arrow buffer, driven by per-code-point tables that Python builds from pyarrow itself, instead of several
 *      pyarrow regex passes that each copy the whole column.
 *   2. The 140-character rule is applied in C before any cleanup, on the raw buffer (no filtered copy).
 *   3. Shingle hashes are probed against a small in-cache bit filter first; the row is only sorted, de-duplicated
 *      and verified if some shingle is really one of the indexed (prefix) shingles. Most rows stop there.
 * Rows that are not valid UTF-8 are flagged and handled by the caller with the old pyarrow path.
 * Build: gcc -O3 -shared -fPIC -o libfastscan2.so fastscan2.c
 *
 * What it is: the C half of an exact near-duplicate ("copypasta") search. A pair (archive text, query text) counts
 * when the exact Jaccard similarity of their sets of word-3-gram shingles is >= t (t = TNUM/TDEN, 7/10 in every run).
 * Method: exact prefix-filter set-similarity join (Chaudhuri, Ganti & Kaushik, ICDE 2006; length filter as in
 * Bayardo, Ma & Srikant, WWW 2007; Xiao et al., WWW 2008), with a one-hash Bloom filter in front of the inverted index.
 *
 * Called only through ctypes from fastscan2.py (library path from $FASTSCAN2_LIB, else ./libfastscan2.so).
 * Inputs : per-code-point lower-case / word-class tables (init_tables); the query side as sorted, de-duplicated
 *          shingle-hash sets plus the prefix-filter postings built in fastscan2.py (init_index); then raw archive
 *          texts as Arrow string buffers (scan_raw), or already-cleaned texts (scan_norm).
 * Outputs: no text. Only hit arrays (archive row number, query number, intersection size c, archive set size m),
 *          per-row flags for the caller's slow path, and counters in stats[].
 * State  : everything lives in file-level static variables, so there is ONE index per process (the last
 *          init_index call wins; earlier allocations are not freed).
 *
 * Jobs (Hyak Slurm) that built and used this file: pilot 43324646 and 43328586; rules check 43328587; first full
 * run array 43329112; 4-vs-12 worker A/B 43332090; second full run array 43332591; whole-pool run array 43340006;
 * rerun of tasks 1, 3, 22 array 43360110.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---------- xxHash64 ----------
 * A plain C version of the standard 64-bit xxHash (XXH64) with its five primes P1..P5. Every shingle is hashed with
 * seed 42, the same hash and seed as the earlier Spark job, so shingle hashes are comparable across all versions.
 * rotl = rotate left; rd64 / rd32 = unaligned little-endian loads (memcpy, so no alignment faults); rnd = one XXH64
 * accumulator round; mrg = merge one accumulator into the hash. */
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
/* XXH64 of the len bytes at p with the given seed. */
static inline uint64_t xxh64(const uint8_t *p, size_t len, uint64_t seed) {
    const uint8_t *end = p + len; uint64_t h;
    if (len >= 32) {   /* long input: four parallel accumulators over 32-byte stripes */
        const uint8_t *lim = end - 32;
        uint64_t v1 = seed + P1 + P2, v2 = seed + P2, v3 = seed, v4 = seed - P1;
        do { v1 = rnd(v1, rd64(p)); v2 = rnd(v2, rd64(p + 8)); v3 = rnd(v3, rd64(p + 16)); v4 = rnd(v4, rd64(p + 24)); p += 32; } while (p <= lim);
        h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
        h = mrg(h, v1); h = mrg(h, v2); h = mrg(h, v3); h = mrg(h, v4);
    } else h = seed + P5;
    h += (uint64_t)len;
    /* tail: remaining 8-byte words, then one 4-byte word, then single bytes; then the final avalanche */
    while (p + 8 <= end) { h ^= rnd(0, rd64(p)); h = rotl(h, 27) * P1 + P4; p += 8; }
    if (p + 4 <= end) { h ^= (uint64_t)rd32(p) * P1; h = rotl(h, 23) * P2 + P3; p += 4; }
    while (p < end) { h ^= (*p) * P5; h = rotl(h, 11) * P1; p++; }
    h ^= h >> 33; h *= P2; h ^= h >> 29; h *= P3; h ^= h >> 32;
    return h;
}
/* Exported copy of xxh64 so Python can call the hash directly (fastscan2.py does not declare or use it). */
uint64_t xxh64_export(const uint8_t *p, size_t len, uint64_t seed) { return xxh64(p, len, seed); }

/* in-place sort of 64-bit hashes: quicksort (median of three) down to runs of 16, then insertion sort.
 * Used to turn a text's list of shingle hashes into a sorted list, so repeats can be dropped (giving a SET) and
 * two sets can be intersected with a linear merge. */
static void sort_u64(uint64_t *a, int64_t n) {
    while (n > 16) {
        /* pv = median of first, middle and last element (Hoare partition around it) */
        uint64_t x = a[0], y = a[n / 2], z = a[n - 1];
        uint64_t pv = x < y ? (y < z ? y : (x < z ? z : x)) : (x < z ? x : (y < z ? z : y));
        int64_t i = 0, j = n - 1;
        while (i <= j) {
            while (a[i] < pv) i++;
            while (a[j] > pv) j--;
            if (i <= j) { uint64_t t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
        }
        /* recurse into the smaller side and loop on the larger one (keeps recursion depth O(log n)) */
        if (j + 1 < n - i) { sort_u64(a, j + 1); a += i; n -= i; } else { sort_u64(a + i, n - i); n = j + 1; }
    }
    /* insertion sort finishes the short run left over */
    for (int64_t i = 1; i < n; i++) { uint64_t v = a[i]; int64_t j = i - 1; while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; } a[j + 1] = v; }
}

/* ---------- per-code-point tables (built by Python from pyarrow) ----------
 * fastscan2.py's build_tables() runs pyarrow's own utf8_lower and the cleanup's RE2 character classes over every
 * Unicode code point, so this C cleanup gives the same output as the pyarrow regex cleanup (norm() in fastscan2.py)
 * in whatever pyarrow version is installed. The arrays are owned by Python (kept alive in _TABLES); C only keeps
 * pointers. */
static const uint32_t *LOW;  /* [0x110000] lower-case code point */
static const uint8_t *CLS;   /* [0x110000] 1 = word character (kept), 0 = separator */
static uint8_t A_LOW[128], A_WORD[128];   /* ASCII fast path: lower-case byte, and word class of that lower-case byte */

/* ---------- index ----------
 * Inverted index of the prefix shingles (shingle hash -> postings = ids of queries that indexed that shingle):
 *   T_key / T_lo / T_hi: open-addressing hash table (linear probing, power-of-two size, T_mask = size - 1).
 *     Slot h holds key T_key[h] and its postings POST[T_lo[h] .. T_hi[h]); T_hi[h] < 0 marks an empty slot.
 *   BLOOM / B_mask: one bit array in front of the table (see init_index).
 *   POST: postings (query ids, int32), owned by Python. QSH / QPTR: every query's sorted, de-duplicated shingle
 *     set, query q = QSH[QPTR[q] .. QPTR[q+1]). NQ = number of queries.
 *   STAMP / ROWCTR: STAMP[q] = number of the last archive row in which query q was already verified, so a query that
 *     appears in several postings lists of the same row is verified only once per row.
 *   TNUM / TDEN: Jaccard threshold as a fraction (7/10), so every test below is done in integers. */
static uint64_t *T_key; static int64_t *T_lo, *T_hi; static uint64_t T_mask;
static uint64_t *BLOOM; static uint64_t B_mask;
static const int32_t *POST; static const uint64_t *QSH; static const int64_t *QPTR; static int64_t NQ;
static int64_t *STAMP; static int64_t ROWCTR = 0;
static int TNUM, TDEN;
/* mix: scrambles a shingle hash (xorshift-multiply finalizer). Its low bits pick the hash-table slot and bits 20 and
 * up (m >> 20) pick the Bloom bit, so slot and bit come from different parts of the mixed value. */
static inline uint64_t mix(uint64_t k) { k ^= k >> 31; k *= 0x7fb5d329728ea185ULL; k ^= k >> 27; return k; }

/* Store pointers to the two 0x110000-entry tables and fill the 128-entry ASCII copies used on the fast path.
 * Returns -1 (caller raises) if some ASCII character lower-cases to a non-ASCII one, which the ASCII path cannot
 * represent; 0 otherwise. */
int init_tables(const uint32_t *low, const uint8_t *cls) {
    LOW = low; CLS = cls;
    for (int i = 0; i < 128; i++) { if (low[i] >= 128) return -1; A_LOW[i] = (uint8_t)low[i]; A_WORD[i] = cls[low[i]]; }
    return 0;
}

/* Build the inverted index from the arrays fastscan2.py's Index prepared:
 *   keys[0..nkeys): distinct prefix shingle hashes (sorted, unique); postings of keys[i] = post[kptr[i] .. kptr[i+1]);
 *   qsh / qptr / nq: every query's full shingle set (needed for exact verification); tnum / tden: threshold.
 * keys and kptr are copied into the hash table; post, qsh and qptr are used in place (Python keeps them alive).
 * Always returns 0 (malloc results are not checked). */
int init_index(const uint64_t *keys, const int64_t *kptr, int64_t nkeys, const int32_t *post,
               const uint64_t *qsh, const int64_t *qptr, int64_t nq, int tnum, int tden) {
    /* hash table: smallest power of two >= 2*nkeys + 16, so at most about half the slots are used (short probes) */
    uint64_t cap = 1; while (cap < (uint64_t)nkeys * 2 + 16) cap <<= 1;
    T_key = malloc(cap * 8); T_lo = malloc(cap * 8); T_hi = malloc(cap * 8); T_mask = cap - 1;
    for (uint64_t i = 0; i < cap; i++) T_hi[i] = -1;   /* all slots empty */
    /* Bloom filter: one bit array with at least 16 bits per indexed shingle (power of two, at least 1024 bits) and
     * ONE bit per shingle. With n keys in b >= 16n bits the expected share of set bits is 1 - exp(-n/b) <= about 6%
     * (less when b is rounded up). Bit 0 => the shingle is certainly not indexed (no false negatives); bit 1 =>
     * look it up in the hash table, which rejects the false positives. The array is small enough to stay in CPU
     * cache, so most archive shingles are rejected without touching the larger hash table. */
    uint64_t bits = 1024; while (bits < (uint64_t)nkeys * 16) bits <<= 1;   /* about 6% of bits set */
    BLOOM = calloc(bits / 64, 8); B_mask = bits - 1;
    for (int64_t i = 0; i < nkeys; i++) {
        uint64_t m = mix(keys[i]);
        uint64_t h = m & T_mask;
        while (T_hi[h] >= 0) h = (h + 1) & T_mask;   /* linear probing to the first empty slot (keys are unique) */
        T_key[h] = keys[i]; T_lo[h] = kptr[i]; T_hi[h] = kptr[i + 1];
        uint64_t b = (m >> 20) & B_mask; BLOOM[b >> 6] |= 1ULL << (b & 63);
    }
    POST = post; QSH = qsh; QPTR = qptr; NQ = nq; TNUM = tnum; TDEN = tden;
    /* STAMP starts at -1; ROWCTR only grows (from 1), so no query starts out as "already seen" */
    STAMP = malloc(sizeof(int64_t) * (size_t)nq); for (int64_t i = 0; i < nq; i++) STAMP[i] = -1;
    return 0;
}

/* Look up shingle hash k: Bloom bit first (0 => not indexed, return -1 at once), then linear probing in the hash
 * table until the key or an empty slot is found. */
static inline int64_t tfind(uint64_t k) { /* slot of key, or -1 */
    uint64_t m = mix(k);
    uint64_t b = (m >> 20) & B_mask;
    if (!(BLOOM[b >> 6] & (1ULL << (b & 63)))) return -1;
    uint64_t h = m & T_mask;
    while (T_hi[h] >= 0 && T_key[h] != k) h = (h + 1) & T_mask;
    return T_hi[h] < 0 ? -1 : (int64_t)h;
}

/* decode one UTF-8 code point at s[i] (i < len). returns byte length, or 0 if invalid.
 * Strict: rejects stray continuation bytes, overlong forms (lead bytes C0/C1, 3-byte < U+0800, 4-byte < U+10000),
 * UTF-16 surrogates D800-DFFF, values above U+10FFFF and sequences cut off by the end of the text. */
static inline int dec(const uint8_t *s, int64_t i, int64_t len, uint32_t *cp) {
    uint8_t b = s[i];
    if (b < 0x80) { *cp = b; return 1; }
    if (b < 0xC2) return 0;
    if (b < 0xE0) { if (i + 1 >= len || (s[i + 1] & 0xC0) != 0x80) return 0; *cp = ((uint32_t)(b & 0x1F) << 6) | (s[i + 1] & 0x3F); return 2; }
    if (b < 0xF0) {
        if (i + 2 >= len || (s[i + 1] & 0xC0) != 0x80 || (s[i + 2] & 0xC0) != 0x80) return 0;
        uint32_t c = ((uint32_t)(b & 0x0F) << 12) | ((uint32_t)(s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
        if (c < 0x800 || (c >= 0xD800 && c <= 0xDFFF)) return 0; *cp = c; return 3;
    }
    if (b < 0xF5) {
        if (i + 3 >= len || (s[i + 1] & 0xC0) != 0x80 || (s[i + 2] & 0xC0) != 0x80 || (s[i + 3] & 0xC0) != 0x80) return 0;
        uint32_t c = ((uint32_t)(b & 0x07) << 18) | ((uint32_t)(s[i + 1] & 0x3F) << 12) | ((uint32_t)(s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
        if (c < 0x10000 || c > 0x10FFFF) return 0; *cp = c; return 4;
    }
    return 0;
}
/* encode code point c as UTF-8 at o; returns the number of bytes written (1 to 4). */
static inline int enc(uint32_t c, uint8_t *o) {
    if (c < 0x80) { o[0] = (uint8_t)c; return 1; }
    if (c < 0x800) { o[0] = 0xC0 | (c >> 6); o[1] = 0x80 | (c & 0x3F); return 2; }
    if (c < 0x10000) { o[0] = 0xE0 | (c >> 12); o[1] = 0x80 | ((c >> 6) & 0x3F); o[2] = 0x80 | (c & 0x3F); return 3; }
    o[0] = 0xF0 | (c >> 18); o[1] = 0x80 | ((c >> 12) & 0x3F); o[2] = 0x80 | ((c >> 6) & 0x3F); o[3] = 0x80 | (c & 0x3F); return 4;
}
/* ASCII whitespace: space, \t \n \v \f \r. This is the set that ends a URL, as in the URL pattern
 * https?://[^ \t\n\x0B\f\r]+ of the pyarrow cleanup. */
static inline int is_aws(uint8_t b) { return b == ' ' || (b >= 9 && b <= 13); } /* Java \s without (?U) */

/* Clean one raw text. L = scratch (>= 2*len+8), N = output (>= 2*len+8), tok = token starts in N (>= len+2).
 * Returns number of tokens (>= 0) and sets *nlen, or -1 if the text is not valid UTF-8.
 * Same output as the pyarrow cleanup norm() in fastscan2.py: lower case, URLs removed, every non-word character
 * becomes a separator, runs of separators collapse to ONE space, no leading or trailing space. Output N is the
 * cleaned words joined by single spaces; tok[k] is the byte offset in N where word k starts. Done in two passes
 * (lower-case into L, then URL removal + word/separator split into N) instead of several whole-column regex copies.
 * Buffer sizes: lower-casing one code point can change its UTF-8 length (e.g. 2 -> 3 bytes), hence 2*len. */
static int64_t clean_one(const uint8_t *s, int64_t len, uint8_t *L, uint8_t *N, int64_t *tok, int64_t *nlen) {
    /* pass 1: lower case (ASCII through the 128-entry table; other code points decoded, mapped through LOW, re-encoded;
     * any invalid UTF-8 aborts with -1 so the caller can send the row to the pyarrow path) */
    int64_t ll = 0, i = 0;
    while (i < len) {
        uint8_t b = s[i];
        if (b < 0x80) { L[ll++] = A_LOW[b]; i++; }
        else { uint32_t cp; int k = dec(s, i, len, &cp); if (!k) return -1; ll += enc(LOW[cp], L + ll); i += k; }
    }
    /* pass 2: drop URLs, keep word characters, everything else separates tokens.
     * intok = 1 while inside a word. A separating space is written only when the NEXT word starts and some word
     * was already written (j > 0), which gives the collapse-to-one-space and the trim in one go. */
    int64_t j = 0, nt = 0; int intok = 0; i = 0;
    while (i < ll) {
        uint8_t b = L[i];
        if (b < 0x80) {
            /* URL: "http" + optional "s" + "://" + at least one non-whitespace byte, up to the next ASCII whitespace
             * (same as the regex https?://[^ \t\n\x0B\f\r]+, applied after lower-casing as in norm()). The URL may
             * start inside a word, as with the regex. It is dropped and acts as a separator. The first test
             * (i + 7 < ll; the "+ 0" does nothing) only ensures there are enough bytes for the shortest URL. */
            if (b == 'h' && i + 7 < ll + 0 && L[i + 1] == 't' && L[i + 2] == 't' && L[i + 3] == 'p') {
                int64_t p = i + 4; if (p < ll && L[p] == 's') p++;
                if (p + 3 < ll && L[p] == ':' && L[p + 1] == '/' && L[p + 2] == '/' && !is_aws(L[p + 3])) {
                    p += 3; while (p < ll && !is_aws(L[p])) p++;
                    intok = 0; i = p; continue;
                }
            }
            /* ASCII word byte: start a new word if needed (space before it unless first), then copy the byte */
            if (A_WORD[b]) { if (!intok) { if (j) N[j++] = ' '; tok[nt++] = j; intok = 1; } N[j++] = b; }
            else intok = 0;
            i++;
        } else {   /* non-ASCII: class of the (already lower-cased) code point from CLS; word code points copied as is */
            uint32_t cp; int k = dec(L, i, ll, &cp);
            if (!k) return -1;
            if (CLS[cp]) { if (!intok) { if (j) N[j++] = ' '; tok[nt++] = j; intok = 1; } memcpy(N + j, L + i, (size_t)k); j += k; }
            else intok = 0;
            i += k;
        }
    }
    *nlen = j; return nt;
}

/* test hook: cleaned text of each row written to out (concatenated), out_off[n+1]; bad rows get length 0 and flag.
 * data/off = an Arrow large_string value buffer and its int64 offsets. Called by clean_c() in fastscan2.py, which
 * test_clean_v1.py uses to compare the C cleanup with norm(). Returns the total bytes written. */
int64_t clean_batch(const uint8_t *data, const int64_t *off, int64_t n, uint8_t *out, int64_t *out_off, uint8_t *bad) {
    int64_t maxlen = 0; for (int64_t r = 0; r < n; r++) if (off[r + 1] - off[r] > maxlen) maxlen = off[r + 1] - off[r];
    uint8_t *L = malloc((size_t)(2 * maxlen + 16)); int64_t *tok = malloc(sizeof(int64_t) * (size_t)(maxlen + 4));
    int64_t pos = 0;
    for (int64_t r = 0; r < n; r++) {
        out_off[r] = pos; int64_t nl = 0;
        int64_t nt = clean_one(data + off[r], off[r + 1] - off[r], L, out + pos, tok, &nl);
        bad[r] = nt < 0; if (nt >= 0) pos += nl;
    }
    out_off[n] = pos; free(L); free(tok); return pos;
}

/* exact part for one row whose shingle hashes (unsorted, with repeats) are in sh[0..n): same as fastscan.c
 * r = row number to report. Every query sharing at least one indexed (prefix) shingle with the row is a candidate;
 * each candidate passes the length filter and then an exact Jaccard test on the two shingle sets. Hits are appended
 * to hit_row / hit_q / hit_inter (intersection size c) / hit_m (row's set size m) at position *nh, but only while
 * *nh < cap; *nh is incremented for every hit regardless, so the caller can tell how much room was needed.
 * stats[2] counts candidate pairs that reached the exact intersection. */
static inline void verify_row(uint64_t *sh, int64_t n, int64_t r, int64_t *hit_row, int32_t *hit_q, int32_t *hit_inter,
                              int32_t *hit_m, int64_t cap, int64_t *nh, int64_t *stats) {
    /* sort, then drop repeats in place: the row's shingles become a sorted SET of m hashes */
    sort_u64(sh, n);
    int64_t m = 0;
    for (int64_t k = 0; k < n; k++) if (m == 0 || sh[k] != sh[m - 1]) sh[m++] = sh[k];
    int64_t stamp = ++ROWCTR;   /* new stamp for this row: each candidate query is verified at most once per row */
    for (int64_t j = 0; j < m; j++) {
        int64_t h = tfind(sh[j]);   /* Bloom filter, then hash table */
        if (h < 0) continue;
        for (int64_t p = T_lo[h]; p < T_hi[h]; p++) {   /* postings: queries that indexed this shingle */
            int32_t q = POST[p];
            if (STAMP[q] == stamp) continue;   /* already handled for this row via another shared shingle */
            STAMP[q] = stamp;
            const uint64_t *a = QSH + QPTR[q]; int64_t na = QPTR[q + 1] - QPTR[q];
            /* length filter: Jaccard >= t is only possible if t*na <= m <= na/t. In integers with t = TNUM/TDEN:
             * skip if TDEN*m < TNUM*na (row too short) or TNUM*m > TDEN*na (row too long). */
            if ((int64_t)TDEN * m < (int64_t)TNUM * na || (int64_t)TNUM * m > (int64_t)TDEN * na) continue;
            stats[2]++;
            /* exact intersection size c of the two sorted sets by a linear merge */
            int64_t x = 0, y = 0, c = 0;
            while (x < na && y < m) { if (a[x] < sh[y]) x++; else if (a[x] > sh[y]) y++; else { c++; x++; y++; } }
            /* exact Jaccard test: c / (na + m - c) >= TNUM/TDEN, written without division */
            if ((int64_t)TDEN * c >= (int64_t)TNUM * (na + m - c)) {
                if (*nh < cap) { hit_row[*nh] = r; hit_q[*nh] = q; hit_inter[*nh] = (int32_t)c; hit_m[*nh] = (int32_t)m; }
                (*nh)++;   /* counted even when the buffer is full (see the return value of scan_raw) */
            }
        }
    }
}

/* Scan RAW texts straight from an Arrow string array.
 * data = value buffer; off32/off64 = offsets (exactly one non-NULL), already pointing at the array's first slot;
 * valid = validity bitmap or NULL, voff = bit offset of the first slot; keep = optional per-row mask (0 skips);
 * minchar = minimum number of characters after trimming spaces (the 140 rule), applied here.
 * flag[r] is set to 1 for rows that are not valid UTF-8 and, if flag_game, to 2 for rows containing "old reddit"
 * (the only rows the game-notice rule can change); flagged rows are NOT scanned here, the caller handles them.
 * stats: [0] rows passing the length rule, [1] rows with an indexed shingle (sorted + verified), [2] pairs verified,
 *        [3] invalid-UTF-8 rows, [4] rows flagged for the game-notice path.
 * returns number of hits (or -(needed) if cap exceeded).
 * Hit buffer protocol: hit_* have room for cap hits. If more hits are found, only the first cap are stored and the
 * return value is minus the total number found; the caller (Index._scan_chunk) then allocates that much room (+16)
 * and runs the whole call again. Note stats[] is added to on every call, so the caller passes a fresh zeroed array
 * each time. Null rows (validity bit 0) and rows with keep[r] == 0 are skipped and never flagged.
 * Order of work per row: cheap raw-length tests, game-notice test, one-pass cleanup, shingling with a Bloom/hash
 * probe of each shingle; only rows with an indexed shingle are sorted, de-duplicated and verified. */
int64_t scan_raw(const uint8_t *data, const int32_t *off32, const int64_t *off64, int64_t n,
                 const uint8_t *valid, int64_t voff, const uint8_t *keep, int64_t minchar, int flag_game, uint8_t *flag,
                 int64_t *hit_row, int32_t *hit_q, int32_t *hit_inter, int32_t *hit_m, int64_t cap, int64_t *stats) {
    /* scratch buffers sized once for the longest row of the batch (see clean_one for the factor 2) */
    int64_t maxlen = 0;
    for (int64_t r = 0; r < n; r++) { int64_t l = off32 ? (int64_t)off32[r + 1] - off32[r] : off64[r + 1] - off64[r]; if (l > maxlen) maxlen = l; }
    uint8_t *L = malloc((size_t)(2 * maxlen + 16)), *N = malloc((size_t)(2 * maxlen + 16));
    int64_t *tok = malloc(sizeof(int64_t) * (size_t)(maxlen + 4));
    uint64_t *sh = malloc(sizeof(uint64_t) * (size_t)(maxlen + 4));
    int64_t nh = 0;
    for (int64_t r = 0; r < n; r++) {
        if (keep && !keep[r]) continue;
        if (valid && !((valid[(voff + r) >> 3] >> ((voff + r) & 7)) & 1)) continue;
        int64_t a = off32 ? off32[r] : off64[r], b = off32 ? off32[r + 1] : off64[r + 1];
        /* minchar skip on the RAW length, before any cleanup: a row with fewer bytes than minchar cannot have
         * minchar characters, so it is dropped without being touched. (minchar is 140 in the pilot and 50 in the full
         * runs; in the rules pipeline the real "visible characters" rule is applied later to matched rows only, and
         * visible characters can never exceed raw characters, so this early skip loses nothing.) */
        if (b - a < minchar) continue;                      /* characters <= bytes */
        const uint8_t *s = data + a; int64_t len = b - a;
        int64_t lo = 0, hi = len;                           /* the 140 rule: characters after trimming spaces */
        while (lo < hi && s[lo] == ' ') lo++;
        while (hi > lo && s[hi - 1] == ' ') hi--;
        if (hi - lo < minchar) continue;
        /* count code points = bytes that are not UTF-8 continuation bytes (10xxxxxx); same as pyarrow
         * utf8_length(utf8_trim(text, ' ')) for valid UTF-8 */
        int64_t nch = 0; for (int64_t i = lo; i < hi; i++) nch += (s[i] & 0xC0) != 0x80;
        if (nch < minchar) continue;
        if (flag_game) {   /* rows that contain "old reddit" (ASCII case-insensitive) may hold the game notice: caller's slow path */
            int g = 0;
            for (int64_t i = 0; i + 10 <= len; i++) {
                if ((s[i] | 0x20) != 'o') continue;
                if ((s[i + 1] | 0x20) == 'l' && (s[i + 2] | 0x20) == 'd' && s[i + 3] == ' ' && (s[i + 4] | 0x20) == 'r' && (s[i + 5] | 0x20) == 'e' &&
                    (s[i + 6] | 0x20) == 'd' && (s[i + 7] | 0x20) == 'd' && (s[i + 8] | 0x20) == 'i' && (s[i + 9] | 0x20) == 't') { g = 1; break; }
            }
            if (g) { flag[r] = 2; stats[4]++; continue; }   /* flagged: not scanned here */
        }
        int64_t nl = 0, nt = clean_one(s, len, L, N, tok, &nl);
        if (nt < 0) { flag[r] = 1; stats[3]++; continue; }   /* invalid UTF-8: flagged, caller uses the pyarrow path */
        if (nt < 3) continue;   /* fewer than 3 words: no shingle */
        stats[0]++;
        /* shingles: word 3-gram k = bytes of N from the start of word k to the end of word k+2, i.e. "w1 w2 w3" with
         * single spaces (tok[k+3] - 1 is the space before word k+3; the last shingle runs to the end of N), hashed
         * with xxhash64 seed 42. nt words give nt - 2 shingles (repeats kept here, removed in verify_row). */
        int64_t ns = nt - 2; int cand = 0;
        for (int64_t k = 0; k < ns; k++) {
            int64_t e = (k + 3 < nt) ? tok[k + 3] - 1 : nl;
            uint64_t h = xxh64(N + tok[k], (size_t)(e - tok[k]), 42);
            sh[k] = h;
            /* until one indexed shingle is found: same test as tfind, inlined (Bloom bit, then hash-table probe).
             * By the prefix filter, a row with NO indexed shingle cannot reach Jaccard >= t with any query, so such a
             * row is dropped without sorting. All shingles are still hashed, since verify_row needs the full set. */
            if (!cand) { uint64_t m = mix(h); uint64_t bb = (m >> 20) & B_mask; if (BLOOM[bb >> 6] & (1ULL << (bb & 63))) { uint64_t t = m & T_mask; while (T_hi[t] >= 0 && T_key[t] != h) t = (t + 1) & T_mask; if (T_hi[t] >= 0) cand = 1; } }
        }
        if (!cand) continue;
        stats[1]++;
        verify_row(sh, ns, r, hit_row, hit_q, hit_inter, hit_m, cap, &nh, stats);
    }
    free(L); free(N); free(tok); free(sh);
    return nh <= cap ? nh : -nh;   /* negative = buffer too small; -return value = hits needed */
}

/* shingles of already-cleaned texts (query side and the invalid-UTF-8 fallback): same as fastscan.c shingle_batch
 * s = one cleaned text (words separated by single spaces, as norm() produces). Writes the xxhash64 seed-42 hashes of
 * all word 3-grams (unsorted, repeats kept) to out and returns their number, or -1 if the text has fewer than
 * 3 words. tok needs room for len/2 + 2 entries (a word is at least one byte plus a space). */
static int64_t shingle_norm(const uint8_t *s, int64_t len, int64_t *tok, uint64_t *out) {
    int64_t nt = 0, i = 0;
    /* word starts: split on single spaces */
    while (i < len) { tok[nt++] = i; while (i < len && s[i] != ' ') i++; i++; }
    if (len == 0) nt = 0;
    if (nt < 3) return -1;
    int64_t n = 0;
    for (int64_t k = 0; k + 2 < nt; k++) { int64_t a = tok[k], b = (k + 3 < nt) ? tok[k + 3] - 1 : len; out[n++] = xxh64(s + a, (size_t)(b - a), 42); }
    return n;
}
/* Shingle SETS of n cleaned texts (data/off = Arrow large_string buffers). Used for the query side (Index in
 * fastscan2.py). Text i's sorted, de-duplicated hashes go to out[out_ptr[i] .. out_ptr[i+1]); counts[i] = set size,
 * or -1 for a text with fewer than 3 words (it then gets no shingles). Returns the total written. */
int64_t shingle_batch(const uint8_t *data, const int64_t *off, int64_t n, uint64_t *out, int64_t *out_ptr, int64_t *counts) {
    int64_t maxlen = 0; for (int64_t i = 0; i < n; i++) if (off[i + 1] - off[i] > maxlen) maxlen = off[i + 1] - off[i];
    int64_t *tok = malloc(sizeof(int64_t) * (size_t)(maxlen / 2 + 4));
    int64_t pos = 0;
    for (int64_t i = 0; i < n; i++) {
        out_ptr[i] = pos;
        int64_t c = shingle_norm(data + off[i], off[i + 1] - off[i], tok, out + pos);
        /* sort and drop repeats in place, so each query is stored as a sorted set */
        if (c > 0) { sort_u64(out + pos, c); int64_t m = 0; for (int64_t k = 0; k < c; k++) if (m == 0 || out[pos + k] != out[pos + m - 1]) out[pos + m++] = out[pos + k]; c = m; pos += c; }
        counts[i] = c;
    }
    out_ptr[n] = pos; free(tok); return pos;
}
/* Scan already-cleaned texts (norm() output; data/off = Arrow large_string buffers). This is the slow-path
 * companion of scan_raw for rows that scan_raw flagged (invalid UTF-8, possible game notice). No length rule and no
 * Bloom pre-check per row: every row with >= 3 words goes to verify_row (stats[0] and stats[1] both count it).
 * Same hit-buffer protocol and return value as scan_raw. */
int64_t scan_norm(const uint8_t *data, const int64_t *off, int64_t n,
                  int64_t *hit_row, int32_t *hit_q, int32_t *hit_inter, int32_t *hit_m, int64_t cap, int64_t *stats) {
    int64_t maxlen = 0; for (int64_t i = 0; i < n; i++) if (off[i + 1] - off[i] > maxlen) maxlen = off[i + 1] - off[i];
    int64_t *tok = malloc(sizeof(int64_t) * (size_t)(maxlen / 2 + 4)); uint64_t *sh = malloc(8 * (size_t)(maxlen / 2 + 4));
    int64_t nh = 0;
    for (int64_t r = 0; r < n; r++) {
        int64_t c = shingle_norm(data + off[r], off[r + 1] - off[r], tok, sh);
        if (c <= 0) continue;
        stats[0]++; stats[1]++;
        verify_row(sh, c, r, hit_row, hit_q, hit_inter, hit_m, cap, &nh, stats);
    }
    free(tok); free(sh);
    return nh <= cap ? nh : -nh;
}

/* All shingle hashes (unsorted, repeats kept) of the RAW rows that pass the length rule, concatenated into out
 * (capacity cap). Used on a SAMPLE of archive rows to learn which query shingles are common in the archive, so the
 * index can prefer shingles that are rare there. Returns the number written, or -1 if cap is too small.
 * Same length rule and cleanup as scan_raw, but no keep mask, no game-notice flag, and rows with invalid UTF-8 are
 * silently skipped (clean_one returns -1, which fails the nt < 3 test). The sample only sets the index ORDER, so
 * skipping rows here cannot change any result. */
int64_t shingle_raw(const uint8_t *data, const int32_t *off32, const int64_t *off64, int64_t n,
                    const uint8_t *valid, int64_t voff, int64_t minchar, uint64_t *out, int64_t cap) {
    int64_t maxlen = 0;
    for (int64_t r = 0; r < n; r++) { int64_t l = off32 ? (int64_t)off32[r + 1] - off32[r] : off64[r + 1] - off64[r]; if (l > maxlen) maxlen = l; }
    uint8_t *L = malloc((size_t)(2 * maxlen + 16)), *N = malloc((size_t)(2 * maxlen + 16));
    int64_t *tok = malloc(sizeof(int64_t) * (size_t)(maxlen + 4));
    int64_t pos = 0;
    for (int64_t r = 0; r < n; r++) {
        if (valid && !((valid[(voff + r) >> 3] >> ((voff + r) & 7)) & 1)) continue;
        int64_t a = off32 ? off32[r] : off64[r], b = off32 ? off32[r + 1] : off64[r + 1];
        if (b - a < minchar) continue;
        const uint8_t *s = data + a; int64_t len = b - a, lo = 0, hi = len;
        while (lo < hi && s[lo] == ' ') lo++;
        while (hi > lo && s[hi - 1] == ' ') hi--;
        int64_t nch = 0; for (int64_t i = lo; i < hi; i++) nch += (s[i] & 0xC0) != 0x80;
        if (nch < minchar) continue;
        int64_t nl = 0, nt = clean_one(s, len, L, N, tok, &nl);
        if (nt < 3) continue;
        if (pos + nt - 2 > cap) { pos = -1; break; }
        for (int64_t k = 0; k + 2 < nt; k++) { int64_t e = (k + 3 < nt) ? tok[k + 3] - 1 : nl; out[pos++] = xxh64(N + tok[k], (size_t)(e - tok[k]), 42); }
    }
    free(L); free(N); free(tok);
    return pos;
}
