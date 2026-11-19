#!/usr/bin/env python3
"""Check that fastscan2's C cleanup gives byte-identical output to fastscan's pyarrow cleanup (norm).
Inputs are generated here (no Reddit data): (a) every Unicode code point, each inside a small fixed context,
(b) random strings mixing ASCII, URL-like pieces, whitespace kinds, cased letters, marks, emoji and random code points.

The cleanup is: lower case, URLs removed, non-word characters -> space, whitespace collapsed, trim (same as the Spark
job). fastscan2.clean_c runs the C version on the raw Arrow buffers; fastscan2.norm is the pyarrow version.
Output: counts, the code points of up to 10 mismatches, and PASS/FAIL on stdout (exit 0 = PASS, 1 = FAIL).
Needs libfastscan2.so built first: fastscan2.py (imported from this script's folder) loads it on import, from
FASTSCAN2_LIB or else fastscan2/libfastscan2.so.
Jobs: no recorded job of its own; it is the "cleanup check" step of run_pilot_fast2_v1.sh (jobs 43324646, 43328586)
and run_rules_check_v1.sh (job 43328587)."""
import random, sys, numpy as np, pyarrow as pa
import fastscan2 as f2
random.seed(20261117)
# every Unicode scalar value (surrogates cannot be in a valid UTF-8 string)
cps = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
tests = []
for c in cps:                                    # (a) 1,112,064 strings
    ch = chr(c)
    tests.append('Ab' + ch + "c'D " + ch + ch + ' x http://q' + ch + 'r' + ch + ' HTTPS://' + ch + ' e')
N_A = len(tests)
pieces = ['http://', 'https://', 'HTTP://', 'Https://', 'http:/', 'hhttp://x', 'https:// ', 'http://\t', ' ', '  ', '\n', '\t', '\x0b', '\x0c', '\r',
          '\x85', '\xa0', ' ', '　', '​', '‌', '‍', '\x1c', '\x00', "'", "''", '_', '-', '.', ',', '!', '*', '[', '](', ')',
          'a', 'B', 'the', 'Word', 'İ', 'ı', 'ß', 'ẞ', 'Σ', 'ς', 'σ', 'K', 'Å', 'Ǆ', 'ǅ', 'é', 'é', 'ͅ', 'Ⓐ', '🄰', '😀', '👍🏽', '中文', 'ｆｕｌｌ',
          '١٢٣', '½', '²', '…', '—', '‘', '’', '“', '”', 'x', '1', '9']
# (b) 300,000 random strings of 0-40 pieces: 80% from the list above, 10% any code point, 10% Latin range
for _ in range(300000):                          # (b)
    k = random.randint(0, 40); s = []
    for _ in range(k):
        r = random.random()
        s.append(random.choice(pieces) if r < 0.8 else chr(random.choice(cps)) if r < 0.9 else chr(random.randint(32, 0x24F)))
    tests.append(''.join(s))
arr = pa.array(tests, type=pa.string())
want = f2.norm(arr).to_pylist()
got = f2.clean_c(arr)
# a mismatch: C cleanup flagged the string as invalid UTF-8 (None), or its bytes differ from the pyarrow result
bad = [i for i in range(len(tests)) if got[i] is None or got[i].decode('utf-8') != want[i]]
print(f'strings tested: {len(tests):,} (all code points: {N_A:,}; random: {len(tests) - N_A:,})')
print(f'mismatches: {len(bad):,}')
for i in bad[:10]:
    print('  first code points of mismatch', i, [hex(ord(x)) for x in tests[i][:12]], repr(want[i][:60]), repr(got[i][:60] if got[i] else None))
print('PASS' if not bad else 'FAIL'); sys.exit(1 if bad else 0)
