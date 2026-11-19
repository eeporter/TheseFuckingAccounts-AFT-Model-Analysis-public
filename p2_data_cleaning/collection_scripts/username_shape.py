# WHAT: helper module. looks_like_username(token, freq) decides whether a bare word (no u/ prefix) is a username.
#       freq = how often each lower-cased word appears as ordinary prose in the corpus.
"""Shared test for 'this bare token is a username, not prose'."""
import re
SHAPED=re.compile(r'[0-9_-]|[a-z][A-Z]')      # digit, underscore, hyphen or internal capital
# 3-20 letters, digits, underscores or hyphens: Reddit's username alphabet.
TOKEN=re.compile(r'^[A-Za-z0-9_-]{3,20}$')

def looks_like_username(tok, freq, rare_max=3):
    if not TOKEN.match(tok): return False
    k=tok.lower()
    if freq.get(k,0) >= rare_max: return False        # recurs as ordinary prose
    if not SHAPED.search(tok): return False           # plain word, no username shape
    if k.isdigit(): return False                      # "277", "000" - numeric fragments
    if re.match(r'^r[_/]', k): return False           # r_HotDealsTrend - a subreddit, not a user
    # "copy-paste", "karma-farming": a HYPHENATED English compound whose every part is common
    # prose. Restricted to hyphen-only tokens on purpose - underscore compounds are a normal
    # username style ("well_that_example_went", "Mr_Example__Wild_Ride") and must survive.
    if '-' in k and '_' not in k and not any(c.isdigit() for c in k):
        parts=[p for p in re.split(r'-+',k) if p.isalpha() and len(p)>2]
        if parts and all(freq.get(p,0)>=rare_max for p in parts): return False
    return True
