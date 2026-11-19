"""Copied comments (#18, wider version, 2026-10-27): prepare the phrase search and check copying inside the pilot.

1. Distinctive pre-accusation comments of the 987 model accounts (capped accounts dropped): text of 60+ characters after
   normalising (lower case, URLs and markdown quote marks removed, whitespace collapsed), one row per account and text
   (earliest use), so an account repeating itself is searched once.
2. Within the pilot: comments whose normalised text matches (exactly, or rapidfuzz ratio >= 90) an EARLIER comment by a
   DIFFERENT pilot account. Seconds, no external queries; catches copying between sample accounts only.
3. Search list for PullPush / Hyak: up to 20 texts per account, longest first.
Writes copy_search_output/ (never overwrites).
"""
import os, sys, re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

# Paths are relative to this script; rules.py from aft_model/ supplies the capped-account rule.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "aft_model"))
import rules
S = os.path.join(HERE, "sample_998_v4_excl_output")
OUT = os.path.join(HERE, "copy_search_output")
# Texts need 60+ characters; near-copy = rapidfuzz ratio >= 90; at most 20 texts per account in the search list.
MIN_CHARS, SIM, PER_ACCOUNT = 60, 90, 20


# Write a file to copy_search_output/, refusing to overwrite one that exists.
def save_new(df, name):
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        raise FileExistsError(path)
    df.to_parquet(path, index=False)
    print("wrote", path, df.shape)


# Normalise text: URLs and markdown quote lines removed, lower case, whitespace collapsed.
def norm(s):
    s = s.fillna("").str.replace(r"https?://\S+", " ", regex=True).str.replace(r"^\s*&gt;.*$", " ", regex=True, flags=re.M)
    return s.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


# --- 1. Distinctive pre-accusation comments of the model accounts (capped accounts dropped) ---
a_full = pd.read_parquet(os.path.join(S, "sample_accounts.parquet")).set_index("username_key")
posts = pd.read_parquet(os.path.join(S, "sample_posts.parquet"), columns=["id", "outer_key", "created_utc"])
c = pd.read_parquet(os.path.join(S, "sample_comments.parquet"), columns=["id", "outer_key", "created_utc", "subreddit", "link_id", "body"])
acc = a_full.drop(rules.capped_accounts(a_full, posts, c))
c = c.drop_duplicates(["outer_key", "id"])
c = c[c.outer_key.isin(acc.index) & (c.created_utc < c.outer_key.map(acc.first_mention_adj_utc))].copy()
c = c[~c.body.isin(["[removed]", "[deleted]"])]
c["text"] = norm(c.body)
c = c[c.text.str.len() >= MIN_CHARS].sort_values("created_utc")
print("distinctive pre-accusation comments", len(c), "accounts", c.outer_key.nunique())
u = c.drop_duplicates(["outer_key", "text"]).reset_index(drop=True)          # earliest use of each text per account
u["n_uses"] = u.set_index(["outer_key", "text"]).index.map(c.groupby(["outer_key", "text"]).size())
print("unique (account, text)", len(u), "unique texts", u.text.nunique())

# 2. copying inside the pilot: an earlier comment by a different pilot account with the same or near-same text
texts = u.text.tolist()
t = u.created_utc.to_numpy(); who = u.outer_key.to_numpy()
rows = []
# Compare texts in blocks of 2,000 against all texts; keep pairs at or above the similarity cut-off.
for i0 in range(0, len(texts), 2000):
    i1 = min(i0 + 2000, len(texts))
    sc = process.cdist(texts[i0:i1], texts, scorer=fuzz.ratio, score_cutoff=SIM, dtype=np.uint8, workers=-1)
    for i, j in zip(*np.nonzero(sc)):
        i += i0
        if who[i] != who[j] and t[j] < t[i]:
            rows.append({"outer_key": who[i], "id": u.id[i], "created_utc": t[i], "orig_account": who[j], "orig_id": u.id[j],
                         "orig_utc": t[j], "score": int(sc[i - i0, j]), "same_thread": u.link_id[i] == u.link_id[j]})
inside = pd.DataFrame(rows, columns=["outer_key", "id", "created_utc", "orig_account", "orig_id", "orig_utc", "score", "same_thread"])
inside = inside.sort_values("orig_utc").drop_duplicates(["outer_key", "id"])                    # earliest original
print("within-pilot copies:", len(inside), "comments by", inside.outer_key.nunique(), "accounts;",
      "from", inside.orig_account.nunique(), "source accounts; same thread", int(inside.same_thread.sum()))
save_new(inside, "within_pilot_copies_v1.parquet")

# 3. search list: up to PER_ACCOUNT texts per account, longest first
u["len"] = u.text.str.len()
q = u.sort_values(["outer_key", "len"], ascending=[True, False]).groupby("outer_key").head(PER_ACCOUNT)
print("search list", len(q), "texts for", q.outer_key.nunique(), "accounts; per account",
      q.groupby("outer_key").size().describe()[["min", "50%", "max"]].to_dict())
save_new(u.drop(columns="len"), "distinctive_comments_v1.parquet")
save_new(q.drop(columns="len"), "search_list_v1.parquet")
