#!/usr/bin/env python3
"""Build the full copypasta candidate set v2 (2026-11-18): every comment and post of ALL accounts in the eligible pool
(sample_998_v4_excl_output/eligible_pool.parquet, steps 1-8, 38,334 accounts) minus rule B (history truncated at the
10k cap), i.e. 38,106 accounts. No sampling. Same columns and post text rule as build_candidates_v1.py
(post text = title + newline + selftext). Histories are read from p2_data_cleaning/data/arctic_{comments,posts} as in
sample_1000_v4_random_eligible.ipynb (file by file, rows deduped on (outer_key, id)).
Writes (never overwrites): candidates_v2.parquet and accounts_dates_v2.parquet (username_key, first_mention_adj_utc).
Data, not committed. Prints counts only.

Run on bia on 2026-11-18 (not a Slurm job). Paths are relative to the folder it is run from (it also reads
candidates_v1.parquet there, only to count how many v1 texts are also in v2). The outputs are the whole-pool query
set; run_full_v3.sh (array 43340006) reads that set from lsh_v2/ on Hyak. rules2.Queries reads the file names
candidates_v1.parquet and sample_accounts_dates_v1.parquet, so the copy into lsh_v2/ (not done by this script) has
to use those names.
Note: no text filtering here; the copy search applies its own query rules (pre2022, minchar, link rule) later."""
import glob, os
import pandas as pd
D = "../../p2_data_cleaning/data"
pool = pd.read_parquet("../sample_998_v4_excl_output/eligible_pool.parquet")
assert len(pool) == 38334 and pool.username_key.is_unique
# rule B: drop accounts whose history hit the 10k cap (truncated)
acc = pool[~pool.history_truncated]
assert len(acc) == 38106
keys = set(acc.username_key)

def read_history(kind, columns):
    """All rows of arctic_<kind>/*.parquet whose outer_key (the account the history belongs to) is in the pool,
    read file by file with a parquet filter; rows deduplicated on (outer_key, id)."""
    parts = [pd.read_parquet(f, columns=columns, filters=[("outer_key", "in", list(keys))])
             for f in sorted(glob.glob(f"{D}/arctic_{kind}/*.parquet"))]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    return df.drop_duplicates(["outer_key", "id"]).reset_index(drop=True)

c = read_history("comments", ["id", "outer_key", "author", "created_utc", "subreddit", "body"])
p = read_history("posts", ["id", "outer_key", "author", "created_utc", "subreddit", "title", "selftext"])
c = c.assign(kind="comment", text=c.body.fillna("")).drop(columns="body")
p = p.assign(kind="post", text=(p.title.fillna("") + "\n" + p.selftext.fillna(""))).drop(columns=["title", "selftext"])
# one table of all texts; kind = "comment" / "post"
x = pd.concat([c, p], ignore_index=True)
v1 = pd.read_parquet("candidates_v1.parquet", columns=["kind", "id", "outer_key"])
print("accounts in pool after rule B:", len(keys), " accounts with any text:", x.outer_key.nunique())
print("comments:", int((x.kind == "comment").sum()), " posts:", int((x.kind == "post").sum()), " total:", len(x))
print("duplicate (kind, id) across accounts:", int(x.duplicated(["kind", "id"]).sum()))
print("v1 texts also in v2:", int(v1.set_index(["kind", "id"]).index.isin(x.set_index(["kind", "id"]).index).sum()), "of", len(v1))
# refuse to overwrite existing outputs
for f in ("candidates_v2.parquet", "accounts_dates_v2.parquet"):
    assert not os.path.exists(f), f"exists, not overwritten: {f}"
x.to_parquet("candidates_v2.parquet", index=False)
acc[["username_key", "first_mention_adj_utc"]].to_parquet("accounts_dates_v2.parquet", index=False)
print("wrote candidates_v2.parquet", x.shape, "and accounts_dates_v2.parquet", (len(acc), 2))
