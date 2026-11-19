#!/usr/bin/env python3
"""Combine hits_parts/ from copy_search_hyak_v1.py into
  hyak_copy_hits_v1.parquet      one row per (searched comment, matching archive comment), not the searched comment itself
  hyak_copy_by_comment_v1.parquet one row per searched comment (all 2,162 in search_list_v1)
A comment counts as COPIED if an archive comment by a DIFFERENT, non-deleted author containing its query was posted at or
before it (hit_utc <= created_utc). Later matches by other authors are counted separately (others copying it).

Older script (12 Nov 2026), second step after copy_search_hyak_v1.py (array 43145502). Run with the working directory
holding hits_parts/ and search_list_v1.parquet (all paths are relative). No Slurm job for this step is recorded in the
code; how and where it was run is not evident from the code.
Inputs: hits_parts/file_*.parquet (the script asserts there are exactly 1,400 of them, one per archive file) and
  search_list_v1.parquet (id, outer_key, created_utc, subreddit, link_id).
Outputs: the two parquet files above; prints summary counts only.
Notes: "earlier" here is at-or-before (<=), and hit_utc is CreatedAt as stored, converted to seconds with no +8 h shift
(unlike the later rules2.py). Same author = lower-cased archive author equals the query's outer_key. Deleted author =
"[deleted]" or missing.
"""
import glob
import numpy as np
import pandas as pd

# ---- step 1: all hits, minus self-matches, with flags -> hyak_copy_hits_v1.parquet
parts = [pd.read_parquet(f) for f in sorted(glob.glob("hits_parts/file_*.parquet"))]
assert len(parts) == 1400, len(parts)
h = pd.concat([d for d in parts if len(d)], ignore_index=True)
h["is_self"] = h.is_self.astype(bool)
h = h.drop_duplicates(["id", "hit_id"])
q = pd.read_parquet("search_list_v1.parquet", columns=["id", "outer_key", "created_utc", "subreddit", "link_id"])
# searched comments that found themselves in the archive (a check that matching works); then drop those rows
found_self = set(h.loc[h.is_self, "id"])
h = h[~h.is_self].copy()
# CreatedAt (timestamp, nanoseconds as int64) -> seconds; no time-zone shift applied
h["hit_utc"] = h.hit_CreatedAt.astype("int64") // 10**9
h["hit_author_key"] = h.hit_author.fillna("").str.lower()
h["same_author"] = h.hit_author_key == h.outer_key
h["deleted_author"] = h.hit_author_key.isin(["[deleted]", ""])
h["earlier"] = h.hit_utc <= h.created_utc
h["gap_days"] = (h.created_utc - h.hit_utc) / 86400
h = h.merge(q[["id", "outer_key", "link_id"]].rename(columns={"link_id": "q_link_id"}), on=["id", "outer_key"], how="left")
# same_thread: the archive comment is in the same submission (link_id) as the searched comment
h["same_thread"] = h.hit_link_id == h.q_link_id
h = h.drop(columns=["is_self", "hit_CreatedAt", "q_link_id"])
h.to_parquet("hyak_copy_hits_v1.parquet", index=False)

# ---- step 2: one row per searched comment -> hyak_copy_by_comment_v1.parquet
# "other" = hits by a different, non-deleted author; src = the earlier ones, sorted so the first per id is the earliest
other = h[~h.same_author & ~h.deleted_author]
src = other[other.earlier].sort_values("hit_utc")
g_src = src.groupby("id")
first = g_src.first()
c = q.copy()
c["found_self"] = c.id.isin(found_self)
c["n_earlier_other"] = c.id.map(g_src.size()).fillna(0).astype(int)
c["n_earlier_other_authors"] = c.id.map(g_src.hit_author_key.nunique()).fillna(0).astype(int)
c["n_later_other"] = c.id.map(other[~other.earlier].groupby("id").size()).fillna(0).astype(int)
c["n_earlier_deleted"] = c.id.map(h[h.deleted_author & h.earlier].groupby("id").size()).fillna(0).astype(int)
c["n_same_author_other_ids"] = c.id.map(h[h.same_author].groupby("id").size()).fillna(0).astype(int)
c["copied"] = c.n_earlier_other > 0
# first_src_*: fields of the earliest earlier hit by another author (groupby.first takes, per column, the first
# non-missing value in that order)
for col in ["hit_id", "hit_author", "hit_subreddit", "hit_utc", "gap_days", "same_thread", "hit_len"]:
    c["first_src_" + col.replace("hit_", "")] = c.id.map(first[col])
c.to_parquet("hyak_copy_by_comment_v1.parquet", index=False)

# summary counts only
print("hits", len(h), "| searched", len(c), "| self found", c.found_self.sum())
print("copied comments", c.copied.sum(), "accounts with >=1 copied", c[c.copied].outer_key.nunique(), "of", c.outer_key.nunique())
print("n_earlier_other quantiles among copied", c.loc[c.copied, "n_earlier_other"].describe().round(1).to_dict())
print("first source same thread", c.first_src_same_thread.sum(), "| gap days median", c.first_src_gap_days.median())
print("comments others copied later", (c.n_later_other > 0).sum())
acc = c.groupby("outer_key").agg(n=("id", "size"), copied=("copied", "sum"))
print("share of accounts with any copy", round((acc.copied > 0).mean(), 3), "| copied share of comments", round(c.copied.mean(), 3))
