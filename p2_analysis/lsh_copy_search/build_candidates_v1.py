#!/usr/bin/env python3
"""Build the full copypasta candidate set (2026-11-12): every comment and post of the 987 accounts in the model sample
(sample_998_v4_excl_output, accounts with history_truncated excluded). 92,067 comments + 102,884 posts = 194,951 texts.
Post text = title + newline + selftext. Writes lsh_copy_search/candidates_v1.parquet (data, not committed); it was
streamed to Hyak at /gscratch/comdata/users/eeporter/reddit_copy_check/lsh/candidates_v1.parquet (md5 checked)."""
import pandas as pd
d = "sample_998_v4_excl_output/"
a = pd.read_parquet(d + "sample_accounts.parquet")
keep = set(a.loc[~a.history_truncated, "username_key"])
c = pd.read_parquet(d + "sample_comments.parquet", columns=["id", "outer_key", "author", "created_utc", "subreddit", "body"])
p = pd.read_parquet(d + "sample_posts.parquet", columns=["id", "outer_key", "author", "created_utc", "subreddit", "title", "selftext"])
c = c[c.outer_key.isin(keep)]; p = p[p.outer_key.isin(keep)]
c = c.assign(kind="comment", text=c.body.fillna("")).drop(columns="body")
p = p.assign(kind="post", text=(p.title.fillna("") + "\n" + p.selftext.fillna(""))).drop(columns=["title", "selftext"])
x = pd.concat([c, p], ignore_index=True)
assert (x.kind == "comment").sum() == 92067 and (x.kind == "post").sum() == 102884 and not x.id.duplicated().any()
x.to_parquet("lsh_copy_search/candidates_v1.parquet", index=False)
