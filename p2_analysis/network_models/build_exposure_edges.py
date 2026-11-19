"""Step 2 of the network models: reply edges between the 757 AFT accounts and everyone they talked to, with the
partner's accusation date and status, so an "exposure to accused accounts" covariate can be proposed later.

Sample: the AFT model's decided sample: all capped accounts dropped (rules.capped_accounts, 11) and the 1-year window
(rules.in_window). Checked against the account ids in aft_model_output/exact_v8_to_cutoff.parquet. aft_model/ is only
imported, never written.

Edges (one row per reply event, created_utc < rules.CUTOFF):
  out  comment_parent  the account's comment -> author of its parent post/comment (parent_authors_output/parts).
                       Parent authors were collected for pre-accusation comments only.
  in   comment_reply   a direct reply to one of the account's comments (replies_output/comments_parts)
  in   post_thread     any comment in the thread of one of the account's posts (replies_output/posts_parts);
                       direct_reply marks the ones whose parent is the account's own post or comment.
A comment found in both inbound stores is kept once, as comment_reply.

Partner attributes come from p2_data_cleaning/data: accused = in accused_accounts_final (74,812). The accusation date
is first_mention_adj_utc from the eligible pool (or sample) where available, else first_mention_utc from
accused_accounts_final (partner_acc_date_source says which). Status comes from account_status_checkpoint_full.

Filtering follows the ego-network rules (network_case_CaseB_ego_network.ipynb): [deleted]/[removed] out; self-replies out;
bots out = the public Pushshift bot list (~/redditdumps/PushshiftDumps/scripts/ignored.txt) + AutoModerator + ExampleMirrorBot, the name
rule (ends in bot/b0t[_-]digits, not robot), and the text rule (>= 50% of a partner's texts in these data self-identify
as a bot). Accounts PRAW reports as deleted are kept, with partner_status as an attribute. Thread co-presence in
megathreads is not a tie: non-direct post_thread rows from threads with more than THREAD_MAX_COMMENTS collected comments
are dropped (direct replies are kept). Contacts = distinct threads (link_id), kept as link_id on every edge.

Writes to p2_analysis/network_models_output/ (the *_output/ convention, git-ignored) and never overwrites.
"""
import os, sys, glob, re
import numpy as np
import pandas as pd

# Paths: P2 = p2_analysis/, S = the 998-account pilot, D = p2_data_cleaning/data/. aft_model/rules.py is imported, never changed.
HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(P2, 'aft_model'))
import rules  # noqa: E402

S = os.path.join(P2, 'sample_998_v4_excl_output')
D = os.path.join(os.path.dirname(P2), 'p2_data_cleaning', 'data')
OUT = os.path.join(P2, 'network_models_output')
DAY = 86400.0
os.makedirs(OUT, exist_ok=True)

# Bot rules: a published bot list, extra known bots, a name rule and a text rule (see the docstring).
BOT_LIST_PATH = '/home/eeporter/redditdumps/PushshiftDumps/scripts/ignored.txt'
EXTRA_BOTS = {'AutoModerator', 'ExampleMirrorBot'}
NOT_BOTS = set()
BOT_NAME_RULE = re.compile(r'(?<!ro)(bot|b0t)[_-]?\d*$', re.IGNORECASE)
BOT_TEXT_RULE = (r"(?i)\bi(?: am|'m|m) an? (?:bot|automated)\b|\^+i'?m \^+a \^+bot|performed automatically"
                 r"|self-deletes|beep,? boop|\[\^*\[?(?:code|source(?: code)?|github|faq|feedback|issues"
                 r"|about this bot|info)\]?\]\((?:https?://)?(?:www\.)?(?:github\.com|reddit\.com/r/\w*bot)")
BOT_TEXT_SHARE = 0.5
THREAD_MAX_COMMENTS = 2000
NON_AUTHORS = {'[deleted]', '[removed]', 'none', 'nan', ''}


# Write to network_models_output/, refusing to overwrite.
def save_new(df, name):
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        raise FileExistsError(path)
    if name.endswith('.parquet'):
        df.to_parquet(path, index=False, engine='pyarrow', compression='zstd')
    else:
        df.to_csv(path, index=False)
    return path


# Read and stack the part files matching a pattern under p2_analysis/, skipping empty ones.
def parts(pattern, cols):
    return pd.concat([d for d in (pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(os.path.join(P2, pattern))))
                      if len(d)], ignore_index=True)


# --- sample -------------------------------------------------------------------------------------------------------
acc = pd.read_parquet(os.path.join(S, 'sample_accounts.parquet')).set_index('username_key')
posts_all = pd.read_parquet(os.path.join(S, 'sample_posts.parquet'), columns=['id', 'outer_key', 'created_utc'])
coms_all = pd.read_parquet(os.path.join(S, 'sample_comments.parquet'),
                           columns=['id', 'parent_id', 'link_id', 'created_utc', 'subreddit', 'outer_key'])
capped = rules.capped_accounts(acc, posts_all, coms_all)
kept = acc.drop(index=capped)
sample = pd.Index(sorted(rules.in_window(kept, rules.HORIZON_DAYS)))
# Check that the sample equals the accounts of the decided AFT build (exact_v8).
v8 = set(pd.read_parquet(os.path.join(P2, 'aft_model_output', 'exact_v8_to_cutoff.parquet'), columns=['id']).id)
assert set(rules.in_window(acc.loc[sorted(v8)], rules.HORIZON_DAYS)) == set(sample)
own_acc = acc['first_mention_adj_utc']
log = {'capped_dropped': len(capped), 'after_capped': len(kept), 'sample_757': len(sample),
       'sample_status': acc.loc[sample, 'account_status'].value_counts().to_dict()}
print(log, flush=True)

# --- edges ----------------------------------------------------------------------------------------------------------
# Outbound edges: each pre-cutoff comment of a sample account, to the author of the post or comment it replied to.
pa = parts('parent_authors_output/parts/*.parquet', ['parent_id', 'parent_author', 'returned']).drop_duplicates('parent_id')
ptext = parts('parent_text_output/parts/*.parquet', ['parent_id', 'parent_title', 'parent_text']).drop_duplicates('parent_id')
ptext['text'] = ptext.parent_title.fillna('') + ' ' + ptext.parent_text.fillna('')
pa = pa.merge(ptext[['parent_id', 'text']], on='parent_id', how='left')
c = coms_all[coms_all.outer_key.isin(sample)]
log['out_candidate_comments'] = len(c)
c = c.merge(pa, on='parent_id', how='left')
log['out_comments_without_parent_author'] = int(c.parent_author.isna().sum())
out = pd.DataFrame({'sample_account': c.outer_key, 'partner_raw': c.parent_author, 'direction': 'out',
                    'source': 'comment_parent', 'created_utc': c.created_utc, 'subreddit': c.subreddit,
                    'item_id': c.id, 'link_id': c.link_id, 'direct_reply': True, 'text': c.text})
out = out[out.partner_raw.notna()]

# Inbound edges: replies to the account's comments (comment_reply) and comments in its post threads (post_thread).
cols = ['id', 'parent_id', 'link_id', 'author', 'created_utc', 'subreddit', 'body', 'source_id', 'outer_key']
rc = parts('replies_output/comments_parts/replies_*.parquet', cols)
rp = parts('replies_output/posts_parts/replies_*.parquet', cols)
rc, rp = rc[rc.outer_key.isin(sample)], rp[rp.outer_key.isin(sample)]
own_items = set('t1_' + coms_all.id) | set('t3_' + posts_all.id)
rc = rc.drop_duplicates(['outer_key', 'id'])
rp = rp.drop_duplicates(['outer_key', 'id'])
n_rp = len(rp)
# A row in both collections is kept once, as comment_reply.
rp = rp[~rp.set_index(['outer_key', 'id']).index.isin(rc.set_index(['outer_key', 'id']).index)]
log['in_post_thread_rows_also_in_comment_reply'] = n_rp - len(rp)
# direct = a post_thread comment whose parent is the account's own post or comment.
rp['direct'] = rp.parent_id.isin(own_items)
# Threads with more than THREAD_MAX_COMMENTS collected comments: drop their non-direct rows (co-presence in megathreads).
tsize = rp.groupby('link_id').id.size()
big = set(tsize.index[tsize > THREAD_MAX_COMMENTS])
m = rp.link_id.isin(big) & ~rp.direct
log['megathreads_over_cap'] = len(big)
log['removed_megathread_copresence'] = int(m.sum())
rp = rp[~m]


# Inbound rows in the shared edge format.
def inbound(r, source):
    return pd.DataFrame({'sample_account': r.outer_key, 'partner_raw': r.author, 'direction': 'in', 'source': source,
                         'created_utc': r.created_utc, 'subreddit': r.subreddit, 'item_id': r.id,
                         'link_id': r.link_id, 'direct_reply': r.direct if source == 'post_thread' else True,
                         'text': r.body})


e = pd.concat([out, inbound(rc, 'comment_reply'), inbound(rp, 'post_thread')], ignore_index=True)
e['created_utc'] = e.created_utc.astype('int64')
log['raw_edges_by_source'] = e.source.value_counts().to_dict()
n0 = len(e)
# Only events before the pool's cutoff date.
e = e[e.created_utc < rules.CUTOFF]
log['removed_after_cutoff'] = n0 - len(e)

# --- filters --------------------------------------------------------------------------------------------------------
e['partner'] = e.partner_raw.astype(str).str.lower()
rm = {}
# Drop deleted/removed authors and self-replies.
m = e.partner_raw.isna() | e.partner.isin(NON_AUTHORS)
rm['deleted_or_removed'] = int(m.sum()); e = e[~m]
m = e.partner == e.sample_account
rm['self'] = int(m.sum()); e = e[~m]
known = {l.strip().lower() for l in open(BOT_LIST_PATH) if l.strip()} | {b.lower() for b in EXTRA_BOTS}
not_bots = {b.lower() for b in NOT_BOTS}
# Text rule: share of a partner's texts in these data that self-identify as a bot.
tx = e[['partner', 'text']].dropna()
share = tx.text.str.contains(BOT_TEXT_RULE).groupby(tx.partner).mean()
text_bots = set(share.index[share >= BOT_TEXT_SHARE])


# Which bot rule removes a partner, or None.
def bot_reason(n):
    if n in not_bots:
        return None
    if n in known:
        return 'known list'
    if BOT_NAME_RULE.search(n):
        return 'name rule'
    if n in text_bots:
        return 'text rule'
    return None


e['bot_reason'] = e.partner.map({n: bot_reason(n) for n in e.partner.unique()})
bots = (e[e.bot_reason.notna()].groupby(['partner', 'bot_reason'])
        .agg(rows=('item_id', 'size'), sample_accounts=('sample_account', 'nunique')).reset_index()
        .sort_values('rows', ascending=False))
save_new(bots, 'removed_bots.csv')
for r_, n_ in e.bot_reason.value_counts().items():
    rm['bot_' + r_.replace(' ', '_')] = int(n_)
log['bot_accounts_by_reason'] = bots.bot_reason.value_counts().to_dict()
log['bot_name_and_text_rule_top'] = bots[bots.bot_reason != 'known list'].head(25).set_index('partner').rows.to_dict()
e = e[e.bot_reason.isna()].drop(columns=['bot_reason', 'text'])
log['removed'] = rm
log['edges_kept_by_source'] = e.source.value_counts().to_dict()

# --- partner attributes -----------------------------------------------------------------------------------------------
# Partner attributes: accused or not, accusation date (edit-adjusted when known), and PRAW status.
acf = pd.read_parquet(os.path.join(D, 'accused_accounts_final.parquet'), columns=['username_key', 'first_mention_utc'])
acf = acf.drop_duplicates('username_key').set_index('username_key').first_mention_utc
pool = pd.read_parquet(os.path.join(S, 'eligible_pool.parquet')).drop_duplicates('username_key').set_index('username_key')
adj = pd.concat([pool.first_mention_adj_utc, acc.first_mention_adj_utc])
adj = adj[~adj.index.duplicated(keep='last')]
status = pd.read_parquet(os.path.join(D, 'account_status_checkpoint_full.parquet')).drop_duplicates('username_key')
status = status.set_index('username_key').account_status

e['partner_accused'] = e.partner.isin(acf.index)
e['partner_in_pool'] = e.partner.isin(pool.index)
e['partner_in_sample'] = e.partner.isin(sample)
e['partner_acc_utc'] = e.partner.map(adj).fillna(e.partner.map(acf))
e['partner_acc_date_source'] = np.select([e.partner.isin(adj.index), e.partner_accused],
                                         ['first_mention_adj_utc', 'first_mention_utc'], None)
e['partner_status'] = e.partner.map(status)
# Timing flags relative to the sample account's own accusation and the time of the reply.
e['own_acc_utc'] = e.sample_account.map(own_acc)
e['pre_own_accusation'] = e.created_utc < e.own_acc_utc
e['partner_accused_at_reply'] = e.partner_acc_utc < e.created_utc
e['partner_accused_before_own'] = e.partner_acc_utc < e.own_acc_utc
e['prior_exposure'] = e.pre_own_accusation & e.partner_accused_at_reply
e['partner_minus_own_days'] = (e.partner_acc_utc - e.own_acc_utc) / DAY
e['sample_status'] = e.sample_account.map(acc.account_status)
e = e.drop(columns='partner_raw').sort_values(['sample_account', 'created_utc']).reset_index(drop=True)
save_new(e, 'edges_757.parquet')
log['edges_rows'] = len(e)
log['edges_pre_own_accusation'] = int(e.pre_own_accusation.sum())
log['edges_to_accused'] = int(e.partner_accused.sum())
log['edges_prior_exposure'] = int(e.prior_exposure.sum())
log['partner_acc_date_source'] = e[e.partner_accused].drop_duplicates('partner').partner_acc_date_source.value_counts().to_dict()
log['top_partners_by_sample_accounts'] = e.groupby('partner').sample_account.nunique().sort_values(ascending=False).head(30).to_dict()
# Counts from every step go to build_log.csv.
pd.Series({k: str(v) for k, v in log.items()}).to_csv(os.path.join(OUT, 'build_log.csv'), header=['value'])
for k, v in log.items():
    print(k, ':', v, flush=True)
