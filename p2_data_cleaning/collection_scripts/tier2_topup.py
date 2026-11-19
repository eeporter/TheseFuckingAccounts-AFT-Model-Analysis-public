# Tier 2 of the account collection, top-up for the final account list: PRAW account status (active / suspended / deleted / forbidden)
# and the full public profile for every accused account. Resumable: progress is checkpointed every EVERY lookups.
# Reads: data/accused_accounts_final.parquet; writes data/account_status_checkpoint_full.parquet and data/account_profiles_full.parquet.
# Run (on bia): python tier2_topup.py [LIMIT]   LIMIT = optional cap on lookups this run.
import os, sys, time, pandas as pd, praw, prawcore
from dotenv import load_dotenv

# Paths are relative to p2_data_cleaning/.
WORK = '/home/eeporter/Dissertation/Reddit_project2_notebooks/p2_data_cleaning'
os.chdir(WORK)
ACCOUNTS   = 'data/accused_accounts_final.parquet'
CHECKPOINT = 'data/account_status_checkpoint_full.parquet'
PROFILES   = 'data/account_profiles_full.parquet'
SEED       = 'data/account_status_checkpoint.parquet'
LIMIT      = int(sys.argv[1]) if len(sys.argv) > 1 else None
EVERY      = 500

# PRAW credentials come from the .env file; nothing is written to Reddit.
load_dotenv('/home/eeporter/Reddit_project1_notebooks/reddit_env_vars.env')
reddit = praw.Reddit(client_id=os.getenv('praw_script'), client_secret=os.getenv('praw_key'),
                     username=os.getenv('login_name'), password=os.getenv('password'),
                     user_agent=os.getenv('user_agent'), ratelimit_seconds=360)
print('read_only =', reddit.read_only, flush=True)

accounts = pd.read_parquet(ACCOUNTS)

# Resume from the full checkpoint if it exists, else from the 1,000-account seed checkpoint.
done_status, done_profiles = {}, []
if os.path.exists(CHECKPOINT):
    prev = pd.read_parquet(CHECKPOINT)
    done_status = dict(zip(prev.username_key, prev.account_status))
    if os.path.exists(PROFILES):
        done_profiles = pd.read_parquet(PROFILES).to_dict('records')
elif os.path.exists(SEED):
    prev = pd.read_parquet(SEED)
    done_status = dict(zip(prev.username_key, prev.account_status))
print(f'resuming: {len(done_status):,} already resolved', flush=True)

# accounts with no status yet, PLUS accounts that have a status but never had their
# 26-field profile captured (the 1,000 seed rows resolved by the original notebook run)
have_profile = {r['username_key'] for r in done_profiles}
need_status  = [u for u in accounts.username_key if u not in done_status]
need_profile = [u for u in accounts.username_key
                if u in done_status and done_status[u] in ('active','suspended')
                and u not in have_profile]
todo = need_status + need_profile
print(f'  {len(need_status):,} need status, {len(need_profile):,} need a profile top-up', flush=True)
if LIMIT: todo = todo[:LIMIT]
print(f'{len(todo):,} to check this pass ({len(accounts)-len(done_status):,} outstanding overall)', flush=True)

# Profile fields not kept (objects or irrelevant flags).
DROP = {'_reddit', 'subreddit', 'is_blocked', 'is_friend'}

# Write the status checkpoint and the profiles table.
def save():
    pd.DataFrame({'username_key': list(done_status), 'account_status': list(done_status.values())}) \
      .to_parquet(CHECKPOINT, engine='pyarrow', compression='zstd', index=False)
    if done_profiles:
        pd.DataFrame(done_profiles).to_parquet(PROFILES, engine='pyarrow', compression='zstd', index=False)

t0 = time.time()
# Look up each account; retry up to 10 times on rate limits or network errors.
for i, name in enumerate(todo, 1):
    retry = 0
    while retry < 10:
        try:
            info = reddit.redditor(name)
            # Force the fetch so suspension / missing id is known now.
            info._fetch()
            if hasattr(info, 'is_suspended') and info.is_suspended:
                status = 'suspended'
            elif hasattr(info, 'id'):
                status = 'active'
            else:
                status = 'deleted'
            # Keep the profile (plain values only) for active and suspended accounts.
            if status in ('active', 'suspended'):
                row = {k: v for k, v in vars(info).items() if not k.startswith('_') and k not in DROP}
                row = {k: v for k, v in row.items() if isinstance(v, (str, int, float, bool, type(None)))}
                row['username_key'] = name
                row['account_status'] = status
                done_profiles.append(row)
            retry = 10
        except prawcore.exceptions.NotFound:
            status = 'deleted'; retry = 10
        except prawcore.exceptions.Forbidden:
            status = 'forbidden'; retry = 10
        except prawcore.exceptions.TooManyRequests:
            retry += 1; time.sleep(min(60, 5 * retry))
            if retry >= 10: status = 'error'
        except Exception as e:
            retry += 1; time.sleep(2)
            if retry >= 10: status = 'error'; print(f'  {name}: giving up ({type(e).__name__})', flush=True)
    done_status[name] = status
    # Checkpoint and progress line every EVERY accounts.
    if i % EVERY == 0 or i == len(todo):
        save()
        el = time.time() - t0
        rate = i / el
        print(f'  {i:,}/{len(todo):,} | {rate:.1f}/s | {len(done_status):,} total resolved | '
              f'~{(len(todo)-i)/rate/3600:.1f} h left', flush=True)

save()
s = pd.Series(done_status)
print('\nfinal counts:\n' + s.value_counts().to_string(), flush=True)
print(f'total resolved: {len(done_status):,} of {len(accounts):,}', flush=True)
