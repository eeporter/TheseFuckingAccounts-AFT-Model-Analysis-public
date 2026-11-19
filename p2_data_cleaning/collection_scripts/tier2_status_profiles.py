# WHAT: Tier 2, first pass. Asks Reddit (PRAW) whether each accused account is active, suspended or deleted,
#       and saves the public profile fields of active and suspended accounts.
# READS: data/accused_accounts_all.parquet; Reddit API credentials from the .env file named below (never committed).
# WRITES: data/account_status_checkpoint_full.parquet, data/account_profiles_full.parquet (rewritten every 500 accounts).
# RUN: python tier2_status_profiles.py [limit]. Resumable; seeds from data/account_status_checkpoint.parquet on the first run.
import os, sys, time, pandas as pd, praw, prawcore
from dotenv import load_dotenv

WORK = '/home/eeporter/Dissertation/Reddit_project2_notebooks/p2_data_cleaning'
os.chdir(WORK)
ACCOUNTS   = 'data/accused_accounts_all.parquet'
CHECKPOINT = 'data/account_status_checkpoint_full.parquet'
PROFILES   = 'data/account_profiles_full.parquet'
SEED       = 'data/account_status_checkpoint.parquet'
LIMIT      = int(sys.argv[1]) if len(sys.argv) > 1 else None
EVERY      = 500

# Credentials come from a local .env file outside the repo.
load_dotenv('/home/eeporter/Reddit_project1_notebooks/reddit_env_vars.env')
reddit = praw.Reddit(client_id=os.getenv('praw_script'), client_secret=os.getenv('praw_key'),
                     username=os.getenv('login_name'), password=os.getenv('password'),
                     user_agent=os.getenv('user_agent'), ratelimit_seconds=360)
print('read_only =', reddit.read_only, flush=True)

accounts = pd.read_parquet(ACCOUNTS)

# Resume from the full checkpoint if it exists, otherwise from the earlier small checkpoint.
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

todo = [u for u in accounts.username_key if u not in done_status]
if LIMIT: todo = todo[:LIMIT]
print(f'{len(todo):,} to check this pass ({len(accounts)-len(done_status):,} outstanding overall)', flush=True)

# Profile attributes that are not plain data and are left out.
DROP = {'_reddit', 'subreddit', 'is_blocked', 'is_friend'}

# Save status for every resolved account, and profiles for active/suspended ones.
def save():
    pd.DataFrame({'username_key': list(done_status), 'account_status': list(done_status.values())}) \
      .to_parquet(CHECKPOINT, engine='pyarrow', compression='zstd', index=False)
    if done_profiles:
        pd.DataFrame(done_profiles).to_parquet(PROFILES, engine='pyarrow', compression='zstd', index=False)

t0 = time.time()
# Look up each account; up to 10 tries on rate limits or network errors.
for i, name in enumerate(todo, 1):
    retry = 0
    while retry < 10:
        try:
            info = reddit.redditor(name)
            info._fetch()
            # Suspended accounts still return an object with is_suspended; deleted ones have no id.
            if hasattr(info, 'is_suspended') and info.is_suspended:
                status = 'suspended'
            elif hasattr(info, 'id'):
                status = 'active'
            else:
                status = 'deleted'
            if status in ('active', 'suspended'):
                # Keep only plain public fields (strings, numbers, booleans).
                row = {k: v for k, v in vars(info).items() if not k.startswith('_') and k not in DROP}
                row = {k: v for k, v in row.items() if isinstance(v, (str, int, float, bool, type(None)))}
                row['username_key'] = name
                row['account_status'] = status
                done_profiles.append(row)
            retry = 10
        # 404: the account no longer exists.
        except prawcore.exceptions.NotFound:
            status = 'deleted'; retry = 10
        # 403: shadowbanned or otherwise hidden.
        except prawcore.exceptions.Forbidden:
            status = 'forbidden'; retry = 10
        except prawcore.exceptions.TooManyRequests:
            retry += 1; time.sleep(min(60, 5 * retry))
            if retry >= 10: status = 'error'
        except Exception as e:
            retry += 1; time.sleep(2)
            if retry >= 10: status = 'error'; print(f'  {name}: giving up ({type(e).__name__})', flush=True)
    done_status[name] = status
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
