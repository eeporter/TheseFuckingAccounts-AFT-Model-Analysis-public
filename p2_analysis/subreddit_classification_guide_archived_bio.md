# Subreddit classification guide — banned/private subreddits with an archived bio (v10)

These subreddits are now banned or private, so Reddit no longer shows their pages, but Arctic Shift's subreddit dump
(`/data/reddit/subreddits_2025-01.zst`) kept what they said about themselves before that. Each input line is one subreddit:
`r/<name>  TITLE: ...  BIO: ...  SIDEBAR: ...` (BIO = public description, SIDEBAR = longer description; either may be
empty, truncated, or in another language).

Assign exactly ONE group, judged by the subreddit's **main stated purpose / what it says it discusses**. Use TITLE, BIO
and SIDEBAR first; use the NAME only to resolve what the text leaves open. If the text gives no usable information about
the topic (e.g. "Welcome!", "Post", a rules list that never says what the subreddit is about) answer `unclear` — the
subreddit then keeps its earlier label, which was based on archived posts.

## Groups (use these exact codes)

- `adult` — sexual/pornographic content or its promotion: nudes, porn, OnlyFans/premium-Snapchat promotion, sexual chat,
  NSFW model/creator promotion, fetish content.
- `crypto` — cryptocurrency: coins/tokens, Bitcoin, Ethereum, altcoins, DeFi, NFTs, blockchain projects, crypto trading,
  mining, airdrops, crypto exchanges/wallets.
- `politics` — politics: parties, ideologies, politicians, elections, government/policy debate, political activism,
  political memes or commentary. A news subreddit with a stated partisan/political focus goes here, not in news.
- `karma_farming` — gaining karma/upvotes or followers: "free karma", "upvote for upvote", karma for new accounts,
  sub4sub / follow4follow exchanges.
- `general` — broad general-interest subreddits not centred on one topic: humour, memes in general, funny or interesting
  pictures/videos, cute animals, general Q&A (AskReddit-style), opinions/discussion, AMAs, general advice.
- `news` — news and current events without a stated partisan focus: world/national/local news, news aggregation, headlines.
- `sales` — buying, selling or promoting for money: marketplaces, items/accounts/services for sale, buy/sell/trade,
  deals, coupons, referral/promo codes, affiliate links, paid services, self-promotion of one's own products, channels or
  content.
- `location` — a specific place: country, region, state, city, town, campus, or a local community defined by place.
- `entertainment` — gaming (video/mobile/tabletop games, esports, streamers of games), sports (sports, teams, leagues,
  athletes, betting talk), TV/film/anime/music (shows, films, anime/manga, cartoons, comics, celebrities, musicians,
  music sharing) and fan communities or topic-specific memes about any of these.
- `other` — anything else: books, podcasts, hobbies, crafts, pets, food, tech/programming, science, education,
  health/mental-health support, identity, religion, relationships, finance (non-crypto), test/gibberish subreddits.
- `unclear` — the text says nothing usable about the topic (see above).

## Tie-breaks
If a subreddit genuinely fits two groups: karma_farming > adult > crypto > sales > politics > news > location >
entertainment > general > other.

## Output
One tab-separated line per input line, in the same order, exactly 4 fields, no header:
`<subreddit name without r/>\t<group code>\t<high|medium|low>\t<reason, at most 12 words, no tabs>`
