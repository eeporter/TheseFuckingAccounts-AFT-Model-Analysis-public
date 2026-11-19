# Subreddit classification guide (v4 sample, live non-NSFW subreddits)

Each input line is one subreddit: `r/<name> TITLE: ... BIO: ... SIDEBAR: ...` (TITLE/BIO/SIDEBAR are what the subreddit says
about itself on Reddit; they may be empty, truncated, or in another language).

Assign exactly ONE group to every subreddit, judged by the subreddit's **main stated purpose / what it says it discusses**.
Use TITLE, BIO and SIDEBAR first. Only if they give no usable information, classify from the subreddit NAME — and only if
the name clearly fits a group; otherwise use `other`.

## Groups (use these exact codes)

- `crypto` — cryptocurrency: coins/tokens, Bitcoin, Ethereum, altcoins, DeFi, NFTs, blockchain projects, crypto trading,
  mining, airdrops, crypto exchanges/wallets.
- `politics` — politics: parties, ideologies, politicians, elections, government/policy debate, political activism,
  political memes or political commentary. A news subreddit whose stated focus is political (e.g. a party, ideology or
  partisan viewpoint) goes here, not in news.
- `karma_farming` — subreddits whose stated purpose is gaining karma/upvotes: "free karma", "upvote for upvote", karma
  for new accounts, "get karma to post elsewhere", follow-for-follow style exchange.
- `general` — broad, general-interest subreddits not centred on one specific topic: humour, memes in general, funny
  pictures/videos/gifs, cute animals, "interesting"/"satisfying"/"wholesome" content, general pics/videos, general
  questions and answers (AskReddit-style), opinions/discussion (Showerthoughts, unpopularopinion), AMAs, general advice.
- `news` — news and current events in general: world/national/local news, news aggregation, journalism, headlines,
  breaking news, without a stated partisan/political focus.
- `sales` — buying, selling or promoting for money: marketplaces, items/services for sale, buy/sell/trade/swap of goods
  or accounts, deals, coupons, discounts, referral codes, promo codes, affiliate links, sponsorship/advertising-for-hire,
  freelancing/hiring-for-pay marketplaces.
- `location` — a specific place: a country, region, state, city, town, neighbourhood, university campus town, or a local
  community defined by place.
- `other` — anything that does not fit the groups above. This includes subreddits centred on one specific topic,
  hobby, game, sport, team, fandom, TV show/film, music artist, profession, school subject, health/mental-health support,
  identity group, technology product, or a topic-specific meme subreddit (e.g. memes about one game or show).

## Tie-breaks
If a subreddit genuinely fits two groups, choose in this order of priority:
karma_farming > crypto > sales > politics > news > location > general > other.
(Example: a city's buy/sell subreddit is `sales`; a country's political party subreddit is `politics`; a crypto
referral-code subreddit is `crypto`.)

## Output
One tab-separated line per input line, in the same order, with exactly these 5 fields and no header:
`<subreddit name without r/>\t<group code>\t<source>\t<confidence>\t<reason>`
- `source` = `bio` if TITLE/BIO/SIDEBAR decided it, `name` if only the name did.
- `confidence` = `high`, `medium` or `low`.
- `reason` = at most 12 words, no tabs.
