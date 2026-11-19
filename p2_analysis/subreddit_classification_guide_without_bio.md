# Subreddit classification guide — banned and private subreddits (no bio available)

These subreddits are now banned or private, so Reddit no longer shows what they say about themselves. Each input line
gives what a research sample of accounts did there:
`r/<name>  POSTS: n  COMMENTS: n  NSFW_SHARE_OF_POSTS: x  TITLES: ...  LINK_DOMAINS: ...  COMMENT_TEXT: ...`
(titles/comments are a few distinct examples, truncated; NSFW share is the share of the sample's posts there flagged NSFW).

Assign exactly ONE group per subreddit by judging **what the subreddit was for**, using its NAME together with the
titles, link domains and comment text. When the evidence is thin (e.g. a few short comments), rely on the name if it clearly
indicates a topic; otherwise use `other`.

## Groups (use these exact codes)

- `adult` — sexual/pornographic content or its promotion: nudes, porn, OnlyFans/Snapchat-premium/"link in comments"
  promotion, sexual chat, NSFW model/creator promotion — even if the name sounds like something else.
- `crypto` — cryptocurrency: coins/tokens, Bitcoin, altcoins, DeFi, NFTs, blockchain, crypto trading/mining/airdrops.
- `politics` — politics: parties, ideologies, politicians, elections, government/policy, political activism or memes.
- `karma_farming` — gaining Reddit karma/upvotes: "free karma", "upvote for upvote", karma for new accounts.
- `general` — broad general-interest: humour/memes in general, funny pictures/videos, cute animals, interesting content,
  general Q&A, opinions/discussion, AMAs.
- `news` — news and current events without a partisan/political focus.
- `sales` — buying, selling or promoting for money: items/accounts for sale ([WTS]/[WTB]), marketplaces, deals, coupons,
  referral/promo codes, sign-up bonuses, paid services (e.g. paid homework/essay help), affiliate links.
- `location` — a specific place (country, region, city, local community).
- `other` — anything else: a specific topic/hobby/game/sport/fandom/show, health or support, identity communities,
  test or placeholder subreddits, or too little evidence to tell.

## Tie-breaks
If it genuinely fits two groups: karma_farming > adult > crypto > sales > politics > news > location > general > other.

## Output
One tab-separated line per input line, in the same order, 5 fields, no header:
`<subreddit name without r/>\t<group code>\t<evidence>\t<confidence>\t<reason>`
- `evidence` = `content` if titles/links/comments decided it, `name` if only the name did, `none` if neither (→ `other`).
- `confidence` = `high`, `medium` or `low`.
- `reason` = at most 12 words, no tabs.
