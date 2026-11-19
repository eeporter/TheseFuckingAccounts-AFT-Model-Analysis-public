# Link-domain classification guide (AFT variable #16, link types)

Each input line is one web domain that the pilot accounts linked to before they were accused:
`<domain>  LINKS: n  ACCOUNTS: n  HOSTS: ...  EXAMPLES: url | url | url  POST_TITLES: ...`
(HOSTS = most-used subdomains; EXAMPLES = up to 3 linked URLs; POST_TITLES = titles of posts that linked it, if any).

Assign exactly ONE code per domain, judged by **what the site mainly is** — use your knowledge of the site first, then
the subdomains, URL paths and titles. For hosting platforms whose content varies (blogspot, wordpress, medium, tumblr,
sites.google, github.io, linktree-style link pages), judge by what these links are used for if the examples make that
clear; otherwise use the platform's typical use.

## Codes

- `porn` — pornography or sexual content: porn tube/image sites, cam sites, escort sites, OnlyFans/Fansly/premium-content
  pages, adult dating/hookup sites, sexual link pages.
- `crypto` — cryptocurrency: exchanges, wallets, coin/token/NFT project sites, airdrop/faucet sites, crypto news,
  blockchain explorers.
- `shop` — selling or earning money: online shops and product pages, marketplaces, affiliate/referral/sign-up-bonus
  links, coupon and deal sites, paid services (essay/homework help, followers for sale, VPN deals), apps or games
  promoted for download, payment/cash-app links, gambling/betting sites.
- `blog_channel` — someone's own content channel or blog: personal/company blogs, content farms, article-spinning and
  clickbait sites, streaming/video channels (Twitch, Vimeo channel pages, Dailymotion, Rumble, Bitchute), podcasts,
  Patreon/Ko-fi, personal websites and link-in-bio pages.
- `news` — news organisations and news aggregators (newspapers, TV news sites, wire services, online news outlets,
  including partisan ones).
- `media_host` — image/GIF/video hosting or screenshot services not already in the visual-media list (e.g. gyazo,
  prnt.sc, postimg, ibb.co, streamja, vimeo player links, catbox, tenor).
- `messaging` — chat or group invites (Discord, Telegram, WhatsApp, Kik, Snapchat add links, Signal).
- `social` — social networks and general platforms: Twitter/X, Facebook, Instagram, TikTok, LinkedIn, Pinterest, Quora,
  Wikipedia, Google search/maps/drive/docs, app stores' generic pages, government/education reference sites.
- `other` — anything else (software/tools, forums, URL shorteners you cannot resolve, broken or nonsense domains).

If unsure between two, prefer in this order: porn > crypto > shop > messaging > news > blog_channel > media_host >
social > other.

## Output
One tab-separated line per input line, in the same order, exactly 4 fields, no header:
`<domain exactly as given>\t<code>\t<high|medium|low>\t<reason, at most 10 words, no tabs>`
