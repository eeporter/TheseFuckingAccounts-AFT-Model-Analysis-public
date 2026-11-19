# Entertainment vs other — guide (subreddits currently in "other")

Each input line: `r/<name>  PREVIOUS_REASON: ...  TITLE: ...  BIO: ...  SIDEBAR: ...  ARCHIVED_TITLES: ...`
(PREVIOUS_REASON is a short note from an earlier classification; TITLE/BIO/SIDEBAR are what the subreddit says about
itself and may be empty for banned/private subreddits, where ARCHIVED_TITLES shows a few post titles instead).

Decide ONE code per subreddit, judging what the subreddit is about (bio first, then name/titles):

- `entertainment` — the subreddit is centred on:
  - **gaming**: video games, mobile games, PC/console gaming, a specific game, game studios, esports, speedrunning,
    game streamers, game mods, tabletop/board/card games (D&D, Magic, chess);
  - **sports**: any sport, teams, clubs, leagues, athletes, fantasy sports, combat sports, motorsport, sports betting talk;
  - **TV / film / anime / music**: TV shows, films, streaming series, anime/manga, cartoons, comics/superheroes,
    celebrities/actors, musicians/bands/rappers, music genres, music sharing, and fan communities or topic-specific memes
    about any of these (e.g. memes about one game or show).
- `other` — everything else (books/literature, podcasts, YouTubers not about games/music, hobbies, crafts, pets, food,
  tech/programming, science, education, health/support, identity, religion, relationships, finance, test/gibberish, etc.).

If unsure, use `other` with low confidence.

Output: one tab-separated line per input line, same order, 4 fields, no header:
`<subreddit name without r/>\t<entertainment|other>\t<high|medium|low>\t<reason, at most 10 words, no tabs>`
