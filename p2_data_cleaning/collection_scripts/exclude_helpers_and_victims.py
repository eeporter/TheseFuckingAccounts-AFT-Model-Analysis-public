# WHAT: helper module, imported by build_accused_table.py and extraction_review_sample_100.py. non_accused(text) returns the names
#       a post frames as helpers or victims, so they are not counted as accused in that post.
"""Role exclusion: names framed as HELPER (credited with finding the account) or VICTIM
(the original account whose content was copied) rather than as the account being accused.

Deliberately conservative. Three traps this encodes, each found in the corpus:
  * "OP is u/X" names the AUTHOR OF THE SUSPICIOUS POST here, not a victim - no rule for it.
  * "u/example_reporter12\n\nReported yesterday" - the verb belongs to the next sentence, so helper
    verbs must sit on the SAME LINE as the mention.
  * "u/example_accused_a spotted in r/x" is passive (accused); "u/example_helper spotted them" is active
    (helper) - the ambiguous verbs therefore require an explicit object.
"""
import re
# N captures a username; U matches the u/ or /user/ prefix; OBJ is the object an active helper verb needs.
N=r'([A-Za-z0-9_-]{3,20})(?![A-Za-z0-9_-])'
U=r'(?<![A-Za-z0-9_./])/?u(?:ser)?/'
OBJ=r'(?:the|this|these|those|them|him|her|it|out|another|more|some|several|two|three|four|five|\d+)\b'

# Helper patterns: the named account found or reported the accused account.
HELPER=[
 re.compile(rf'(?:detected|reported|found|spotted|caught|flagged)\s+by\s+{U}{N}',re.I),
 re.compile(rf'\bthanks?\s+(?:to\s+)?{U}{N}',re.I),
 re.compile(rf'\bcourtesy\s+of\s+{U}{N}',re.I),
 re.compile(rf'(?:h/t|hat\s*tip)\s+{U}{N}',re.I),
 re.compile(rf'(?:shout\s*out|props|kudos|cheers)\s+(?:to\s+)?{U}{N}',re.I),
 re.compile(rf'\bas\s+{U}{N}[ \t]+(?:reported|noted|said|pointed)',re.I),
 re.compile(rf'{U}{N}[ \t]+(?:pointed\s+out|noticed|helped)\b',re.I),
 re.compile(rf'{U}{N}[ \t]+(?:found|spotted|caught|reported)[ \t]+{OBJ}',re.I),
 re.compile(rf'{U}{N}[ \t]+for[ \t]+(?:noticing|catching|helping|finding|reporting|the\s+heads)',re.I),
]
# Victim pattern: 'copy/paste of u/X comment' with no possessive.
VICTIM=[
 re.compile(rf'\bof\s+{U}{N}\s+(?:comment|submission|title|post)\b',re.I),
]

# Lower-cased names matched by any helper or victim pattern in this text.
def non_accused(text):
    out=set()
    for pat in HELPER+VICTIM:
        for m in pat.finditer(text):
            if m.group(1): out.add(m.group(1).lower())
    return out
