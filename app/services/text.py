"""Tokenisation shared by every retriever so comparisons stay apples-to-apples."""
import re
from functools import lru_cache

import snowballstemmer

_WORD = re.compile(r"[a-z0-9]+")
_STEMMER = snowballstemmer.stemmer("english")

# Question scaffolding and function words. Deliberately short and generic: no
# domain terms, nothing derived from the evaluation queries.
STOP_WORDS = frozenset(
    """a an and are as at be but by can could do does for from how i if in is it
    its me my of on or should so than that the their them then there these they
    this to was we were what when where which who why will with would you your
    much many""".split()
)


@lru_cache(maxsize=None)
def _stem(word: str) -> str:
    return _STEMMER.stemWord(word)


def tokenize(text: str, *, stem: bool = True, bigrams: bool = False) -> list[str]:
    """Lower-case alphanumeric tokens, stop words removed, optionally stemmed.

    With ``bigrams=True`` adjacent token pairs (after stop-word removal) are appended
    as ``a_b`` tokens, so a phrase such as "vitamin d" can match as a unit.
    """
    words = [w for w in _WORD.findall(text.lower()) if w not in STOP_WORDS]
    toks = [_stem(w) for w in words] if stem else words
    if bigrams:
        toks = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    return toks
