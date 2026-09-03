"""Text cleaning for the Disaster Tweets dataset.

The single source of truth for preprocessing. Import this from notebooks so that
train and test are guaranteed to receive identical treatment -- applying cleaning
to only one split is the classic way to produce a good validation score and a bad
leaderboard score.

Two presets, because the two candidate model families want opposite things:

    mode="transformer"  light. Repairs the text and drops non-linguistic noise
                        (URLs, @mentions), but keeps casing, punctuation and
                        digits, which WordPiece/BPE tokenizers use as signal.

    mode="lstm"         aggressive. Everything above, plus lowercasing and the
                        removal of punctuation and digits, for a from-scratch
                        Embedding layer with a small vocabulary.

Stopwords are deliberately NOT removed in either mode: both target architectures
are sequence models that depend on word order.
"""

from __future__ import annotations

import html
import re
from urllib.parse import unquote

__all__ = ["clean_text", "fix_encoding", "clean_keyword", "MOJIBAKE_MAP"]


# ---------------------------------------------------------------------------
# Encoding repair
# ---------------------------------------------------------------------------
# This dataset ships mojibake that no single-byte codec round-trip repairs, so
# the map below is derived empirically from the corpus. Each entry was confirmed
# against surrounding context, e.g.:
#     'RÌ©union'        -> 'Réunion'
#     'å£279.00'        -> '£279.00'
#     '91å¡F'           -> '91°F'
#     'ÛÏAirplaneÛ\x9d' -> '"Airplane"'
# Order matters: longer sequences must be replaced before their prefixes, since
# '\x89Û' is a prefix of nearly every other entry.
MOJIBAKE_MAP: list[tuple[str, str]] = [
    ("\x89Û_", "..."),      # horizontal ellipsis (tweet truncation marker)
    ("\x89Û\x9d", '"'),     # right double quotation mark
    ("\x89ÛÏ", '"'),        # left double quotation mark
    ("\x89Ûª", "'"),        # right single quotation mark / apostrophe
    ("\x89Û÷", "'"),        # left single quotation mark
    ("\x89ÛÒ", " - "),      # en dash
    ("\x89ÛÓ", " - "),      # em dash
    ("\x89Û¢", " "),        # bullet
    ("\x89ã¢", " "),        # bullet (variant)
    ("\x89âÂ", " "),        # currency/symbol remnant
    ("\x89Û", " "),         # bare prefix; must come after every '\x89Û…' entry
    ("åÊ", " "),            # non-breaking space
    ("åÈ", " "),            # decorative separator
    ("å¨", " "),
    ("åÇ", " "),
    ("å£", "£"),
    ("å¡", "°"),
    ("å«", "'"),
    ("Ì©", "é"),
    ("Ì¼", "ú"),
    ("Ìà", "Ç"),
    ("Ìü", " "),
    ("ÌÑ", " "),
    ("Ì", " "),             # bare leftover; keep last of the 'Ì…' entries
    ("å", " "),             # bare leftover; keep last of the 'å…' entries
]

# '\bhttps?\b' catches the bare word 'http' left by tweets that contain
# 'http http://t.co/x' -- 'http\S+' cannot match a trailing 'http' with no path.
_URL_RE = re.compile(r"https?://\S*|www\.\S+|\bhttps?\b", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HASH_SYMBOL_RE = re.compile(r"#")
_C1_RE = re.compile(r"[\x80-\x9f]")
_APOSTROPHE_RE = re.compile(r"['\u2018\u2019]")
_PUNCT_RE = re.compile(r"[^\w\s]")
_UNDERSCORE_RE = re.compile(r"_+")
_DIGIT_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")


def fix_encoding(text: str) -> str:
    """Repair mojibake and HTML entities.

    `html.unescape` runs first so that '&amp;' becomes '&' rather than surviving
    cleaning as the bare token 'amp' (300 occurrences in train). It is applied
    repeatedly because a few rows are double-encoded.
    """
    # Two rows are double-encoded ('&amp;amp;'), so unescape until it stabilises
    # rather than exactly once.
    for _ in range(3):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    for bad, good in MOJIBAKE_MAP:
        text = text.replace(bad, good)
    # Defensive: drop any C1 control byte the map above did not account for.
    return _C1_RE.sub(" ", text)


def clean_text(text: str, mode: str = "lstm") -> str:
    """Clean one tweet. See module docstring for the two modes."""
    if mode not in ("lstm", "transformer"):
        raise ValueError(f"mode must be 'lstm' or 'transformer', got {mode!r}")
    if not isinstance(text, str):
        return ""

    text = fix_encoding(text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    # Drop the '#' but keep the word: '#earthquake' -> 'earthquake'.
    text = _HASH_SYMBOL_RE.sub("", text)

    if mode == "lstm":
        text = text.lower()
        # Delete apostrophes rather than replacing them with a space, so that
        # "can't" becomes "cant" instead of the tokens "can" + "t". Splitting
        # instead would emit 3,233 stray one-letter tokens across the corpus.
        text = _APOSTROPHE_RE.sub("", text)
        text = _PUNCT_RE.sub(" ", text)
        # '_' is a \w character, so _PUNCT_RE leaves it behind.
        text = _UNDERSCORE_RE.sub(" ", text)
        text = _DIGIT_RE.sub(" ", text)

    # Always collapse whitespace last: URL/mention removal leaves ragged gaps,
    # and the raw text contains 435 embedded newlines.
    return _WS_RE.sub(" ", text).strip()


def clean_keyword(keyword: str) -> str:
    """URL-decode a keyword ('body%20bags' -> 'body bags') and normalise case."""
    if not isinstance(keyword, str):
        return "missing"
    return unquote(keyword).strip().lower()
