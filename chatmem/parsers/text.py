"""Text-repair helpers shared across parsers.

Instagram's JSON export writes non-ASCII text as UTF-8 bytes that have been
misinterpreted as latin-1 and re-encoded as JSON \\uXXXX escapes (Python's
json module decodes those escapes into a str of latin-1 codepoints). The fix
is the inverse of that mistake: encode the (wrongly-decoded) str back to
bytes as latin-1, then decode those bytes as UTF-8.
"""

from __future__ import annotations

import re
import unicodedata


def fix_mojibake(s: str) -> str:
    """Undo a UTF-8-bytes-written-as-latin-1 mojibake round-trip.

    Text that was never mangled this way (plain ASCII, or already-correct
    non-Latin text) will not round-trip through latin-1 encoding and is
    returned unchanged.
    """
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_alias(name: str) -> str:
    """Normalize a display name for identity matching.

    mojibake-fix -> NFKC -> casefold -> whitespace-collapsed. Used to decide
    whether two raw sender names refer to the same declared identity.
    """
    fixed = fix_mojibake(name)
    nfkc = unicodedata.normalize("NFKC", fixed)
    folded = nfkc.casefold().strip()
    return _WHITESPACE_RE.sub(" ", folded)
