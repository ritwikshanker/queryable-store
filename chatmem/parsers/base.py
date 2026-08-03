"""Parser protocol: the seam that lets other export formats plug in later.

A Parser turns an on-disk export into ParsedThread objects with raw
(un-resolved) sender names -- identity resolution and sessionizing happen
afterward, in the ingest pipeline, so a parser only needs to know its own
export format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol

from chatmem.models import ParsedThread


class Parser(Protocol):
    name: str

    def detect(self, path: Path) -> bool:
        """Return True if `path` looks like an export this parser can read."""
        ...

    def parse(self, path: Path) -> Iterator[ParsedThread]:
        """Yield one ParsedThread per conversation found under `path`."""
        ...


PARSERS: list[Parser] = []


def register(parser: Parser) -> Parser:
    PARSERS.append(parser)
    return parser


def select_parser(path: Path, name: str | None = None) -> Parser:
    """Pick a parser for `path`, either by explicit name or by auto-detection."""
    if name is not None:
        for parser in PARSERS:
            if parser.name == name:
                return parser
        available = ", ".join(p.name for p in PARSERS) or "(none registered)"
        raise ValueError(f"no parser named {name!r} (available: {available})")

    candidates = [p for p in PARSERS if p.detect(path)]
    if not candidates:
        raise ValueError(
            f"could not auto-detect an export format at {path}; "
            f"pass --source explicitly (available: "
            f"{', '.join(p.name for p in PARSERS) or '(none registered)'})"
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(
            f"multiple parsers claim to handle {path} ({names}); pass --source to disambiguate"
        )
    return candidates[0]
