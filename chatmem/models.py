"""Normalized data model shared by parsers, the store, and the sessionizer.

Parsers (chatmem.parsers.*) produce RawMessage/ParsedThread objects with raw,
un-resolved sender names. The ingest pipeline resolves those names to
person_id via chatmem.identity before anything is written to storage or
handed to sessionize().
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RawMessage:
    """A single message as a parser emits it, before identity resolution."""

    thread_id: str
    sender: str  # raw display name, exactly as it appears in the export
    timestamp_ms: int
    text: str | None
    media_type: str | None
    ordinal: int  # position within the raw parse order; disambiguates ties


@dataclass(frozen=True)
class ParsedThread:
    """One conversation thread as produced by a Parser.parse() call."""

    thread_id: str
    title: str | None
    participants: list[str]
    source: str
    messages: list[RawMessage] = field(default_factory=list)
    # Rows the parser chose not to emit, by reason (e.g. "unsent", "call",
    # "reaction", "empty"). Format-specific; the ingest pipeline just sums
    # and reports these, it does not interpret the reason strings.
    dropped: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    """A normalized message row, after identity resolution, as stored."""

    id: str
    thread_id: str
    sender: str  # raw display name, preserved verbatim
    person_id: str
    timestamp_utc: str  # ISO-8601 with 'Z'
    timestamp_ms: int
    text: str | None
    media_type: str | None
    seq: int  # 0-based position within the thread, ascending


@dataclass(frozen=True)
class Person:
    person_id: str
    display_name: str
    origin: str  # 'config' | 'auto'


@dataclass(frozen=True)
class Session:
    id: int | None
    thread_id: str
    start_seq: int
    end_seq: int
    start_ts: str
    end_ts: str
    message_count: int
    # When `chatmem extract` last processed this session (ISO-8601), or None
    # if it never has. Lets extract resume without redoing sessions, even
    # ones that legitimately produced zero statements.
    extracted_at: str | None = None


@dataclass(frozen=True)
class Statement:
    """A self-statement the target made, extracted from one session.

    start_ts/end_ts are the min/max timestamp among source_message_ids (or
    the session's own bounds if the LLM cited nothing) -- the citation shown
    back to the user.
    """

    id: int | None
    person_id: str
    session_id: int
    thread_id: str
    text: str
    source_message_ids: list[str]
    start_ts: str
    end_ts: str
    created_at: str
    embedding: list[float] | None = None
    # A chatmem.topics key, assigned by `chatmem digest`, or None until it has
    # run. Extraction never sets it, so a statement added by a later extract
    # is picked up by the next digest without re-classifying the rest.
    topic: str | None = None
