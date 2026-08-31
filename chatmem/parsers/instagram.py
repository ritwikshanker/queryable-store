"""Parser for Instagram's JSON export (messages/inbox/<thread>/message_N.json).

Landmines handled here, deliberately:

  - Text is double-encoded (UTF-8 bytes written out as latin-1). Fixed via
    fix_mojibake() on every string field.
  - A thread is split across message_1.json, message_2.json, ... and
    messages are newest-first *within* each file. All files are merged and
    sorted ascending by timestamp_ms, with a stable tiebreaker so repeated
    timestamps (the export can and does emit these) still come out in true
    chronological order.
  - Reactions, "Liked a message", call logs, and unsend tombstones are
    dropped (counted by reason, not silently discarded).
  - Media messages carry no text; a placeholder row is kept with
    `media_type` set.
  - The export's field names are not to be trusted: a real sample contains
    a message with no `content` key at all, and a typo'd field
    (`is_geobloced_for_viewer` instead of `is_geoblocked_for_viewer`) on a
    neighboring message. Every field access here goes through `.get()` with
    a default -- never a bare index.

Message ids have no source-provided equivalent, so one is derived
deterministically downstream (in the ingest pipeline) from
(thread_id, timestamp_ms, sender, text, media_type, ordinal) -- `ordinal`
here is each message's 0-based position in the final, fully sorted order,
which is what makes that hash reproducible across re-ingests and distinct
for genuine same-timestamp duplicates.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from chatmem.models import ParsedThread, RawMessage
from chatmem.parsers.base import register
from chatmem.parsers.text import fix_mojibake

_MESSAGE_FILE_RE = re.compile(r"^message_(\d+)\.json$")

# Heuristic, English-locale-only. Instagram does not mark these with a
# dedicated "type" field in the export; they are ordinary messages whose
# `content` string happens to be one of these generated phrases.
_CALL_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^missed (a )?video (chat|call)$",
        r"^missed (a )?audio call$",
        r"^video (chat|call) ended$",
        r"^audio call ended$",
        r"^started (a )?video (chat|call)$",
        r"^started (a )?audio call$",
    )
]
_REACTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^liked a message$",
        r"^reacted .+ to (your|their|the) message$",
    )
]

# key in the raw message dict -> media_type, checked in this priority order.
_MEDIA_LIST_KEYS = (
    ("photos", "photo"),
    ("videos", "video"),
    ("audio_files", "audio"),
    ("gifs", "gif"),
    ("files", "file"),
)


def _message_files(thread_dir: Path) -> list[Path]:
    numbered = []
    for p in thread_dir.iterdir():
        m = _MESSAGE_FILE_RE.match(p.name)
        if m:
            numbered.append((int(m.group(1)), p))
    numbered.sort(key=lambda t: t[0])
    return [p for _, p in numbered]


def _is_thread_dir(path: Path) -> bool:
    return path.is_dir() and len(_message_files(path)) > 0


def _load_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"{path}: not valid UTF-8: {e}") from e
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{path}: invalid JSON at line {e.lineno}, column {e.colno} "
            f"(character {e.pos}): {e.msg}"
        ) from e
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object at the top level")
    return data


def _timestamp_ms(raw: dict[str, Any]) -> int | None:
    """The message's timestamp, or None if it is missing or unusable.

    Coercing a missing timestamp to 0 would date the message to 1970, sort it
    to the front of the thread, and distort the idle gaps sessionize() splits
    on -- so callers drop these instead, counted like any other drop reason.
    """
    ts = raw.get("timestamp_ms")
    if isinstance(ts, bool) or not isinstance(ts, int) or ts <= 0:
        return None
    return ts


def _detect_media_type(raw: dict[str, Any]) -> str | None:
    for key, media_type in _MEDIA_LIST_KEYS:
        val = raw.get(key)
        if isinstance(val, list) and len(val) > 0:
            return media_type
    if raw.get("sticker"):
        return "sticker"
    if raw.get("share"):
        return "share"
    return None


def _classify(raw: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    """Return (drop_reason, text, media_type); drop_reason is None if kept."""
    raw_content = raw.get("content")
    content = fix_mojibake(raw_content) if isinstance(raw_content, str) else None
    # Generated phrases arrive with trailing whitespace in real exports
    # ("Reacted <emoji> to your message "), so the anchored patterns below are
    # matched against a stripped copy. The message keeps its original text.
    probe = content.strip() if content else None

    if raw.get("is_unsent"):
        return "unsent", None, None

    if raw.get("call_duration") is not None:
        return "call", None, None
    if probe and any(p.match(probe) for p in _CALL_PATTERNS):
        return "call", None, None

    if probe and any(p.match(probe) for p in _REACTION_PATTERNS):
        return "reaction", None, None

    media_type = _detect_media_type(raw)

    text = content
    if text is None and media_type == "share":
        share = raw.get("share")
        share_text = share.get("share_text") if isinstance(share, dict) else None
        if isinstance(share_text, str):
            text = fix_mojibake(share_text)

    if text is None and media_type is None:
        return "empty", None, None

    return None, text, media_type


class InstagramParser:
    name = "instagram"

    def detect(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        if _is_thread_dir(path):
            return True
        return any(_is_thread_dir(child) for child in path.iterdir() if child.is_dir())

    def parse(self, path: Path) -> Iterator[ParsedThread]:
        if _is_thread_dir(path):
            yield self._parse_thread_dir(path)
            return
        for child in sorted(path.iterdir()):
            if _is_thread_dir(child):
                yield self._parse_thread_dir(child)

    def _parse_thread_dir(self, thread_dir: Path) -> ParsedThread:
        files = _message_files(thread_dir)
        thread_id = thread_dir.name

        title: str | None = None
        participants: list[str] = []
        seen_participants: set[str] = set()
        # (timestamp_ms, file_index, index_within_file, raw_message)
        collected: list[tuple[int, int, int, dict[str, Any]]] = []
        dropped: Counter[str] = Counter()

        for file_index, path in enumerate(files):
            data = _load_json(path)

            for p in data.get("participants") or []:
                name = p.get("name") if isinstance(p, dict) else None
                if isinstance(name, str):
                    fixed = fix_mojibake(name)
                    if fixed not in seen_participants:
                        seen_participants.add(fixed)
                        participants.append(fixed)

            if title is None:
                raw_title = data.get("title")
                if isinstance(raw_title, str):
                    title = fix_mojibake(raw_title)

            for index_within_file, raw in enumerate(data.get("messages") or []):
                if not isinstance(raw, dict):
                    continue
                ts = _timestamp_ms(raw)
                if ts is None:
                    dropped["missing_timestamp"] += 1
                    continue
                collected.append((ts, file_index, index_within_file, raw))

        # Ascending by timestamp; ties broken so that within-file order
        # (newest-first) is reversed back into chronological order, and
        # earlier files win ties against later ones.
        collected.sort(key=lambda t: (t[0], t[1], -t[2]))

        messages: list[RawMessage] = []

        for ordinal, (ts, _file_index, _idx, raw) in enumerate(collected):
            raw_sender = raw.get("sender_name")
            sender = fix_mojibake(raw_sender) if isinstance(raw_sender, str) else "Unknown"

            reason, text, media_type = _classify(raw)
            if reason is not None:
                dropped[reason] += 1
                continue

            messages.append(
                RawMessage(
                    thread_id=thread_id,
                    sender=sender,
                    timestamp_ms=ts,
                    text=text,
                    media_type=media_type,
                    ordinal=ordinal,
                )
            )

        return ParsedThread(
            thread_id=thread_id,
            title=title,
            participants=participants,
            source=self.name,
            messages=messages,
            dropped=dict(dropped),
        )


register(InstagramParser())
