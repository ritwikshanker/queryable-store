"""WhatsApp "Export chat" parser (the .txt file, with or without media).

WhatsApp exports one plain-text file per conversation, one message per line,
except that a message containing newlines continues on unprefixed lines. Two
header layouts are in the wild:

    [12/05/2023, 9:41:02 AM] Dana: hey            <- iOS
    12/05/2023, 9:41 AM - Dana: hey               <- Android

Neither carries a time zone or a year-month-day order, so both are inferred
per file (see _pick_date_order) and timestamps are read as UTC. That keeps a
thread internally consistent, which is all sessionize() needs -- it only ever
looks at gaps between messages in the same thread.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from chatmem.models import ParsedThread, RawMessage
from chatmem.parsers.base import register
from chatmem.parsers.text import fix_mojibake

# WhatsApp sprinkles directionality marks and narrow no-break spaces through
# exported lines; they carry no meaning here and only break the regexes.
_INVISIBLE = str.maketrans({"‎": "", "‏": "", " ": " ", " ": " "})

_IOS_RE = re.compile(
    r"^\[(?P<date>\d{1,2}[./-]\d{1,2}[./-]\d{2,4}), "
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]\s*(?P<rest>.*)$"
)
_ANDROID_RE = re.compile(
    r"^(?P<date>\d{1,2}[./-]\d{1,2}[./-]\d{2,4}), "
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?) - (?P<rest>.*)$"
)

# "Dana: hey" -> ("Dana", "hey"). A line with no sender is a system notice
# ("Messages are end-to-end encrypted", "Dana created this group").
_SENDER_RE = re.compile(r"^(?P<sender>[^:]{1,80}): (?P<text>.*)$", re.DOTALL)

_MEDIA_PLACEHOLDERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^<Media omitted>$", re.IGNORECASE), "media"),
    (re.compile(r"^(image|photo) omitted$", re.IGNORECASE), "photo"),
    (re.compile(r"^video omitted$", re.IGNORECASE), "video"),
    (re.compile(r"^(audio|voice message) omitted$", re.IGNORECASE), "audio"),
    (re.compile(r"^sticker omitted$", re.IGNORECASE), "sticker"),
    (re.compile(r"^GIF omitted$", re.IGNORECASE), "gif"),
    (re.compile(r"^document omitted$", re.IGNORECASE), "file"),
    (re.compile(r"^Contact card omitted$", re.IGNORECASE), "share"),
]

_DELETED_RE = re.compile(
    r"^(This message was deleted|You deleted this message)\.?$", re.IGNORECASE
)
_SYSTEM_TEXT_RE = re.compile(
    r"^(Messages and calls are end-to-end encrypted|"
    r"Your security code with .* changed|"
    r"Missed (voice|video) call)",
    re.IGNORECASE,
)

_TIME_FORMATS = ("%I:%M:%S %p", "%I:%M %p", "%H:%M:%S", "%H:%M")


def _clean(line: str) -> str:
    return line.translate(_INVISIBLE).replace("\r", "")


def _parse_time(value: str) -> tuple[int, int, int] | None:
    normalized = re.sub(r"\s+", " ", value.strip()).upper()
    for fmt in _TIME_FORMATS:
        try:
            t = datetime.strptime(normalized, fmt)
        except ValueError:
            continue
        return t.hour, t.minute, t.second
    return None


def _date_parts(value: str) -> tuple[int, int, int] | None:
    parts = re.split(r"[./-]", value)
    if len(parts) != 3:
        return None
    try:
        a, b, year = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    if year < 100:
        year += 2000
    return a, b, year


def _pick_date_order(dates: list[tuple[int, int, int]]) -> bool:
    """Return True if the export writes day first (DD/MM), False for MM/DD.

    WhatsApp uses the exporting phone's locale and records no hint about it,
    so it has to be inferred: a component above 12 can only be a day, and
    failing that, the reading that keeps the file in chronological order --
    exports are always chronological -- is the right one. If neither test
    decides, day-first wins as the more common layout worldwide.
    """
    if any(a > 12 for a, _b, _y in dates):
        return True
    if any(b > 12 for _a, b, _y in dates):
        return False

    def ordered(day_first: bool) -> bool:
        keys = [
            (y, (a if not day_first else b), (b if not day_first else a)) for a, b, y in dates
        ]
        return all(x <= y for x, y in zip(keys, keys[1:]))

    if ordered(True) and not ordered(False):
        return True
    if ordered(False) and not ordered(True):
        return False
    return True


def _to_timestamp_ms(parts: tuple[int, int, int], time: tuple[int, int, int], day_first: bool) -> int | None:
    a, b, year = parts
    day, month = (a, b) if day_first else (b, a)
    try:
        dt = datetime(
            year, month, day, time[0], time[1], time[2], tzinfo=timezone.utc
        )
    except ValueError:
        return None
    return int(dt.timestamp() * 1000)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "whatsapp_thread"


def _chat_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".txt" else []
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".txt")


def _header(line: str) -> tuple[re.Match[str], bool] | None:
    """Match a line's timestamp header, returning (match, is_ios)."""
    m = _IOS_RE.match(line)
    if m is not None:
        return m, True
    m = _ANDROID_RE.match(line)
    if m is not None:
        return m, False
    return None


class WhatsAppParser:
    name = "whatsapp"

    def detect(self, path: Path) -> bool:
        for file in _chat_files(path):
            if self._looks_like_chat(file):
                return True
        return False

    def _looks_like_chat(self, file: Path) -> bool:
        try:
            with file.open("r", encoding="utf-8", errors="replace") as fh:
                for _ in range(20):
                    line = fh.readline()
                    if not line:
                        return False
                    line = _clean(line).strip()
                    if line and _header(line) is not None:
                        return True
        except OSError:
            return False
        return False

    def parse(self, path: Path) -> Iterator[ParsedThread]:
        for file in _chat_files(path):
            if self._looks_like_chat(file):
                yield self._parse_file(file)

    def _parse_file(self, file: Path) -> ParsedThread:
        thread_id = _slug(file.stem)
        dropped: Counter[str] = Counter()

        # (date_parts, time_parts, body) in file order; the date order can't
        # be resolved until every date in the file has been seen.
        raw_entries: list[tuple[tuple[int, int, int], tuple[int, int, int], str]] = []

        for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = _clean(line)
            matched = _header(line.strip()) if line.strip() else None
            if matched is None:
                # A continuation of the previous message's text.
                if raw_entries and line.strip():
                    date, time, body = raw_entries[-1]
                    raw_entries[-1] = (date, time, f"{body}\n{line}")
                continue
            match, _is_ios = matched
            date = _date_parts(match.group("date"))
            time = _parse_time(match.group("time"))
            if date is None or time is None:
                dropped["unparsable_timestamp"] += 1
                continue
            raw_entries.append((date, time, match.group("rest")))

        day_first = _pick_date_order([d for d, _t, _b in raw_entries])

        messages: list[RawMessage] = []
        participants: list[str] = []
        seen: set[str] = set()

        for ordinal, (date, time, body) in enumerate(raw_entries):
            timestamp_ms = _to_timestamp_ms(date, time, day_first)
            if timestamp_ms is None:
                dropped["unparsable_timestamp"] += 1
                continue

            sender_match = _SENDER_RE.match(body)
            if sender_match is None:
                dropped["system"] += 1
                continue

            sender = fix_mojibake(sender_match.group("sender").strip())
            text: str | None = fix_mojibake(sender_match.group("text").strip())

            if _DELETED_RE.match(text or ""):
                dropped["deleted"] += 1
                continue
            if _SYSTEM_TEXT_RE.match(text or ""):
                dropped["system"] += 1
                continue

            media_type: str | None = None
            for pattern, kind in _MEDIA_PLACEHOLDERS:
                if pattern.match(text or ""):
                    media_type = kind
                    text = None
                    break

            if not text and media_type is None:
                dropped["empty"] += 1
                continue

            if sender not in seen:
                seen.add(sender)
                participants.append(sender)

            messages.append(
                RawMessage(
                    thread_id=thread_id,
                    sender=sender,
                    timestamp_ms=timestamp_ms,
                    text=text,
                    media_type=media_type,
                    ordinal=ordinal,
                )
            )

        messages.sort(key=lambda m: (m.timestamp_ms, m.ordinal))

        return ParsedThread(
            thread_id=thread_id,
            title=file.stem,
            participants=participants,
            source=self.name,
            messages=messages,
            dropped=dict(dropped),
        )


register(WhatsAppParser())
