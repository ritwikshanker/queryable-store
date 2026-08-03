"""Split an ordered stream of messages into sessions.

Pure function over an already-ordered sequence of messages -- no DB
involved, so it is trivially testable. Two rules, applied in order:

  1. Split on an idle gap: wherever the time since the previous message
     exceeds `idle_gap_minutes`.
  2. Within any run still longer than `max_messages`, recursively split at
     the largest idle gap found in the *middle 50%* of that run, then
     recurse on both halves.

Rule 2 is a soft cap, not a hard one: if a run has no positive interior gap
in its search window (e.g. every message landed at the same timestamp),
it is left intact rather than cut mid-exchange on message count alone.
"""

from __future__ import annotations

from typing import Protocol, Sequence


class _HasTimestampAndSeq(Protocol):
    timestamp_ms: int
    seq: int


def sessionize(
    messages: Sequence[_HasTimestampAndSeq],
    idle_gap_minutes: float = 120.0,
    max_messages: int = 40,
) -> list[tuple[int, int]]:
    """Return a list of inclusive (start_seq, end_seq) ranges.

    `messages` must already be ordered ascending (by seq / timestamp_ms).
    The returned ranges are contiguous, non-overlapping, and together cover
    every input message.
    """
    if not messages:
        return []

    idle_gap_ms = idle_gap_minutes * 60_000

    runs: list[list[_HasTimestampAndSeq]] = []
    current: list[_HasTimestampAndSeq] = [messages[0]]
    for prev, msg in zip(messages, messages[1:]):
        if msg.timestamp_ms - prev.timestamp_ms > idle_gap_ms:
            runs.append(current)
            current = [msg]
        else:
            current.append(msg)
    runs.append(current)

    final_runs: list[list[_HasTimestampAndSeq]] = []
    for run in runs:
        final_runs.extend(_split_oversized(run, max_messages))

    return [(run[0].seq, run[-1].seq) for run in final_runs]


def _split_oversized(
    run: list[_HasTimestampAndSeq], max_messages: int
) -> list[list[_HasTimestampAndSeq]]:
    n = len(run)
    if n <= max_messages:
        return [run]

    # Interior gap positions are i = 1..n-1 (the gap between run[i-1] and
    # run[i]). Restrict the search to the middle 50% of those positions.
    lo = max(1, n // 4)
    hi = min(n - 1, (3 * n) // 4)
    if lo > hi:
        lo, hi = 1, n - 1

    best_idx: int | None = None
    best_gap = 0
    for i in range(lo, hi + 1):
        gap = run[i].timestamp_ms - run[i - 1].timestamp_ms
        if gap > best_gap:
            best_gap = gap
            best_idx = i

    if best_idx is None:
        # No positive interior gap in the window: never split mid-exchange
        # on count alone. The cap is a soft target.
        return [run]

    left, right = run[:best_idx], run[best_idx:]
    return _split_oversized(left, max_messages) + _split_oversized(right, max_messages)
