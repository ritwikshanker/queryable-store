from dataclasses import dataclass

from chatmem.sessionize import sessionize


@dataclass
class M:
    timestamp_ms: int
    seq: int


def _msgs(timestamps: list[int]) -> list[M]:
    return [M(timestamp_ms=t, seq=i) for i, t in enumerate(timestamps)]


def test_empty_input_yields_no_sessions():
    assert sessionize([]) == []


def test_three_gaps_yield_four_sessions():
    hour = 60 * 60 * 1000
    # 4 groups of 3 messages, each group separated by a 3h gap (> default 2h).
    timestamps = []
    t = 0
    for group in range(4):
        for i in range(3):
            timestamps.append(t)
            t += 1000
        t += 3 * hour
    msgs = _msgs(timestamps)

    ranges = sessionize(msgs, idle_gap_minutes=120, max_messages=40)

    assert ranges == [(0, 2), (3, 5), (6, 8), (9, 11)]


def test_dominant_interior_gap_splits_oversized_run_exactly_once():
    # 100 messages, all other deltas zero, except one huge gap right in the
    # middle. Every gap besides the dominant one is zero on purpose: once
    # the run is split there, each half has no positive interior gap left
    # to recurse on (soft cap), so the split happens exactly once.
    # idle_gap_minutes is set high enough that only the max_messages path
    # can trigger a split.
    n = 100
    dominant_gap_ms = 10_000_000
    timestamps = []
    t = 0
    for i in range(n):
        timestamps.append(t)
        if i == 49:
            t += dominant_gap_ms
    msgs = _msgs(timestamps)

    ranges = sessionize(msgs, idle_gap_minutes=10_000, max_messages=40)

    assert ranges == [(0, 49), (50, 99)]


def test_gapless_oversized_run_stays_whole_soft_cap():
    n = 100
    msgs = _msgs([0] * n)  # every message at the same instant: no interior gap at all

    ranges = sessionize(msgs, idle_gap_minutes=10_000, max_messages=40)

    assert ranges == [(0, n - 1)]


def test_ranges_are_contiguous_non_overlapping_and_cover_everything():
    hour = 60 * 60 * 1000
    timestamps = [0, 1000, 2000, 2000 + 3 * hour, 2000 + 3 * hour + 500]
    msgs = _msgs(timestamps)

    ranges = sessionize(msgs, idle_gap_minutes=120, max_messages=40)

    covered = []
    for start, end in ranges:
        assert start <= end
        covered.extend(range(start, end + 1))
    assert covered == list(range(len(timestamps)))

    for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:]):
        assert next_start == prev_end + 1
