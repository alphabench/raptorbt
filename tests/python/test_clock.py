"""Clock timer scheduling."""

import time

from raptorbt.strategy.clock import Clock

MIN_NS = 60_000_000_000
EPOCH_TS = 1_767_592_200_000_000_000  # 2026-01-05 05:50 UTC


def _reference_next(first, interval, ts):
    while first <= ts:
        first += interval
    return first


def test_missed_points_collapse_to_one_firing_and_the_next_point_after_now():
    for first, interval, ts in [(0, 7, 0), (5, 3, 100), (-40, 9, 41), (10, 10, 10), (1, 1, 999)]:
        c = Clock()
        c.set_timer("t", interval, start_ns=first)
        events = c._advance(ts)
        assert [e.ts_scheduled for e in events] == [first]
        assert c._timers["t"][0] == _reference_next(first, interval, ts)


def test_a_timer_set_before_the_first_bar_does_not_step_through_epoch_time():
    # Default start is one interval after "now", which is 0 before any bar:
    # the first advance to an epoch timestamp skips ~3e7 missed points.
    c = Clock()
    c.set_timer("t", MIN_NS)
    t0 = time.perf_counter()
    events = c._advance(EPOCH_TS)
    assert time.perf_counter() - t0 < 0.05
    assert len(events) == 1
    assert c._timers["t"][0] == _reference_next(MIN_NS, MIN_NS, EPOCH_TS)
