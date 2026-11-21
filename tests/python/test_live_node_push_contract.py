"""What a push to a LiveNode shows, whatever the path inside it takes.

Every strategy's clock reads the time of the latest push of its symbol, between pushes and in
every hook, whether or not the strategy handles that kind of event; timers and alerts fire in
the same order with strategies that handle nothing beside them; every refusal of ``push_tick``,
``push_bar`` and ``push_group`` is the same exception, code and message, in the same order of
precedence, and changes nothing; a group delivers its bars by close time and journals what its
bars pushed one by one would.
"""

from __future__ import annotations

import decimal

import numpy as np
import pytest

from raptorbt import BacktestConfig, LiveNode, Strategy
from raptorbt.errors import InvalidData, InvalidParameter, StateError, UnknownSymbol, UnsupportedOperation
from raptorbt.journal import Journal
from raptorbt.strategy.clock import Clock

NS = 1_000_000_000
MIN = 60 * NS
NAN, INF = float("nan"), float("inf")
GOOD = (100.0, 101.0, 99.0, 100.5, 10.0)


# -- clocks -----------------------------------------------------------------------


class Bars(Strategy):
    def on_bar(self, ctx):
        pass


class Silent(Strategy):
    pass


class Ticker(Strategy):
    def __init__(self):
        super().__init__()
        self.seen = []

    def on_trade(self, ctx, tick):
        self.seen.append(("trade", ctx.symbol, ctx.event_ns, self.clock.timestamp_ns()))

    def on_quote(self, ctx, quote):
        self.seen.append(("quote", ctx.symbol, ctx.event_ns, self.clock.timestamp_ns()))


class Timed(Strategy):
    def __init__(self):
        super().__init__()
        self.seen = []

    def on_start(self, ctx):
        self.clock.set_timer("t", 10 * NS, start_ns=10 * NS)

    def on_timer(self, ctx, event):
        self.seen.append((ctx.symbol, event.ts_scheduled, event.ts_fired, self.clock.timestamp_ns()))


# (symbol, timestamp in seconds, ltp, bid, ask); ("time", seconds) is advance_time.
CLOCK_OPS = [
    ("X", 1, 10.0, 9.9, 10.1),  # a print and a quote at one instant
    ("Y", 1, 20.0, 0.0, 0.0),  # another symbol at the same instant, a print only
    ("X", 2, 0.0, 9.9, 10.1),  # a quote only
    ("Y", 3, 0.0, 0.0, 0.0),  # a refused row: time only
    ("X", 5, 0.0, 9.9, 0.0),  # a one-sided book is refused too
    ("X", 4, 10.0, 0.0, 0.0),  # a refused row does not move the feed: an earlier print is taken
    ("X", 4, 10.0, 9.9, 10.1),  # the same instant again
    ("time", 12),
    ("X", 12, 10.0, 9.9, 10.1),
    ("Y", 11, 20.0, 0.0, 0.0),  # advance_time does not move the feed either
    ("Y", 25, 0.0, 0.0, 0.0),
    ("Y", 25, 20.0, 19.9, 20.1),
]


def test_every_clock_reads_the_latest_push_of_its_symbol_between_pushes_and_in_hooks():
    entries = [(Bars(), ["X"]), (Silent(), ["X", "Y"]), (Ticker(), ["X", "Y"]), (Timed(), ["Y"])]
    node = LiveNode(entries, ["X", "Y"])
    want = {k: 0 for k in range(len(entries))}
    fed = 0
    for op in CLOCK_OPS:
        if op[0] == "time":
            node.advance_time(op[1] * NS)
            want = {k: op[1] * NS for k in want}
        else:
            symbol, t, ltp, bid, ask = op
            node.push_tick(symbol, t * NS, ltp, bid, ask)
            for k, (_, symbols) in enumerate(entries):
                if symbol in symbols:
                    want[k] = t * NS
            if ltp > 0 or (bid > 0 and ask > 0):
                fed = max(fed, t * NS)
        got = {k: s.clock.timestamp_ns() for k, (s, _) in enumerate(entries)}
        assert got == want, op
        assert node.now_ns == fed, op
    ticker, timed = entries[2][0], entries[3][0]
    assert all(clock == ts for *_, ts, clock in ticker.seen)
    assert [(kind, symbol, ts // NS) for kind, symbol, ts, _ in ticker.seen] == [
        ("trade", "X", 1), ("quote", "X", 1), ("trade", "Y", 1), ("quote", "X", 2),
        ("trade", "X", 4), ("trade", "X", 4), ("quote", "X", 4), ("trade", "X", 12),
        ("quote", "X", 12), ("trade", "Y", 11), ("trade", "Y", 25), ("quote", "Y", 25),
    ]
    assert timed.seen == [("Y", 10 * NS, 12 * NS, 12 * NS), ("Y", 20 * NS, 25 * NS, 25 * NS)]


class Alarmed(Strategy):
    """An alert at 25 s; on its print at 31 s on X it sets a timer that is due at once."""

    def __init__(self, log):
        super().__init__()
        self.events = log

    def on_start(self, ctx):
        self.clock.set_time_alert("a", 25 * NS)

    def on_trade(self, ctx, tick):
        now = self.clock.timestamp_ns()
        self.events.append(("trade", "C", ctx.symbol, ctx.event_ns, now))
        if ctx.symbol == "X" and now == 31 * NS:
            self.clock.set_timer("late", 100 * NS, start_ns=31 * NS)

    def on_timer(self, ctx, event):
        self.events.append(("timer", "C", ctx.symbol, event.name, event.ts_scheduled, event.ts_fired,
                         self.clock.timestamp_ns()))


class Every10(Strategy):
    def __init__(self, log):
        super().__init__()
        self.events = log

    def on_start(self, ctx):
        self.clock.set_timer("t", 10 * NS, start_ns=10 * NS)

    def on_bar(self, ctx):
        pass

    def on_timer(self, ctx, event):
        self.events.append(("timer", "A", ctx.symbol, event.name, event.ts_scheduled, event.ts_fired,
                         self.clock.timestamp_ns()))


def test_timers_fire_in_order_beside_strategies_that_handle_nothing():
    log = []
    node = LiveNode([(Every10(log), ["X", "Y"]), (Silent(), ["X"]), (Alarmed(log), ["X", "Y"])],
                    ["X", "Y"])
    node.push_tick("X", 5 * NS, 10.0, 9.9, 10.1)
    node.push_tick("Y", 7 * NS, 20.0)
    node.push_tick("X", 12 * NS, 0.0, 9.9, 10.1)  # a quote: A handles none, its timer fires on it
    node.push_tick("Y", 20 * NS, 0.0)  # a refused row
    node.push_tick("X", 25 * NS, 10.0)
    node.advance_time(31 * NS)
    node.push_tick("X", 31 * NS, 10.0, 9.9, 10.1)  # the timer set on the print fires on the quote
    node.push_tick("Y", 45 * NS, 20.0, 19.9, 20.1)
    s = NS
    assert log == [
        ("trade", "C", "X", 5 * s, 5 * s),
        ("trade", "C", "Y", 7 * s, 7 * s),
        ("timer", "A", "X", "t", 10 * s, 12 * s, 12 * s),
        ("timer", "A", "Y", "t", 10 * s, 20 * s, 20 * s),
        ("timer", "A", "X", "t", 20 * s, 25 * s, 25 * s),
        ("timer", "C", "X", "a", 25 * s, 25 * s, 25 * s),
        ("trade", "C", "X", 25 * s, 25 * s),
        ("timer", "A", "X", "t", 30 * s, 31 * s, 31 * s),
        ("timer", "A", "Y", "t", 30 * s, 31 * s, 31 * s),
        ("timer", "C", "Y", "a", 25 * s, 31 * s, 31 * s),
        ("trade", "C", "X", 31 * s, 31 * s),
        ("timer", "C", "X", "late", 31 * s, 31 * s, 31 * s),
        ("timer", "A", "Y", "t", 40 * s, 45 * s, 45 * s),
        ("trade", "C", "Y", 45 * s, 45 * s),
    ]


class _SharedClock(Clock):
    """A clock whose per-symbol copies are all one object."""

    def __init__(self, shared):
        super().__init__()
        self.shared = shared

    def clone_schedule(self):
        return self.shared


class OnShared(Strategy):
    def __init__(self, shared):
        super().__init__()
        self._shared = shared

    def on_start(self, ctx):
        self.clock = _SharedClock(self._shared)


@pytest.mark.parametrize("across", [False, True], ids=["one-strategy", "two-strategies"])
def test_a_clock_two_symbols_share_reads_the_latest_push(across):
    shared = Clock()
    if across:
        entries = [(OnShared(shared), ["X"]), (OnShared(shared), ["Y"])]
    else:
        entries = [(OnShared(shared), ["X", "Y"])]
    node = LiveNode(entries, ["X", "Y"])
    for symbol, t in (("X", 5), ("Y", 7), ("X", 5)):
        node.push_tick(symbol, t * NS, 10.0, 9.9, 10.1)
        assert shared.timestamp_ns() == t * NS, (symbol, t)


# -- refusals, byte for byte --------------------------------------------------------

FINITE_VALID = ["finite open, high, low and close"]


def _dict_key_error(key):
    """What this Python says of an unhashable dict key (the wording changed in 3.14)."""
    try:
        {}.get(key)
    except TypeError as e:
        return str(e)
    raise AssertionError(f"{key!r} is hashable")


UNHASHABLE = _dict_key_error(["A"])


def _nonfinite(symbol, name, shown, value, operation):
    return {
        "type": "InvalidData", "code": "non_finite_price",
        "message": f"{symbol}: {name} is not finite ({shown}) in a pushed event",
        "operation": operation, "parameter": name, "value": value, "has_value": True,
        "valid": FINITE_VALID, "skill": "market-data", "symbol": None,
    }


def _unknown(symbol, operation):
    return {
        "type": "UnknownSymbol", "code": "unknown_symbol",
        "message": f"{operation}: unknown symbol {symbol!r}", "operation": operation,
        "parameter": "symbol", "value": symbol, "has_value": True, "valid": ["A", "B", "C"],
        "skill": "market-data", "symbol": None,
    }


FINISHED = {
    "type": "StateError", "code": "node_finished", "message": "node is finished; create a new one",
    "operation": "LiveNode", "parameter": None, "value": None, "has_value": False,
    "valid": ["create a new LiveNode, or LiveNode.restore from the journal"],
    "skill": "live-trading", "symbol": None,
}


class Seen(Strategy):
    def __init__(self):
        super().__init__()
        self.seen = []

    def on_bar(self, ctx):
        self.seen.append(("bar", ctx.symbol, ctx.timestamp))

    def on_trade(self, ctx, tick):
        self.seen.append(("trade", ctx.symbol, ctx.event_ns))


def _node(config=None, **kw):
    s = Seen()
    kw.setdefault("bar_stamp", "close")
    return s, LiveNode([s], ["A", "B", "C"], config or BacktestConfig(initial_capital=1e6, fees=0.0), **kw)


def _refused(call, cls, want):
    """``call`` raises exactly ``cls``; ``want`` is its ``to_dict()`` or, for a builtin, its text."""
    with pytest.raises(BaseException) as e:
        call()
    assert type(e.value) is cls
    if isinstance(want, dict):
        assert e.value.to_dict() == want
    else:
        assert str(e.value) == want
    return e.value


TICK_OP = "LiveNode.push_tick"
TICK_REFUSALS = {
    "nan-ltp": ((NAN, 1.0, 2.0), InvalidData, _nonfinite("A", "ltp", "nan", "nan", TICK_OP)),
    "inf-bid": ((1.0, INF, 2.0), InvalidData, _nonfinite("A", "bid", "inf", "inf", TICK_OP)),
    "-inf-ask": ((1.0, 1.0, -INF), InvalidData, _nonfinite("A", "ask", "-inf", "-inf", TICK_OP)),
    "ltp-named-first": ((NAN, INF, 2.0), InvalidData, _nonfinite("A", "ltp", "nan", "nan", TICK_OP)),
    "numpy-nan": ((1.0, np.float64("nan"), 2.0), InvalidData,
                  _nonfinite("A", "bid", "nan", repr(np.float64("nan")), TICK_OP)),
    "numpy32-inf": ((1.0, 1.0, np.float32("inf")), InvalidData,
                    _nonfinite("A", "ask", "inf", repr(np.float32("inf")), TICK_OP)),
    "decimal-nan": ((decimal.Decimal("NaN"), 1.0, 2.0), InvalidData,
                    _nonfinite("A", "ltp", "NaN", "Decimal('NaN')", TICK_OP)),
    "nan-before-bad-type": ((NAN, 1.0, "x"), InvalidData, _nonfinite("A", "ltp", "nan", "nan", TICK_OP)),
    "str-ltp": (("1", 1.0, 2.0), TypeError, "must be real number, not str"),
    "none-bid": ((1.0, None, 2.0), TypeError, "must be real number, not NoneType"),
    "list-ask": ((1.0, 1.0, [2.0]), TypeError, "must be real number, not list"),
}


@pytest.mark.parametrize("case", list(TICK_REFUSALS))
def test_push_tick_refuses_a_bad_price_exactly_and_changes_nothing(case):
    prices, cls, want = TICK_REFUSALS[case]
    s, node = _node()
    _refused(lambda: node.push_tick("A", MIN, *prices), cls, want)
    assert node.push_tick("A", MIN, 1.0, 0.9, 1.1) == 2
    assert s.seen == [("trade", "A", MIN)]
    assert node.health()["ungrouped_same_timestamp_pushes"] == 0


def test_push_tick_refuses_an_unknown_symbol_exactly_and_counts_its_push():
    # A push refused for its symbol is still a push at its timestamp (``health()`` counts the
    # pushes that share one, ungrouped); a push refused for its price is not.
    s, node = _node()
    _refused(lambda: node.push_tick("Z", MIN, 1.0), UnknownSymbol, _unknown("Z", TICK_OP))
    node.push_tick("A", MIN, 1.0)
    assert node.health()["ungrouped_same_timestamp_pushes"] == 1
    with pytest.raises(InvalidData):
        node.push_tick("B", MIN, NAN)
    node.push_tick("A", MIN, 1.0)
    assert node.health()["ungrouped_same_timestamp_pushes"] == 1
    node.push_tick("C", MIN, 1.0)
    assert node.health()["ungrouped_same_timestamp_pushes"] == 2
    assert s.seen == [("trade", "A", MIN), ("trade", "A", MIN), ("trade", "C", MIN)]

    s, node = _node()
    _refused(lambda: node.push_tick(["A"], MIN, 1.0), TypeError, UNHASHABLE)
    node.push_tick("A", MIN, 1.0)
    assert node.health()["ungrouped_same_timestamp_pushes"] == 1
    assert s.seen == [("trade", "A", MIN)]


def test_push_tick_refusals_in_their_order_of_precedence():
    s, node = _node()
    node.push_tick("A", 2 * MIN, 1.0)
    _refused(lambda: node.push_tick("A", MIN, 1.0), InvalidParameter, {
        "type": "InvalidParameter", "code": "invalid_parameter",
        "message": "Invalid parameter: A live feed: timestamps must not decrease, but event 1 is "
                   f"{MIN} after {2 * MIN}",
        "operation": "Runtime.push_tick", "parameter": None, "value": None, "has_value": False,
        "valid": ["the argument the message names"], "skill": "core", "symbol": None,
    })
    _refused(lambda: node.push_tick("A", "x", 1.0), TypeError, "'str' object cannot be interpreted as an integer")
    _refused(lambda: node.push_tick("A", 3 * MIN, NAN, "x"), InvalidData,
             _nonfinite("A", "ltp", "nan", "nan", TICK_OP))
    _refused(lambda: node.push_tick("Z", MIN, NAN), InvalidData, _nonfinite("Z", "ltp", "nan", "nan", TICK_OP))
    assert s.seen == [("trade", "A", 2 * MIN)]
    node.finish()
    _refused(lambda: node.push_tick("A", 3 * MIN, NAN), StateError, FINISHED)
    _refused(lambda: node.push_tick("Z", 3 * MIN, 1.0), StateError, FINISHED)


def _config(**kw):
    c = BacktestConfig(initial_capital=1e6, fees=0.0)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _on_bars(name, value, symbol, operation):
    return {
        "type": "UnsupportedOperation", "code": "unsupported_on_bars",
        "message": f"{operation}: {name} applies to tick runs only, and {symbol!r} run on bars, where "
                   f"it would change nothing. Run on ticks, or leave {name} at its default.",
        "operation": operation, "parameter": name, "value": value, "has_value": True,
        "valid": ["run on ticks (ticks=)", "leave order_latency_ns at 0 and partial_fills False"],
        "skill": "core", "symbol": None,
    }


def _stateful(code, message, operation, value, valid):
    return {
        "type": "StateError", "code": code, "message": message, "operation": operation,
        "parameter": "timestamp", "value": value, "has_value": True, "valid": valid,
        "skill": "live-trading", "symbol": None,
    }


def _duplicate(symbol, ts, operation):
    return _stateful("duplicate_bar", f"{symbol!r} already has a bar at {ts}: this one repeats it, drop the retry",
                     operation, ts, ["push each bar once"])


def _conflicting(symbol, ts, operation):
    return _stateful("conflicting_bar", f"{symbol!r} already has a bar at {ts} with other values: a bar is not revised",
                     operation, ts, ["a delivered bar is not revised: push the next bar"])


BAR_OP, GROUP_OP = "LiveNode.push_bar", "LiveNode.push_group"
BAR_REFUSALS = {
    "nan-open": ((NAN, 101.0, 99.0, 100.5, 10.0), InvalidData, _nonfinite("A", "open", "nan", "nan", BAR_OP)),
    "inf-high": ((100.0, INF, 99.0, 100.5, 10.0), InvalidData, _nonfinite("A", "high", "inf", "inf", BAR_OP)),
    "nan-low": ((100.0, 101.0, NAN, 100.5, 10.0), InvalidData, _nonfinite("A", "low", "nan", "nan", BAR_OP)),
    "-inf-close": ((100.0, 101.0, 99.0, -INF, 10.0), InvalidData,
                   _nonfinite("A", "close", "-inf", "-inf", BAR_OP)),
    "numpy-nan-close": ((100.0, 101.0, 99.0, np.float64("nan"), 10.0), InvalidData,
                        _nonfinite("A", "close", "nan", repr(np.float64("nan")), BAR_OP)),
    "open-named-first": ((NAN, 101.0, 99.0, NAN, 10.0), InvalidData, _nonfinite("A", "open", "nan", "nan", BAR_OP)),
    "str-high": ((100.0, "x", 99.0, 100.5, 10.0), TypeError, "must be real number, not str"),
    "none-open": ((None, 101.0, 99.0, 100.5, 10.0), TypeError, "must be real number, not NoneType"),
    "str-volume": ((100.0, 101.0, 99.0, 100.5, "v"), TypeError, "must be real number, not str"),
}


@pytest.mark.parametrize("case", list(BAR_REFUSALS))
def test_push_bar_refuses_a_bad_bar_exactly_and_changes_nothing(case):
    bar, cls, want = BAR_REFUSALS[case]
    s, node = _node()
    _refused(lambda: node.push_bar("A", MIN, *bar), cls, want)
    node.push_bar("A", MIN, *GOOD)
    assert s.seen == [("bar", "A", MIN)]


@pytest.mark.parametrize("case", [c for c in BAR_REFUSALS if BAR_REFUSALS[c][1] is InvalidData])
def test_push_group_refuses_a_bad_bar_exactly_and_delivers_none_of_the_group(case):
    bar, cls, want = BAR_REFUSALS[case]
    s, node = _node()
    want = {**want, "message": want["message"].replace("A:", "B:", 1), "operation": GROUP_OP}
    _refused(lambda: node.push_group(MIN, {"A": GOOD, "B": bar, "C": GOOD}), cls, want)
    assert node.open_group is None and s.seen == []
    node.push_group(MIN, {"A": GOOD, "B": GOOD, "C": GOOD})
    assert s.seen == [("bar", "A", MIN), ("bar", "B", MIN), ("bar", "C", MIN)]


def test_push_bar_and_push_group_refusals_in_their_order_of_precedence():
    # A setting bars cannot honour comes before a bad price.
    s, node = _node(_config(partial_fills=True))
    _refused(lambda: node.push_bar("A", MIN, NAN, 1.0, 1.0, 1.0, 1.0), UnsupportedOperation,
             _on_bars("partial_fills", True, "A", BAR_OP))
    _refused(lambda: node.push_group(MIN, {"A": GOOD}), UnsupportedOperation,
             _on_bars("partial_fills", True, "A", GROUP_OP))
    assert node.push_tick("A", MIN, 1.0) == 1
    s, node = _node(_config(order_latency_ns=5))
    _refused(lambda: node.push_bar("B", MIN, *GOOD), UnsupportedOperation,
             _on_bars("order_latency_ns", 5, "B", BAR_OP))
    _refused(lambda: node.push_group(MIN, {"B": GOOD, "A": GOOD}), UnsupportedOperation,
             _on_bars("order_latency_ns", 5, "B", GROUP_OP))
    assert s.seen == []

    # A bad price comes before a repeated bar; a repeat is told from a revision.
    s, node = _node()
    node.push_bar("A", MIN, *GOOD)
    _refused(lambda: node.push_bar("A", MIN, NAN, 1.0, 1.0, 1.0, 1.0), InvalidData,
             _nonfinite("A", "open", "nan", "nan", BAR_OP))
    _refused(lambda: node.push_bar("A", MIN, *GOOD), StateError, _duplicate("A", MIN, BAR_OP))
    _refused(lambda: node.push_bar("A", MIN, 1.0, 1.0, 1.0, 1.0, 1.0), StateError, _conflicting("A", MIN, BAR_OP))
    _refused(lambda: node.push_bar("A", MIN, 100, 101, 99, 100, 10), StateError, _conflicting("A", MIN, BAR_OP))
    _refused(lambda: node.push_bar("Z", MIN, *GOOD), UnknownSymbol, _unknown("Z", BAR_OP))
    _refused(lambda: node.push_bar(["A"], MIN, *GOOD), TypeError, UNHASHABLE)
    _refused(lambda: node.push_bar("A", 0, *GOOD), InvalidParameter, {
        "type": "InvalidParameter", "code": "invalid_parameter",
        "message": f"Invalid parameter: A live feed: timestamps must not decrease, but event 1 is 0 after {MIN}",
        "operation": "Runtime.push_bar", "parameter": None, "value": None, "has_value": False,
        "valid": ["the argument the message names"], "skill": "core", "symbol": None,
    })
    assert s.seen == [("bar", "A", MIN)]

    # Inside an open group: a bar stamped otherwise is refused before a repeat is looked at.
    node.begin_group(2 * MIN)
    mismatch = _stateful("group_timestamp_mismatch",
                         f"the open group is at {2 * MIN}: a bar for 'A' stamped {MIN} is not part of it",
                         BAR_OP, MIN, [2 * MIN])
    _refused(lambda: node.push_bar("A", MIN, *GOOD), StateError, mismatch)
    _refused(lambda: node.push_group(3 * MIN, {"B": GOOD}), StateError, _stateful(
        "group_timestamp_mismatch", f"the open group is at {2 * MIN}: a bar for 'B' stamped {3 * MIN} is not part of it",
        GROUP_OP, 3 * MIN, [2 * MIN]))
    _refused(lambda: node.push_group(2 * MIN, {"B": GOOD}), StateError, {
        "type": "StateError", "code": "group_open",
        "message": "a timestamp group is already open: flush() it before opening another",
        "operation": "Runtime.begin_group", "parameter": None, "value": None, "has_value": False,
        "valid": ["flush() the open group first"], "skill": "live-trading", "symbol": None,
    })
    node.push_bar("A", 2 * MIN, *GOOD)
    node.flush()
    assert s.seen == [("bar", "A", MIN), ("bar", "A", 2 * MIN)]

    # A group with any refused bar delivers none of it.
    for bars, cls, want in (
        ({"C": GOOD, "Z": GOOD}, UnknownSymbol, _unknown("Z", GROUP_OP)),
        ({"C": GOOD, "A": GOOD}, StateError, _duplicate("A", 2 * MIN, GROUP_OP)),
        ({"C": GOOD, "A": (1.0,) * 5}, StateError, _conflicting("A", 2 * MIN, GROUP_OP)),
        ({"C": GOOD, "B": {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}}, KeyError, "'volume'"),
        ({"C": GOOD, "B": (1.0, 1.0, 1.0, 1.0)}, ValueError, "not enough values to unpack (expected 5, got 4)"),
        ({"C": GOOD, "B": (1.0, "x", 1.0, 1.0, 1.0)}, TypeError, "must be real number, not str"),
    ):
        _refused(lambda: node.push_group(2 * MIN, bars), cls, want)
        assert node.open_group is None
    _refused(lambda: node.push_group(MIN, {"B": GOOD, "A": GOOD}), InvalidParameter, {
        "type": "InvalidParameter", "code": "invalid_parameter",
        "message": f"Invalid parameter: A live feed: timestamps must not decrease, but event 2 is {MIN} after {2 * MIN}",
        "operation": "Runtime.check_bar", "parameter": None, "value": None, "has_value": False,
        "valid": ["the argument the message names"], "skill": "core", "symbol": None,
    })
    assert s.seen == [("bar", "A", MIN), ("bar", "A", 2 * MIN)]
    node.push_group(2 * MIN, {"C": GOOD, "B": {"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 3}})
    assert s.seen[2:] == [("bar", "B", 2 * MIN), ("bar", "C", 2 * MIN)]

    node.finish()
    _refused(lambda: node.push_bar("A", 3 * MIN, NAN, 1.0, 1.0, 1.0, 1.0), StateError, FINISHED)
    _refused(lambda: node.push_group(3 * MIN, {"A": GOOD}), StateError, FINISHED)


def test_a_bar_needs_its_stamp_named_before_anything_else_is_checked():
    s, node = _node(bar_stamp=None)

    def missing(operation):
        return {
            "type": "InvalidParameter", "code": "missing_bar_stamp",
            "message": f"{operation}: bar_stamp is required: say whether the bar timestamps are bar opens "
                       "('open', with bar_ns, the bar width in nanoseconds) or bar closes ('close')",
            "operation": operation, "parameter": "bar_stamp", "value": None, "has_value": True,
            "valid": ["open", "close"], "skill": "timeframes", "symbol": None,
        }

    _refused(lambda: node.push_bar("A", MIN, *GOOD), InvalidParameter, missing(BAR_OP))
    _refused(lambda: node.push_group(MIN, {"A": GOOD}), InvalidParameter, missing(GROUP_OP))
    # The price is checked before the stamp is asked for.
    _refused(lambda: node.push_bar("A", MIN, NAN, 1.0, 1.0, 1.0, 1.0), InvalidData,
             _nonfinite("A", "open", "nan", "nan", BAR_OP))
    assert s.seen == []


# -- groups -------------------------------------------------------------------------


@pytest.mark.parametrize("order, want", [
    (("A", "B", "C"), ["A", "C", "B"]),
    (("B", "A", "C"), ["A", "C", "B"]),
    # B closes after A and C: pushed between them, it completes C before A has arrived.
    (("C", "B", "A"), ["C", "A", "B"]),
])
def test_a_group_of_bars_of_different_widths_is_delivered_by_close_time(order, want):
    s, node = _node(bar_stamp="open", bar_ns={"A": MIN, "B": 5 * MIN, "C": MIN})
    for k in range(3):
        node.push_group(10 * k * MIN, {sym: (100.0 + k, 101.0 + k, 99.0 + k, 100.5 + k, 10.0) for sym in order})
    assert s.seen == [("bar", sym, 10 * k * MIN) for k in range(3) for sym in want]
    assert s.clock.timestamp_ns() == 25 * MIN


def test_a_group_journals_and_delivers_what_its_bars_pushed_one_by_one_do():
    def run(grouped):
        journal = Journal({"case": "group"}, memory=10_000)
        s, node = _node(bar_stamp="open", bar_ns={"A": MIN, "B": 5 * MIN, "C": MIN}, journal=journal)
        for k in range(3):
            bars = {"B": GOOD, "A": (100.0 + k, 101.0 + k, 99.0 + k, 100.5 + k, 10.0), "C": GOOD}
            if grouped:
                node.push_group(10 * k * MIN, bars, arrival_ns=10 * k * MIN + 7)
            else:
                node.begin_group(10 * k * MIN)
                for symbol, bar in bars.items():
                    node.push_bar(symbol, 10 * k * MIN, *bar, arrival_ns=10 * k * MIN + 7)
                node.flush()
        with pytest.raises(StateError):
            node.push_group(20 * MIN, {"C": GOOD})
        node.push_tick("A", 21 * MIN + 1, 100.0)
        records = [(r.seq, r.kind, r.ts_event, r.ts_arrival, r.source, r.payload, r.chain)
                   for r in journal.since(0)]
        return s.seen, records, journal.head()[:2]

    grouped, one_by_one = run(True), run(False)
    assert [kind for _, kind, *_ in grouped[1]].count("bar") == 9
    assert grouped == one_by_one


def test_push_group_calls_a_push_bar_a_subclass_overrides_once_per_symbol():
    calls = []

    class Node(LiveNode):
        def push_bar(self, symbol, timestamp, open, high, low, close, volume, *, arrival_ns=None):
            calls.append((symbol, timestamp, (open, high, low, close, volume), arrival_ns))
            super().push_bar(symbol, timestamp, open, high, low, close, volume, arrival_ns=arrival_ns)

    s = Seen()
    node = Node([s], ["A", "B", "C"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close")
    node.push_group(MIN, {"C": GOOD, "A": GOOD, "B": GOOD}, arrival_ns=MIN + 7)
    assert calls == [("C", MIN, GOOD, MIN + 7), ("A", MIN, GOOD, MIN + 7), ("B", MIN, GOOD, MIN + 7)]
    assert s.seen == [("bar", "A", MIN), ("bar", "B", MIN), ("bar", "C", MIN)]
    with pytest.raises(StateError):
        node.push_group(MIN, {"A": GOOD})
    assert len(calls) == 3
