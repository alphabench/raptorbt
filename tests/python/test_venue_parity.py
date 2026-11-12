"""A backtest and a paper or live run of one strategy say the same thing.

Each test runs one scripted strategy through ``run_strategies`` (the batch),
an in-process ``LiveNode`` and a ``LiveNode`` over ``PaperExecution``, and
asserts hand-computed numbers. They are the regression tests of the
simulation rules: settlement, cross-instrument
orders, symbol order, lot and tick terms, entry-bar protection, sizing,
marketable limits and silent settings.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

import raptorbt as rb
from raptorbt import BacktestConfig, LiveNode, Strategy, run_strategies
from raptorbt.errors import InvalidOrder, InvalidParameter, UnsupportedOperation
from raptorbt.execution import PaperExecution
from raptorbt.strategy import orders as O

MIN = 60_000_000_000
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
PATHS = ("batch", "node", "paper")


def ist(y, m, d, hh, mm):
    return int(dt.datetime(y, m, d, hh, mm, tzinfo=IST).timestamp()) * 10**9


def fmt(ns):
    return dt.datetime.fromtimestamp(ns / 1e9, IST).strftime("%m-%d %H:%M")


def bars(rows, t0):
    """OHLC rows stamped at their close; ``t0`` is the first bar's open."""
    a = np.array(rows, dtype=float)
    ts = (
        np.array(t0, dtype=np.int64)
        if isinstance(t0, list)
        else np.array([t0 + (i + 1) * MIN for i in range(len(rows))], dtype=np.int64)
    )
    return {
        "timestamps": ts,
        "open": a[:, 0],
        "high": a[:, 1],
        "low": a[:, 2],
        "close": a[:, 3],
        "volume": np.full(len(rows), 1e6),
    }


class Script(Strategy):
    """Runs ``script[(symbol, n)]`` or ``script[n]`` on the n-th bar of a symbol."""

    def __init__(self, script):
        super().__init__()
        self.script, self.n = script, {}

    def on_bar(self, ctx, bar=None):
        n = self.n.get(ctx.symbol, 0)
        self.n[ctx.symbol] = n + 1
        act = self.script.get((ctx.symbol, n)) or self.script.get(n)
        if act:
            act(self, ctx)


def rows_of(data):
    """Every bar of every symbol in delivery order: time, then symbol."""
    return sorted(
        (int(data[s]["timestamps"][i]), s, i) for s in data for i in range(len(data[s]["timestamps"]))
    )


def run(script, data, cfg, paths=PATHS, strategy=None, **kw):
    """``{path: result}`` for the scripted strategy on each path."""
    out = {}
    for path in paths:
        s = strategy() if strategy else Script(script)
        if path == "batch":
            out[path] = run_strategies([s], data, cfg(), bar_stamp="close", **kw)[0]
            continue
        node = LiveNode(
            [s],
            sorted(data),
            cfg(),
            bar_stamp="close",
            execution=PaperExecution() if path == "paper" else None,
            **kw,
        )
        for ts, sym, i in rows_of(data):
            a = data[sym]
            node.push_bar(sym, ts, *(float(a[c][i]) for c in ("open", "high", "low", "close", "volume")))
        out[path] = node.finish()[0]
    return out


def trades(result):
    return [
        (t.symbol, round(t.entry_price, 6), round(t.exit_price, 6), round(t.size, 6), fmt(t.exit_time))
        for t in result.result.trades()
    ]


class Venue:
    """A recording venue: it keeps the requests the node sends it."""

    namespace = "T1"
    keeps_own_records = True  # a test venue: it remembers the node's orders itself

    def __init__(self):
        self.requests = []

    def submit(self, r):
        self.requests.append(r)

    def cancel(self, cid):
        pass

    def modify(self, cid, **kw):
        pass


# ---------------------------------------------------------------------------
# expiry settlement
# ---------------------------------------------------------------------------

EXPIRY = ist(2026, 11, 26, 15, 30)
EXPIRY_TS = [ist(2026, 11, 26, 15, m) for m in (27, 28, 29, 30)] + [ist(2026, 11, 27, 9, 16)]
UNDERLYING = bars(
    [(24100,) * 4, (24100,) * 4, (24180,) * 4, (24200,) * 4, (24300,) * 4], EXPIRY_TS
)


def call_option(price="underlying"):
    o = rb.InstrumentSpec.option(
        "C", strike=24000.0, right="call", expiration_ns=EXPIRY, lot_size=75.0, underlying="NIFTY",
        venue="NFO",
    )
    o.settlement_price = price
    return o


def expiry_cfg():
    return BacktestConfig(initial_capital=1e6, fees=0.0)


@pytest.mark.parametrize("own_bar_at_expiry", [True, False])
@pytest.mark.parametrize("order", [("C", "NIFTY"), ("NIFTY", "C")])
def test_settlement_does_not_depend_on_symbol_order(order, own_bar_at_expiry):
    """The underlying's 24200 print at 15:30 settles the call at 200, whichever symbol is listed
    first and whether or not the option has a bar of its own at the expiry instant (75 x (200 -
    150) = 3750)."""
    option_bars = bars(
        [(150,) * 4, (150,) * 4, (180,) * 4, (195,) * 4][: 4 if own_bar_at_expiry else 3],
        EXPIRY_TS[: 4 if own_bar_at_expiry else 3],
    )
    all_data = {"C": option_bars, "NIFTY": UNDERLYING}
    data = {k: all_data[k] for k in order}
    out = run(
        {("C", 1): lambda s, c: s.submit_order(O.Market(side="buy", units=75.0))},
        data,
        expiry_cfg,
        paths=("batch", "node"),
        instruments={"C": call_option(), "NIFTY": rb.InstrumentSpec.equity("NIFTY")},
    )
    for path, result in out.items():
        assert trades(result) == [("C", 150.0, 200.0, 75.0, "11-26 15:30")], path
        assert result.result.trades()[0].pnl == pytest.approx(3750.0)


@pytest.mark.parametrize("order", [("C", "NIFTY"), ("NIFTY", "C")])
def test_auto_settlement_uses_the_underlying_when_it_is_in_the_run(order):
    """An index option cash-settles at intrinsic value against the underlying's print, not at the
    option's last trade (150), when the underlying is registered and the option has no bar at the
    expiry."""
    all_data = {"C": bars([(150,) * 4, (150,) * 4], EXPIRY_TS[:2]), "NIFTY": UNDERLYING}
    out = run(
        {("C", 0): lambda s, c: s.submit_order(O.Market(side="buy", units=75.0))},
        {k: all_data[k] for k in order},
        expiry_cfg,
        paths=("batch",),
        instruments={"C": call_option("auto"), "NIFTY": rb.InstrumentSpec.equity("NIFTY")},
    )
    assert trades(out["batch"]) == [("C", 150.0, 200.0, 75.0, "11-26 15:30")]


def test_auto_settlement_refuses_an_index_option_when_the_run_has_no_underlying_price():
    """An index option cash-settles at the official index close. With no
    underlying in the run, ``auto`` does not settle it at the option's own last trade (150): the
    settlement is refused like ``underlying`` is, the strategy hears why, and the run's own end
    closes the position (``EndOfData``). ``settlement_price="last_price"`` still asks for the
    option's own price."""
    class Holder(Strategy):
        def __init__(self):
            super().__init__()
            self.refused = []

        def on_bar(self, ctx):
            if ctx.idx == 0:
                self.submit_order(O.Market(side="buy", units=75.0))

        def on_contract_event(self, ctx, event):
            if event.kind == "lifecycle_refused":
                self.refused.append((event.reject_reason, event.outcome))

    def go(price):
        s = Holder()
        out = rb.run_strategies(
            [s], {"C": bars([(150,) * 4] * 4, EXPIRY_TS[:4])}, expiry_cfg(),
            instruments={"C": call_option(price)}, bar_stamp="close",
        )
        return s, out[0].result.trades()

    auto, trades_auto = go("auto")
    assert auto.refused == [("underlying_price_missing", "settlement")]
    assert [str(t.exit_reason) for t in trades_auto] == ["EndOfData"]

    last, trades_last = go("last_price")
    assert last.refused == []
    assert [str(t.exit_reason) for t in trades_last] == ["Settlement"]
    assert trades_last[0].exit_price == 150.0


# ---------------------------------------------------------------------------
# an order on another instrument
# ---------------------------------------------------------------------------

T0 = ist(2026, 11, 2, 9, 15)


def two_instruments(n=6):
    base = {}
    for k, s in enumerate(("IDX", "FA")):
        c = 100.0 + 10 * k + np.arange(n) * 1.0
        base[s] = {
            "timestamps": T0 + (np.arange(n, dtype=np.int64) + 1) * MIN,
            "open": c - 0.5,
            "high": c + 1,
            "low": c - 1,
            "close": c,
            "volume": np.full(n, 100.0),
        }
    return base


class BuyFAOnIDX(Strategy):
    """On the second IDX bar (09:17), buy 3 units of FA."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def on_bar(self, ctx, bar):
        if ctx.symbol != "IDX":
            return
        self.n += 1
        if self.n == 2:
            self.submit_order(O.Market(side="buy", units=3.0, instrument="FA"))


@pytest.mark.parametrize("order", [("IDX", "FA"), ("FA", "IDX")])
def test_next_bar_open_order_on_another_instrument_fills_at_its_next_open(order):
    """Decided on IDX at 09:17, FA's order fills at FA's next open (09:18, 111.5): in the batch,
    in a node and over paper, and not by panic."""
    base = two_instruments()
    data = {k: base[k] for k in order}
    out = run(
        None,
        data,
        lambda: BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing="next_bar_open"),
        strategy=BuyFAOnIDX,
    )
    for path, result in out.items():
        assert [t[:4] for t in trades(result)] == [("FA", 111.5, 115.0, 3.0)], path


def test_auto_roll_under_next_bar_open_closes_and_reopens_without_a_panic():
    from raptorbt.strategy import Future, Holding, Requirements, Underlying
    from raptorbt.strategy.continuous import RollEvents

    exp_a, exp_b = T0 + 20 * MIN, T0 + 400 * MIN
    specs = {
        "FA": rb.InstrumentSpec.futures_contract("FA", exp_a, 1.0, underlying="IDX"),
        "FB": rb.InstrumentSpec.futures_contract("FB", exp_b, 1.0, underlying="IDX"),
    }

    class Carry(RollEvents, Strategy):
        requirements = Requirements(
            roles={
                "index": Underlying(asset="index"),
                "fut": Future(of="index", contract="continuous", roll_days=5 / 1440),
            },
            timeframes=["1m"],
            holding=Holding.overnight(),
            data=["bar"],
        )
        auto_roll = True

        def __init__(self):
            super().__init__()
            self.n = 0

        def on_bar(self, ctx, bar):
            if ctx.symbol != "IDX":
                return
            self.n += 1
            if self.n == 1:
                self.submit_order(O.Market(side="buy", units=3.0, instrument=ctx.role("fut")))

    n = 30
    data = {}
    for k, s in enumerate(("IDX", "FA", "FB")):
        c = 100.0 + 10 * k + np.arange(n) * 0.1
        data[s] = {
            "timestamps": T0 + (np.arange(n, dtype=np.int64) + 1) * MIN,
            "open": c - 0.05, "high": c + 1, "low": c - 1, "close": c,
            "volume": np.full(n, 100.0),
        }
    r = run_strategies(
        [Carry()], data, BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing="next_bar_open"),
        bar_stamp="close", instruments=specs, bindings={"index": "IDX"},
    )[0]
    assert [(t.symbol, round(t.entry_price, 2), round(t.exit_price, 2)) for t in r.result.trades()] == [
        ("FA", 110.05, 111.45),
        ("FB", 121.45, 122.9),
    ]


# ---------------------------------------------------------------------------
# lot and tick terms are refused, not floored or snapped
# ---------------------------------------------------------------------------

FUT_EXP = ist(2026, 11, 24, 15, 30)
LOT_ROWS = [(100, 101, 99, 100), (100.02, 104, 99, 100.03), (105, 106, 104, 105.07), (104, 105, 102, 103)]


def future():
    return {
        "F": rb.InstrumentSpec.futures_contract(
            "F", FUT_EXP, lot_size=75.0, multiplier=1.0, price_increment=0.05
        )
    }


OFF_TERMS = [
    ("off-lot market", lambda s, c: s.submit_order(O.Market(side="buy", units=100.0)), "off_lot_quantity", "units", ["75.0", "150.0"]),
    ("off-lot below one lot", lambda s, c: s.submit_order(O.Market(side="buy", units=10.0)), "off_lot_quantity", "units", ["75.0"]),
    ("off-tick limit", lambda s, c: s.submit_order(O.Limit(side="buy", price=100.03, units=75.0)), "off_tick_price", "limit_price", ["100.0", "100.05"]),
    ("off-tick stop", lambda s, c: s.submit_order(O.StopMarket(side="buy", trigger=100.07, units=75.0)), "off_tick_price", "trigger_price", ["100.05", "100.1"]),
    (
        "off-tick bracket",
        lambda s, c: s.submit_bracket(O.Market(side="buy", units=75.0), stop_trigger=97.33, target_price=103.21),
        "off_tick_price",
        "trigger_price",
        ["97.3", "97.35"],
    ),
    ("off-tick entry stop", lambda s, c: s.enter(size_frac=1.0, stop_price=97.33), "off_tick_price", "stop_price", ["97.3", "97.35"]),
]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("name,act,code,param,valid", OFF_TERMS, ids=[t[0] for t in OFF_TERMS])
def test_off_grid_terms_are_refused_in_every_mode(name, act, code, param, valid, path):
    with pytest.raises(InvalidOrder) as e:
        run({1: act}, {"F": bars(LOT_ROWS, T0)}, lambda: BacktestConfig(initial_capital=10000.0, fees=0.0),
            paths=(path,), instruments=future())
    assert e.value.code == code
    assert e.value.parameter == param
    assert [str(float(v)) for v in e.value.valid] == valid


def test_the_venue_is_never_sent_an_off_grid_term():
    """The simulation refuses what the venue would be sent as written: nothing off the grid
    reaches the recording venue, and on-grid terms arrive unchanged."""
    for act in (o[1] for o in OFF_TERMS[:5]):
        venue = Venue()
        node = LiveNode(
            [Script({1: act})], ["F"], BacktestConfig(initial_capital=10000.0, fees=0.0),
            bar_stamp="close", execution=venue, instruments=future(),
        )
        a = bars(LOT_ROWS, T0)
        with pytest.raises(InvalidOrder):
            for i in range(2):
                node.push_bar("F", int(a["timestamps"][i]), *(float(a[c][i]) for c in ("open", "high", "low", "close", "volume")))
        assert venue.requests == []
    venue = Venue()
    node = LiveNode(
        [Script({1: lambda s, c: s.submit_order(O.Limit(side="buy", price=100.05, units=150.0))})], ["F"],
        BacktestConfig(initial_capital=100000.0, fees=0.0), bar_stamp="close", execution=venue, instruments=future(),
    )
    a = bars(LOT_ROWS, T0)
    for i in range(2):
        node.push_bar("F", int(a["timestamps"][i]), *(float(a[c][i]) for c in ("open", "high", "low", "close", "volume")))
    assert [(r.quantity, r.limit_price) for r in venue.requests] == [(150.0, 100.05)]


@pytest.mark.parametrize("path", PATHS)
def test_capital_fraction_sizing_is_a_whole_number_of_lots(path):
    """20000 of capital at 100.03 x 1 buys 199 units; a lot is 75, so the entry is 150 (2 lots),
    and a fraction that does not afford one lot (10000 -> 0 lots) opens nothing."""
    data = {"F": bars(LOT_ROWS, T0)}
    two = run({1: lambda s, c: s.enter(size_frac=1.0), 3: lambda s, c: s.close_position()}, data,
              lambda: BacktestConfig(initial_capital=20000.0, fees=0.0), paths=(path,), instruments=future())
    assert [t[3] for t in trades(two[path])] == [150.0]
    none = run({1: lambda s, c: s.enter(size_frac=0.5)}, data,
               lambda: BacktestConfig(initial_capital=10000.0, fees=0.0), paths=(path,), instruments=future())
    assert trades(none[path]) == []


def whole_units(size_increment):
    """A cash equity and a contract quoted in lots (an MCX future: lot 1), both lot 1.0, with
    the size step given. A lot of 1.0 is the neutral default: only the step makes units whole."""
    return {
        "equity": {"F": rb.InstrumentSpec.equity("F", price_increment=0.05, size_increment=size_increment)},
        "future": {
            "F": rb.InstrumentSpec.futures_contract(
                "F", FUT_EXP, lot_size=1.0, price_increment=0.05, size_increment=size_increment
            )
        },
    }


@pytest.mark.parametrize("kind", ["equity", "future"])
@pytest.mark.parametrize("path", PATHS)
def test_a_size_step_of_one_sizes_a_capital_fraction_in_whole_units(kind, path):
    """20000 of capital at 100.03 is 199.94 units: 199 whole units with a size step of 1.0, and
    the fraction itself (a fractional share no account holds) with the neutral lot alone."""
    data = {"F": bars(LOT_ROWS, T0)}
    script = {1: lambda s, c: s.enter(size_frac=1.0), 3: lambda s, c: s.close_position()}
    def cfg():
        return BacktestConfig(initial_capital=20000.0, fees=0.0)

    whole = run(script, data, cfg, paths=(path,), instruments=whole_units(1.0)[kind])
    assert [t[3] for t in trades(whole[path])] == [199.0]
    (neutral,) = [t[3] for t in trades(run(script, data, cfg, paths=(path,), instruments=whole_units(0.0)[kind])[path])]
    assert 199.0 < neutral < 200.0


@pytest.mark.parametrize("kind", ["equity", "future"])
@pytest.mark.parametrize("path", PATHS)
def test_a_fractional_quantity_off_a_size_step_of_one_is_refused(kind, path):
    with pytest.raises(InvalidOrder) as e:
        run({1: lambda s, c: s.submit_order(O.Market(side="buy", units=2.5))}, {"F": bars(LOT_ROWS, T0)},
            lambda: BacktestConfig(initial_capital=10000.0, fees=0.0), paths=(path,),
            instruments=whole_units(1.0)[kind])
    assert (e.value.code, e.value.parameter) == ("off_lot_quantity", "units")
    assert [float(v) for v in e.value.valid] == [2.0, 3.0]


def test_the_option_factory_takes_a_size_step():
    spec = rb.InstrumentSpec.option("O", 100.0, "call", 10**18, 1.0, size_increment=1.0)
    assert (spec.lot_size, spec.size_increment) == (1.0, 1.0)
    assert rb.InstrumentSpec.option("O", 100.0, "call", 10**18, 1.0).size_increment == 0.0


# ---------------------------------------------------------------------------
# one rule for the entry bar's protective legs
# ---------------------------------------------------------------------------

PROTECT_BASE = [(100, 101, 99, 100), (100, 101, 99, 100)]
PROTECT_CASES = {
    # entry at the 09:18 open; the rest of that bar trades through both legs: the stop is first
    "both legs touched": ([(100, 106, 94, 101), (101, 102, 100, 101.5), (101.5,) * 4], 100.0, 95.0),
    # the stop alone is touched
    "stop only": ([(100, 101, 94, 99), (99, 99, 99, 99), (99,) * 4], 100.0, 95.0),
    # the target alone is touched
    "target only": ([(100, 106, 99, 101), (101, 102, 100, 101.5), (101.5,) * 4], 100.0, 105.0),
    # the entry opens below the stop: the stop trades at the open
    "gap below the stop": ([(93, 94, 90, 92), (92.5, 93, 91, 91.5), (91,) * 4], 93.0, 93.0),
    # the entry opens above the target: the target is marketable on arrival and trades at the open
    "gap above the target": ([(107, 108, 106, 107), (106.5, 107, 106, 106.2), (106,) * 4], 107.0, 107.0),
}
PROTECT_APIS = {
    "enter": lambda s, c: s.enter(size_frac=0.1, stop_price=95.0, target_price=105.0),
    "typed": lambda s, c: s.submit_order(O.Market(side="buy", units=10.0, stop_price=95.0, target_price=105.0)),
    "bracket": lambda s, c: s.submit_bracket(O.Market(side="buy", units=10.0), stop_trigger=95.0, target_price=105.0),
}


@pytest.mark.parametrize("api", PROTECT_APIS)
@pytest.mark.parametrize("case", PROTECT_CASES)
def test_entry_bar_protection_is_one_rule_for_every_api_and_mode(case, api):
    tail, entry, exit_ = PROTECT_CASES[case]
    out = run(
        {1: PROTECT_APIS[api]}, {"X": bars(PROTECT_BASE + tail, T0)},
        lambda: BacktestConfig(initial_capital=10000.0, fees=0.0, fill_timing="next_bar_open"),
    )
    for path, result in out.items():
        got = [(t.entry_price, t.exit_price, fmt(t.exit_time)) for t in result.result.trades()]
        assert got == [(entry, exit_, "11-02 09:18")], (path, got)


@pytest.mark.parametrize("path", PATHS)
def test_configured_fixed_stop_also_acts_on_the_entry_bar(path):
    def cfg():
        c = BacktestConfig(initial_capital=10000.0, fees=0.0, fill_timing="next_bar_open")
        c.set_fixed_stop(0.05)
        return c

    out = run({1: lambda s, c: s.enter(size_frac=0.1)}, {"X": bars(PROTECT_BASE + PROTECT_CASES["stop only"][0], T0)},
              cfg, paths=(path,))
    assert [(t.entry_price, t.exit_price, fmt(t.exit_time)) for t in out[path].result.trades()] == [
        (100.0, 95.0, "11-02 09:18")
    ]


# ---------------------------------------------------------------------------
# capital-fraction sizing
# ---------------------------------------------------------------------------

SIZE_ROWS = [(100, 101, 99, 100), (102, 104, 101, 103), (105, 106, 104, 105), (104, 105, 102, 103)]


@pytest.mark.parametrize("timing", ["same_bar_close", "next_bar_open"])
@pytest.mark.parametrize("kind", ["enter", "typed"])
def test_venue_is_sent_the_quantity_sized_at_the_decision_price(kind, timing):
    """Decided on the 09:17 bar (close 103): half of 10000 buys 5000/103 units, the quantity the
    venue is sent under either timing (under next_bar_open the fill, at 105, does not resize it)."""
    act = (
        (lambda s, c: s.enter(size_frac=0.5))
        if kind == "enter"
        else (lambda s, c: s.submit_order(O.Market(side="buy", size_frac=0.5)))
    )
    venue = Venue()
    node = LiveNode(
        [Script({1: act})], ["X"], BacktestConfig(initial_capital=10000.0, fees=0.0, fill_timing=timing),
        bar_stamp="close", execution=venue,
    )
    a = bars(SIZE_ROWS, T0)
    for i in range(2):
        node.push_bar("X", int(a["timestamps"][i]), *(float(a[c][i]) for c in ("open", "high", "low", "close", "volume")))
    assert [r.quantity for r in venue.requests] == [pytest.approx(5000 / 103)]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("kind", ["enter", "typed"])
def test_next_bar_open_fraction_entry_is_the_same_size_in_batch_node_and_paper(kind, path):
    """The entry fills at the 09:18 open (105) with the quantity sized at 103."""
    act = (
        (lambda s, c: s.enter(size_frac=0.5))
        if kind == "enter"
        else (lambda s, c: s.submit_order(O.Market(side="buy", size_frac=0.5)))
    )
    out = run(
        {1: act},
        {"X": bars(SIZE_ROWS, T0)},
        lambda: BacktestConfig(initial_capital=10000.0, fees=0.0, fill_timing="next_bar_open"),
        paths=(path,),
    )
    ((_, entry, _, size, _),) = trades(out[path])
    assert entry == 105.0
    assert size == pytest.approx(5000 / 103, rel=1e-6)


# ---------------------------------------------------------------------------
# a limit marketable when sent
# ---------------------------------------------------------------------------

LIMIT_BASE = [(100, 101, 99, 100), (102, 104, 101, 103)]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    "price,tail,entry,at",
    [
        (104.0, [(105, 106, 104.2, 105.5), (105.5,) * 4], 103.0, "11-02 09:17"),
        (104.0, [(105, 106, 103.5, 105), (105,) * 4], 103.0, "11-02 09:17"),
        (104.0, [(102.5, 103, 102, 102.8), (103,) * 4], 103.0, "11-02 09:17"),
        (101.5, [(102, 102.5, 101, 102), (102,) * 4], 101.5, "11-02 09:18"),
        (101.5, [(100, 100.5, 99.5, 100), (100,) * 4], 100.0, "11-02 09:18"),
    ],
    ids=["marketable, never returns", "marketable, returns", "marketable, gaps lower", "resting", "resting, gaps through"],
)
def test_a_limit_marketable_when_sent_fills_at_the_decision_price(price, tail, entry, at, path):
    """A buy limit at or above the decision bar's close (103) trades there like a market order;
    a limit below it rests and fills at its price, or at the open if the bar gaps through it."""
    out = run(
        {1: lambda s, c: s.submit_order(O.Limit(side="buy", price=price, units=10.0))},
        {"X": bars(LIMIT_BASE + tail, T0)},
        lambda: BacktestConfig(initial_capital=1e5, fees=0.0),
        paths=(path,),
    )
    assert [(t.entry_price, fmt(t.entry_time)) for t in out[path].result.trades()] == [(entry, at)]


# ---------------------------------------------------------------------------
# M15: settings that change nothing are refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make", [BacktestConfig, rb.InstrumentConfig])
@pytest.mark.parametrize("setter", ["set_fixed_stop", "set_trailing_stop"])
@pytest.mark.parametrize("percent", [5.0, 0.0, -0.05])
def test_a_stop_percent_is_a_fraction(make, setter, percent):
    cfg = make(initial_capital=1.0) if make is BacktestConfig else make()
    with pytest.raises(InvalidParameter) as e:
        getattr(cfg, setter)(percent)
    assert e.value.code == "invalid_config" and e.value.parameter == "percent"
    getattr(cfg, setter)(0.05)


@pytest.mark.parametrize("setting", [{"order_latency_ns": 30 * 10**9}, {"partial_fills": True}])
def test_tick_only_settings_are_refused_on_bars(setting):
    data = {"X": bars(SIZE_ROWS, T0)}
    with pytest.raises(UnsupportedOperation) as e:
        run({}, data, lambda: BacktestConfig(initial_capital=1e4, fees=0.0, **setting), paths=("batch",))
    assert e.value.code == "unsupported_on_bars"
    node = LiveNode([Script({})], ["X"], BacktestConfig(initial_capital=1e4, fees=0.0, **setting), bar_stamp="close")
    with pytest.raises(UnsupportedOperation):
        node.push_bar("X", int(data["X"]["timestamps"][0]), 100.0, 101.0, 99.0, 100.0, 1.0)


# ---------------------------------------------------------------------------
# Carried follow-ups
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", PATHS)
def test_an_exit_decided_on_the_event_the_entry_fills_acts_on_the_next_open(path):
    """Entry decided at 09:17 fills at the 09:18 open (105); a close decided at 09:18, before
    the entry is booked on a venue, trades at the 09:19 open (104) in every mode."""
    out = run(
        {1: lambda s, c: s.enter(size_frac=0.5), 2: lambda s, c: s.close_position()},
        {"X": bars(SIZE_ROWS, T0)},
        lambda: BacktestConfig(initial_capital=10000.0, fees=0.0, fill_timing="next_bar_open"),
        paths=(path,),
    )
    assert [(t[1], t[2], t[4]) for t in trades(out[path])] == [(105.0, 104.0, "11-02 09:19")]


@pytest.mark.parametrize("path", PATHS)
def test_next_bar_open_excursions_include_the_entry_bar_in_every_mode(path):
    """Bought at the 09:18 open (100) on a bar of range 94..106: the 6-point swings both ways
    count toward the excursions (60 each on 10 units), whatever API placed the order."""
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (100, 106, 94, 101), (101, 102, 100, 101.5), (101.5,) * 4]
    for act in (
        lambda s, c: s.submit_order(O.Market(side="buy", units=10.0)),
        lambda s, c: s.enter(size_frac=0.1),
    ):
        out = run({1: act}, {"X": bars(rows, T0)},
                  lambda: BacktestConfig(initial_capital=10000.0, fees=0.0, fill_timing="next_bar_open"),
                  paths=(path,))
        (t,) = out[path].result.trades()
        assert (t.mfe_pnl, t.mae_pnl) == pytest.approx((60.0, -60.0)) or t.size != 10.0


def test_unfunded_entries_are_reported_apart_from_risk_refusals():
    """One lot costs 7500 and the account has 1000: the entry sizes to nothing. It is not a risk
    limit, so `rejected_entries` stays 0 and `unfunded_entries` counts it."""
    out = run({1: lambda s, c: s.enter(size_frac=1.0)}, {"F": bars(LOT_ROWS, T0)},
              lambda: BacktestConfig(initial_capital=1000.0, fees=0.0), paths=("batch",), instruments=future())
    res = out["batch"]
    assert (res.rejected_entries, res.unfunded_entries) == (0, 1)
    assert [s.unfunded_entries for s in res.per_instrument] == [1]


# ---------------------------------------------------------------------------
# A roll keeps the notional exposure
# ---------------------------------------------------------------------------


def _roll_run(policy, lot_new=1.0):
    from raptorbt.strategy import Future, Holding, Requirements, Underlying
    from raptorbt.strategy.continuous import RollEvents

    exp_a, exp_b = T0 + 20 * MIN, T0 + 400 * MIN
    specs = {
        "FA": rb.InstrumentSpec.futures_contract("FA", exp_a, 1.0, underlying="IDX"),
        "FB": rb.InstrumentSpec.futures_contract("FB", exp_b, lot_new, underlying="IDX"),
    }
    sized = []

    class Carry(RollEvents, Strategy):
        requirements = Requirements(
            roles={
                "index": Underlying(asset="index"),
                "fut": Future(of="index", contract="continuous", roll_days=5 / 1440),
            },
            timeframes=["1m"], holding=Holding.overnight(), data=["bar"],
        )
        auto_roll = True
        roll_sizing = policy

        def __init__(self):
            super().__init__()
            self.n = 0

        def on_bar(self, ctx, bar):
            if ctx.symbol != "IDX":
                return
            self.n += 1
            if self.n == 1:
                self.submit_order(O.Market(side="buy", units=3.0, instrument=ctx.role("fut")))

        def on_roll_sized(self, ctx, roll, sizing):
            sized.append(sizing)

    n = 30
    data = {}
    for k, s in enumerate(("IDX", "FA", "FB")):
        c = 100.0 + 10 * k + np.arange(n) * 0.1
        data[s] = {
            "timestamps": T0 + (np.arange(n, dtype=np.int64) + 1) * MIN,
            "open": c - 0.05, "high": c + 1, "low": c - 1, "close": c, "volume": np.full(n, 100.0),
        }
    r = run_strategies([Carry()], data, BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close",
                       instruments=specs, bindings={"index": "IDX"})[0]
    return r, sized


def test_a_roll_keeps_the_notional_by_default():
    """3 units of FA closed at 111.4 are 334.2 of exposure; FB (121.4, lot 1) buys
    floor(334.2 / 121.4) = 2, and the residual is what 2 leaves out."""
    r, sized = _roll_run("notional")
    assert [(t.symbol, t.size) for t in r.result.trades()] == [("FA", 3.0), ("FB", 2.0)]
    (s,) = sized
    assert (s["lots_old"], s["lots_new"]) == (3.0, 2.0)
    assert s["notional_old"] == pytest.approx(334.2)
    # The roll is decided on the bar of t, and every instrument's bar of t is visible to it (the
    # equal-timestamp rule): FB's close there is 121.4, so two lots are 242.8 and 91.4 is left
    # unexposed. FB listed before or after FA or IDX, the sizing reads the same bar.
    assert s["notional_new"] == pytest.approx(242.8)
    assert s["residual"] == pytest.approx(91.4)


def test_a_roll_can_keep_the_lot_count():
    r, sized = _roll_run("lots")
    assert [(t.symbol, t.size) for t in r.result.trades()] == [("FA", 3.0), ("FB", 3.0)]
    assert sized == []


def test_an_unknown_roll_sizing_is_refused():
    with pytest.raises(InvalidParameter) as e:
        _roll_run("exposure")
    assert e.value.code == "invalid_roll_sizing"
