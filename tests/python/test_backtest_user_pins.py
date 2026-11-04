"""Pins for user-visible backtest behaviour. Each test states what a user's backtest should do. Bars are built
in full below."""

import datetime as dt

import numpy as np
import pytest

from _market_harness import MIN, PaperExecution, Script, bars, ist
from raptorbt.strategy import orders as O

import raptorbt as rb  # noqa: E402
from raptorbt import BacktestConfig, LiveNode, MarginModel, Strategy, run_strategies  # noqa: E402

OHLCV = ("open", "high", "low", "close", "volume")


def _node_run(strategy, data, cfg, paper, **kw):
    syms = list(data)
    node = LiveNode([strategy], syms, cfg, bar_stamp="close", execution=PaperExecution() if paper else None, **kw)
    for i in range(len(data[syms[0]]["timestamps"])):
        node.push_group(int(data[syms[0]]["timestamps"][i]), {s: tuple(float(data[s][c][i]) for c in OHLCV) for s in syms})
    return node.finish()[0]


# an after_fill tag on a user's own order filled it at the open of the bar it decided on
@pytest.mark.parametrize("mode", ["batch", "node", "paper"])
def test_a_user_order_tagged_after_fill_is_refused_not_filled_at_the_open_of_the_decided_bar(mode):
    t0 = ist(2026, 11, 2, 9, 15)
    # 09:18 bar: open 100, close 110. The decision reads it (bar_stamp close) and buys.
    data = {"X": bars([(100, 100, 100, 100), (100, 101, 99, 100), (100, 111, 99, 110), (110, 111, 109, 110)], t0)}

    class A(Strategy):
        def __init__(self):
            super().__init__()
            self.n, self.f = 0, []

        def on_bar(self, ctx, bar):
            self.n += 1
            if self.n == 3 and ctx.bar.close > ctx.bar.open:
                self.submit_order(O.Market(side="buy", units=10.0, tags={"after_fill": "1"}))

        def on_order_event(self, ctx, e):
            if e.kind == "order_filled":
                self.f.append((ctx.event_ns, e.price))

    s = A()
    cfg = BacktestConfig(initial_capital=1e5, fees=0.0, fill_timing="next_bar_open")
    with pytest.raises(rb.errors.InvalidOrder) as e:
        if mode == "batch":
            run_strategies([s], data, cfg, bar_stamp="close")
        else:
            _node_run(s, data, cfg, mode == "paper")
    assert e.value.code == "invalid_order_tags"
    assert s.f == []


# the same through a fill hook that reads another instrument's closed bar
@pytest.mark.parametrize("mode", ["batch", "node", "paper"])
def test_a_tagged_answer_from_a_fill_hook_is_refused(mode):
    t0 = ist(2026, 11, 2, 9, 15)
    data = {"X": bars([(100,) * 4] * 4, t0),
            "Y": bars([(100, 100, 100, 100), (100, 101, 99, 100), (100, 111, 99, 110), (110, 111, 109, 110)], t0)}

    class B(Strategy):
        def __init__(self):
            super().__init__()
            self.n, self.f = 0, []

        def on_bar(self, ctx, bar):
            if ctx.symbol == "X":
                self.n += 1
                if self.n == 2:
                    self.submit_order(O.Market(side="buy", units=1.0))

        def on_order_event(self, ctx, e):
            if e.kind == "order_filled" and e.client_order_id == "O-0":
                # the hook no longer sees Y's bar of this instant, so the condition on its
                # close is gone; the tag is refused whatever the hook read.
                self.submit_order(O.Market(side="buy", units=10.0, instrument="Y", tags={"after_fill": "1"}))
            if e.kind == "order_filled":
                self.f.append((e.client_order_id, e.price))

    s = B()
    cfg = BacktestConfig(initial_capital=1e5, fees=0.0, fill_timing="next_bar_open")
    with pytest.raises(rb.errors.InvalidOrder) as e:
        if mode == "batch":
            run_strategies([s], data, cfg, bar_stamp="close")
        else:
            _node_run(s, data, cfg, mode == "paper")
    assert e.value.code == "invalid_order_tags"


def _limit_then_answer(answer_with_mark, timing, mode, rows):
    """A limit buy at 95 fills inside the third bar (open 100, low 94); the fill hook sends a market sell."""
    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.n, self.fills = 0, []

        def on_bar(self, ctx):
            self.n += 1
            if self.n == 2:
                self.submit_order(O.Limit(side="buy", price=95.0, units=10.0), client_id="lim")

        def on_order_event(self, ctx, e):
            if e.kind == "order_filled":
                self.fills.append((e.client_order_id, e.price, ctx.event_ns))
                if e.client_order_id == "lim":
                    out = O.Market(side="sell", units=10.0)
                    self.submit_order(out, client_id="out")

    t0 = ist(2026, 11, 2, 9, 15)
    data = {"X": bars(rows, t0)}
    cfg = BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing=timing)
    s = S()
    if mode == "batch":
        run_strategies([s], data, cfg, bar_stamp="close")
    else:
        node = LiveNode([s], ["X"], cfg, bar_stamp="close", execution=PaperExecution() if mode == "paper" else None)
        for i, (o, h, l, c) in enumerate(rows):
            node.push_bar("X", int(data["X"]["timestamps"][i]), o, h, l, c, 1e6)
        node.finish()
    return {f[0]: f[1] for f in s.fills}


# an answer to a fill inside the bar is never priced at that bar's open
@pytest.mark.parametrize("mode", ["batch", "node", "paper"])
def test_a_follow_up_to_a_fill_inside_the_bar_does_not_fill_at_that_bars_open(mode):
    rows = [(100, 100, 100, 100), (100, 100, 100, 100), (100, 101, 94, 96), (96, 97, 95, 96), (96, 97, 95, 96)]
    fills = _limit_then_answer(True, "next_bar_open", mode, rows)
    assert fills["lim"] == 95.0
    # the sell meets the next bar's open (96), never the open (100) of the bar the limit filled in
    assert fills["out"] == 96.0, fills


# control: an answer to a fill at the open does meet that bar's open (a roll's open, a hedge's sold leg)
@pytest.mark.parametrize("mode", ["batch", "node", "paper"])
def test_a_follow_up_to_a_fill_at_the_open_meets_that_bars_open(mode):
    class S(Strategy):
        def __init__(self):
            super().__init__()
            self.n, self.fills = 0, []

        def on_bar(self, ctx):
            self.n += 1
            if self.n == 2:
                self.submit_order(O.Market(side="buy", units=10.0), client_id="in")

        def on_order_event(self, ctx, e):
            if e.kind == "order_filled":
                self.fills.append((e.client_order_id, e.price))
                if e.client_order_id == "in":
                    self.submit_order(O.Market(side="sell", units=10.0), client_id="out")

    rows = [(100, 100, 100, 100), (100, 100, 100, 100), (101, 111, 99, 110), (110, 111, 109, 110)]
    t0 = ist(2026, 11, 2, 9, 15)
    data = {"X": bars(rows, t0)}
    s = S()
    cfg = BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing="next_bar_open")
    if mode == "batch":
        run_strategies([s], data, cfg, bar_stamp="close")
    else:
        node = LiveNode([s], ["X"], cfg, bar_stamp="close", execution=PaperExecution() if mode == "paper" else None)
        for i, (o, h, l, c) in enumerate(rows):
            node.push_bar("X", int(data["X"]["timestamps"][i]), o, h, l, c, 1e6)
        node.finish()
    assert dict(s.fills) == {"in": 101.0, "out": 101.0}, s.fills


# M1: an answer that keeps answering halts the strategy with a reason; the node keeps running
def test_a_follow_up_loop_halts_the_strategy_and_books_no_more_fills():
    class P(Strategy):
        def __init__(self):
            super().__init__()
            self.bars, self.fills = 0, 0

        def on_bar(self, ctx):
            self.bars += 1
            if self.bars == 2:
                self.submit_order(O.Market(side="buy", units=1.0), client_id="seed")

        def on_order_event(self, ctx, e):
            if e.kind == "order_filled":
                self.fills += 1
                side = "sell" if self.fills % 2 else "buy"
                self.submit_order(O.Market(side=side, units=1.0), client_id=f"pp{self.fills}")

    s = P()
    node = LiveNode([s], ["A"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close")
    for i in range(1, 7):
        node.push_bar("A", i * MIN, 100 + i, 100 + i, 100 + i, 100 + i, 1000)  # never raises
        if i == 3:
            fills_at_halt = s.fills
    assert fills_at_halt == 9 and s.fills == 9  # the seed and eight answers; the ninth was not booked
    res = node.finish()[0]
    assert res.result.halted
    assert not [o for o in res.result.orders() if o.status in ("submitted", "working", "accepted")]


# the engine's own roll open was sized on the close of the bar whose open it fills at
@pytest.mark.parametrize("timing,lots,price,at", [("next_bar_open", 3.0, 320.0, 15), ("same_bar_close", 2.0, 340.0, 14)])
def test_a_roll_is_sized_on_a_price_known_when_its_open_fills(timing, lots, price, at):
    from raptorbt.strategy import Future, Holding, Requirements, Underlying
    from raptorbt.strategy.continuous import RollEvents

    t0 = ist(2026, 11, 2, 9, 15)
    specs = {"FA": rb.InstrumentSpec.futures_contract("FA", t0 + 20 * MIN, 1.0, underlying="IDX"),
             "FB": rb.InstrumentSpec.futures_contract("FB", t0 + 400 * MIN, 1.0, underlying="IDX")}

    class Carry(RollEvents, Strategy):
        requirements = Requirements(roles={"index": Underlying(asset="index"),
                                           "fut": Future(of="index", contract="continuous", roll_days=5 / 1440)},
                                    timeframes=["1m"], holding=Holding.overnight(), data=["bar"])
        auto_roll = True

        def __init__(self):
            super().__init__()
            self.n, self.fills, self.sized = 0, [], []

        def on_bar(self, ctx, bar):
            if ctx.symbol == "IDX":
                self.n += 1
                if self.n == 1:
                    self.submit_order(O.Market(side="buy", units=3.0, instrument=ctx.role("fut")))

        def on_roll_sized(self, ctx, roll, sizing):
            self.sized.append(sizing)

        def on_order_event(self, ctx, e):
            if e.kind == "order_filled":
                self.fills.append((e.client_order_id, e.price, e.size))

    n = 30
    ts = t0 + (np.arange(n, dtype=np.int64) + 1) * MIN
    flat = lambda v: {"timestamps": ts, "open": np.full(n, v), "high": np.full(n, v + 15), "low": np.full(n, v - 15),
                      "close": np.full(n, v), "volume": np.full(n, 100.0)}
    data = {"IDX": flat(100.0), "FA": flat(330.0), "FB": flat(330.0)}
    data["FB"]["open"][at] = 320.0  # the bar the roll's close fills in: open 320, close 340
    data["FB"]["close"][at] = 340.0
    s = Carry()
    run_strategies([s], data, BacktestConfig(initial_capital=1e7, fees=0.0, fill_timing=timing),
                   bar_stamp="close", instruments=specs, bindings={"index": "IDX"})
    (opened,) = [f for f in s.fills if f[0].endswith(":o0")]
    # 3 x 330 = 990 closed. next_bar_open: the open fills at 320 and FB's last close before it is 330: 3 lots.
    # same_bar_close: the open fills at the close, 340: 2 lots.
    assert opened == ("fut:roll0:o0", price, lots), opened
    # the exposure is reported at the price the open filled at
    (sizing,) = s.sized
    assert sizing["price_new"] == price and sizing["notional_new"] == lots * price
    assert sizing["residual"] == pytest.approx(990.0 - lots * price)


# closing a hedged sold call and its hedge in one decision depends on symbol names
@pytest.mark.parametrize("mode", ["batch", "paper"])
@pytest.mark.parametrize("sold_k,hedge_k", [(100, 105), (95, 100)])
def test_closing_a_call_spread_in_one_decision_closes_both_legs_whatever_the_symbols(mode, sold_k, hedge_k):
    from _chain_fixture import EXP1, U, symbol, universe

    data, inst = universe()
    sold, hedge = symbol(sold_k, "call", EXP1), symbol(hedge_k, "call", EXP1)

    class C(Strategy):
        def on_bar(self, ctx):
            if ctx.symbol != U:
                return
            if ctx.idx == 3:
                self.submit_order(O.Market(side="buy", units=50.0, instrument=hedge), client_id="hedge")
            if ctx.idx == 4:
                self.submit_order(O.Market(side="sell", units=50.0, instrument=sold), client_id="sold")
            if ctx.idx == 8:
                self.submit_order(O.Market(side="buy", units=50.0, instrument=sold, reduce_only=True), client_id="close_sold")
                self.submit_order(O.Market(side="sell", units=50.0, instrument=hedge, reduce_only=True), client_id="close_hedge")

    cfg = BacktestConfig(initial_capital=1000.0, fees=0.0, margin_model=MarginModel.defined_risk())
    kw = dict(instruments=inst, account_type="margin", leverage=1.0)
    r = run_strategies([C()], data, cfg, bar_stamp="close", **kw)[0] if mode == "batch" else _node_run(C(), data, cfg, True, **kw)
    rows = {o.client_id: o.status for o in r.result.orders()}
    assert rows["close_sold"] == rows["close_hedge"] == "filled", rows


# close_structure on a bear call spread whose hedge sorts first ends stuck, hedge held
@pytest.mark.parametrize("spot", [100.0, 95.0])
def test_close_structure_closes_every_leg_whatever_the_strike_digits(spot):
    from _chain_fixture import U, specs, universe
    from raptorbt import OptionChain, structures as st

    chain = OptionChain(specs())

    class OC(Strategy):
        def __init__(self):
            super().__init__()
            self.group = None

        def on_bar(self, ctx):
            if ctx.symbol != U:
                return
            if self.group is None and ctx.idx == 3:
                self.group = self.submit_structure(ctx, st.bear_call_spread(short_steps=0, long_steps=1),
                                                   chain=chain, underlying=U, spot=spot)
            if ctx.idx == 8:
                self.close_structure(self.group)

    data, inst = universe()
    s = OC()
    r = run_strategies([s], data, BacktestConfig(initial_capital=1000.0, fees=0.0, margin_model=MarginModel.defined_risk()),
                       instruments=inst, account_type="margin", leverage=1.0, bar_stamp="close")[0]
    assert s.structure_status(s.group) == "closed"
    assert not [o for o in r.result.orders() if o.reject_reason == "insufficient_margin"]


# one special-session Saturday bar flips daily annualisation from 252 to 365
def test_daily_annualisation_does_not_flip_on_a_special_saturday_session():
    out = {}
    for include_sat in (False, True):
        days, d = [], dt.date(2025, 1, 1)
        while d <= dt.date(2025, 3, 31):
            if d.weekday() < 5 or (include_sat and d == dt.date(2025, 2, 1)):  # Budget-day session
                days.append(d)
            d += dt.timedelta(days=1)
        ts = [ist(x.year, x.month, x.day, 15, 30) for x in days]
        px = np.round(100 * np.cumprod(1 + np.random.default_rng(7).normal(0.001, 0.01, len(ts))), 2)
        data = {"X": bars([(p,) * 4 for p in px], ts)}
        r = run_strategies([Script({0: lambda s, c: s.submit_order(O.Market(side="buy", units=50.0))})], data,
                           BacktestConfig(initial_capital=1e4, fees=0.0), bar_stamp="close",
                           instruments={"X": rb.InstrumentSpec.equity("X", venue="NSE")})[0]
        ret = np.asarray(r.result.returns(), float)
        ret = ret[np.isfinite(ret)]
        out[include_sat] = (r.result.metrics.sharpe_ratio / (ret.mean() / ret.std(ddof=1))) ** 2
    assert out[True] == pytest.approx(252.0, rel=0.01), out


# a short daily series with one special Saturday session annualises on trading days too
def test_a_short_daily_series_with_one_special_session_annualises_on_252():
    days, d = [], dt.date(2025, 1, 27)
    while len(days) < 11:
        if d.weekday() < 5 or d == dt.date(2025, 2, 1):  # Budget-day Saturday
            days.append(d)
        d += dt.timedelta(days=1)
    assert any(x.weekday() == 5 for x in days)
    ts = [ist(x.year, x.month, x.day, 15, 30) for x in days]
    px = np.round(100 * np.cumprod(1 + np.random.default_rng(7).normal(0.001, 0.01, len(ts))), 2)
    data = {"X": bars([(p,) * 4 for p in px], ts)}
    r = run_strategies([Script({0: lambda s, c: s.submit_order(O.Market(side="buy", units=50.0))})], data,
                       BacktestConfig(initial_capital=1e4, fees=0.0), bar_stamp="close",
                       instruments={"X": rb.InstrumentSpec.equity("X", venue="NSE")})[0]
    ret = np.asarray(r.result.returns(), float)
    ret = ret[np.isfinite(ret)]
    assert (r.result.metrics.sharpe_ratio / (ret.mean() / ret.std(ddof=1))) ** 2 == pytest.approx(252.0, rel=0.01)


# a custom (non-Indicator) object for another symbol is a bar stale or not by symbol name
@pytest.mark.parametrize("other", ["FA", "ZFA"])
def test_a_custom_indicator_of_another_symbol_is_current_at_the_decision(other):
    t0 = ist(2026, 11, 2, 9, 15)
    data = {"IDX": bars([(1,) * 4, (2,) * 4, (3,) * 4], t0), other: bars([(10,) * 4, (20,) * 4, (30,) * 4], t0)}

    class Last:
        def __init__(self):
            self.value, self.initialized = None, False

        def update_bar(self, o, h, l, c):
            self.value, self.initialized = c, True

    class L(Strategy):
        def __init__(self):
            super().__init__()
            self.r = []

        def on_start(self, ctx):
            self.c = self.register_indicator(Last(), symbol=other)

        def on_bar(self, ctx, bar):
            if ctx.symbol == "IDX":
                self.r.append(self.c.value)

    s = L()
    run_strategies([s], data, BacktestConfig(initial_capital=1e5), bar_stamp="close")
    assert s.r == [10.0, 20.0, 30.0], s.r


# Decided and pinned: a CNC buy is charged provisionally as delivery and restated to intraday on a same-day exit
def test_a_same_day_round_trip_carries_the_provisional_delivery_charge_until_its_exit():
    t0 = ist(2026, 11, 2, 9, 15)
    data = {"X": bars([(1000,) * 4, (1000,) * 4, (1000,) * 4, (1010,) * 4, (1010,) * 4], t0)}
    sc = {1: lambda s, c: s.submit_order(O.Market(side="buy", units=100.0)),
          3: lambda s, c: s.submit_order(O.Market(side="sell", units=100.0))}
    r = run_strategies([Script(sc)], data, BacktestConfig(initial_capital=1e6, fee_schedule=rb.FeeSchedule.exchange_presets()),
                       bar_stamp="close", instruments={"X": rb.InstrumentSpec.equity("X", venue="NSE")})[0]
    eq = [round(float(x), 2) for x in r.result.equity_curve()]
    # The entry is charged as delivery, as a broker's CNC buy is (STT 100 + stamp 15 on 100,000 plus the
    # exchange charges): equity shows 118.74 of charges between entry and exit, drawdown, day-loss limits and
    # capital-edge sizing see it. The exit restates the round trip to the intraday row (entry 6.74) and
    # refunds the difference, as the contract note does.
    assert eq[1] == pytest.approx(1e6 - 118.74, abs=0.01), eq
    (trade,) = r.result.trades()
    assert trade.entry_fees == pytest.approx(6.74, abs=0.01)
    assert eq[3] == pytest.approx(1e6 + 1000.0 - trade.fees, abs=0.01)


# restating a same-day round trip does not charge a per-order brokerage again on each slice
def test_restating_same_day_slices_keeps_the_brokerage_as_booked():
    import copy

    from raptorbt.strategy import Holding, Requirements
    from test_fee_schedule import CARD

    card = copy.deepcopy(CARD)
    for row in card["rates"][:2]:
        row["brokerage"] = {"per_order": "20"}
    for row in card["rates"]:
        row["transaction_tax"] = {"kind": "NONE"}
        for k in ("exchange_turnover_percent", "sebi_percent", "stamp_duty_percent"):
            row[k] = "0"
        row["gst_percent"] = "0"
    card["rates"][1].pop("dp_charge")
    fs = rb.RateCard.from_dict(card).fee_schedule()

    class Overnight(Script):
        requirements = Requirements(holding=Holding.overnight())

    spec = rb.InstrumentSpec.equity("X", venue="NSE", session=rb.SessionSchedule.nse_equity())

    def total_fees(script):
        data = {"X": bars([(1000,) * 4] * 8, ist(2026, 11, 2, 9, 15))}
        r = run_strategies([Overnight(script)], data, BacktestConfig(initial_capital=1e6, fee_schedule=fs),
                           bar_stamp="close", instruments={"X": spec})[0]
        return sum(t.fees for t in r.result.trades()), r

    buy = {1: lambda s, c: s.submit_order(O.Market(side="buy", units=99.0))}
    one, _ = total_fees({**buy, 3: lambda s, c: s.submit_order(O.Market(side="sell", units=99.0))})
    three, r = total_fees({**buy, **{i: (lambda s, c: s.submit_order(O.Market(side="sell", units=33.0))) for i in (3, 4, 5)}})
    # a broker charges 20 on the entry order and 20 on each exit order
    assert one == pytest.approx(40.0, abs=0.01)
    assert three == pytest.approx(80.0, abs=0.01)
    assert float(list(r.result.equity_curve())[-1]) == pytest.approx(1e6 - 80.0, abs=0.01)


# the venue-implied MCX session is the contract's class, however the spec is built
@pytest.mark.parametrize("symbol,close", [("COTTON", 21 * 60), ("COTWASOIL", 21 * 60), ("KAPAS", 21 * 60),
                                          ("CPO", 17 * 60), ("RBDPMOLEIN", 17 * 60),
                                          ("MENTHAOIL", 17 * 60), ("CRUDEOIL", 23 * 60 + 55)])
def test_an_mcx_contract_takes_the_session_of_its_class_in_every_constructor(symbol, close):
    import json

    expiry = ist(2026, 11, 30, 21, 0)
    built = rb.InstrumentSpec.futures_contract(symbol, expiry, lot_size=1.0, venue="MCX")
    assert built.session.windows_on(20261102)[-1][1] == close
    # by underlying, for a listed contract of that name
    by_underlying = rb.InstrumentSpec.futures_contract(symbol + "26DECFUT", expiry, 1.0, underlying=symbol, venue="MCX")
    assert by_underlying.session.windows_on(20261102)[-1][1] == close
    # the venue set afterwards
    late = rb.InstrumentSpec.futures_contract(symbol, expiry, lot_size=1.0)
    late.venue = "MCX"
    assert late.session.windows_on(20261102)[-1][1] == close
    # from_json with a venue and no session applies the same rule
    j = json.loads(built.to_json())
    j["session"] = None
    rebuilt = rb.InstrumentSpec.from_json(json.dumps(j))
    assert rebuilt.session is not None and rebuilt.session.windows_on(20261102)[-1][1] == close


# a contract of no known class is not given a guessed session
@pytest.mark.parametrize("symbol", ["X", "F", "SOMENEWAGRI"])
def test_an_mcx_contract_of_no_known_class_gets_no_implied_session(symbol):
    expiry = ist(2026, 11, 30, 21, 0)
    with pytest.raises(Exception, match="COTWASOIL") as e:
        rb.InstrumentSpec.futures_contract(symbol, expiry, lot_size=1.0, venue="MCX")
    assert getattr(e.value, "code", None) == "mcx_session_unknown" and "09:00-17:00" in str(e.value)
    late = rb.InstrumentSpec.futures_contract(symbol, expiry, lot_size=1.0)
    with pytest.raises(Exception, match="mcx_session_unknown|COTWASOIL"):
        late.venue = "MCX"
    # naming the session is the way out
    ok = rb.InstrumentSpec.futures_contract(symbol, expiry, lot_size=1.0, venue="MCX", session=rb.SessionSchedule.mcx())
    assert ok.session is not None


def test_a_cotton_future_on_mcx_is_closed_at_2130():
    mcx = rb.InstrumentSpec.futures_contract("COTTON", ist(2026, 11, 30, 21, 0), lot_size=25.0, venue="MCX")
    ts = [ist(2026, 11, 2, 20, 58), ist(2026, 11, 2, 20, 59), ist(2026, 11, 2, 21, 30), ist(2026, 11, 2, 21, 31)]
    s = Script({2: lambda s, c: s.submit_order(O.Market(side="buy", units=25.0))})
    r = run_strategies([s], {"COTTON": bars([(100,) * 4] * 4, ts)}, BacktestConfig(initial_capital=1e6, fees=0.0),
                       bar_stamp="close", instruments={"COTTON": mcx})[0]
    assert r.result.trades() == [] and [x[1] for x in s.rejections] == ["market_closed"]


# taking a hedge off a sold option, whatever the cause, never aborts the run
def _hedged_run(mode, variant):
    from _chain_fixture import EXP1, N_BARS, U, symbol, universe
    from test_structure_hedge_first import config, push_bar

    hedge, sold = symbol(105, "call", EXP1), symbol(100, "call", EXP1)

    class Closes(Strategy):
        def on_bar(self, ctx):
            if ctx.symbol != U:
                return
            if ctx.idx == 3:
                # A level a rising-to-falling market reaches on bar 5, after the sold leg is held (sent
                # on bar 4, filled at its close). A level the first bar examines (bar 4) trades while
                # the bar forms, before the decision that sells the call, and finds no sold leg.
                kw = dict(stop_price=5.7, target_price=9.0) if variant == "stop" else {}
                self.submit_order(O.Market(side="buy", units=50.0, instrument=hedge, **kw), client_id="hedge")
            if ctx.idx == 4:
                self.submit_order(O.Market(side="sell", units=50.0, instrument=sold), client_id="sold")
            if ctx.idx == 6 and variant == "close_position":
                self.close_position(symbol=hedge)

    data, inst = universe()
    kw = dict(instruments=inst, account_type="margin", leverage=1.0, bar_stamp="close")
    if mode == "batch":
        return run_strategies([Closes()], data, config(300.0), **kw)[0]
    node = LiveNode([Closes()], list(data), config(300.0), execution=PaperExecution() if mode == "paper" else None, **kw)
    for i in range(N_BARS):
        push_bar(node, data, i)
    return node.finish()[0]


def _exits(r):
    return [(t.symbol[-5:], str(t.exit_reason), t.exit_idx) for t in r.result.trades()]


@pytest.mark.parametrize("mode", ["batch", "node", "paper"])
def test_a_forced_exit_of_a_hedge_closes_the_sold_leg_first(mode):
    exits = _exits(_hedged_run(mode, "stop"))
    # the stop on the hedge fires: the engine closes the sold call with an order, then the hedge
    # with another (the same orders in every mode, not a local booking of the sold leg)
    assert exits == [("100CE", "Order", 5), ("105CE", "Order", 5)], exits


@pytest.mark.parametrize("mode", ["batch", "node", "paper"])
def test_closing_a_hedge_alone_is_refused_and_the_run_goes_on(mode):
    r = _hedged_run(mode, "close_position")
    # both legs ride to the end of the data: the close of the hedge at bar 6 was refused
    assert {e[2] for e in _exits(r)} == {39}, _exits(r)


# Calendars and diagonals (margined across expiries): both legs close whichever month sorts first
@pytest.mark.parametrize("mode", ["batch", "paper"])
@pytest.mark.parametrize("sold,hedge", [(("A", 100), ("B", 100)), (("B", 100), ("A", 100)),
                                        (("A", 100), ("B", 105)), (("B", 105), ("A", 100))])
def test_closing_a_calendar_or_diagonal_in_one_decision_closes_both_legs(mode, sold, hedge):
    from _chain_fixture import EXP1, EXP2, U, symbol, universe

    exp = {"A": EXP1, "B": EXP2}
    sold_s, hedge_s = symbol(sold[1], "call", exp[sold[0]]), symbol(hedge[1], "call", exp[hedge[0]])
    data, inst = universe()

    class C(Strategy):
        def on_bar(self, ctx):
            if ctx.symbol != U:
                return
            if ctx.idx == 3:
                self.submit_order(O.Market(side="buy", units=50.0, instrument=hedge_s), client_id="hedge")
            if ctx.idx == 4:
                self.submit_order(O.Market(side="sell", units=50.0, instrument=sold_s), client_id="sold")
            if ctx.idx == 8:
                self.submit_order(O.Market(side="buy", units=50.0, instrument=sold_s, reduce_only=True), client_id="close_sold")
                self.submit_order(O.Market(side="sell", units=50.0, instrument=hedge_s, reduce_only=True), client_id="close_hedge")

    cfg = BacktestConfig(initial_capital=3000.0, fees=0.0, margin_model=MarginModel.spread_credit(0.15, 0.05))
    kw = dict(instruments=inst, account_type="margin", leverage=1.0)
    r = run_strategies([C()], data, cfg, bar_stamp="close", **kw)[0] if mode == "batch" else _node_run(C(), data, cfg, True, **kw)
    rows = {o.client_id: o.status for o in r.result.orders()}
    assert rows == {"hedge": "filled", "sold": "filled", "close_sold": "filled", "close_hedge": "filled"}, rows


# a supervisor reading health() sees which strategy a follow-up loop halted
def test_health_names_the_strategy_a_follow_up_loop_halted():
    class P(Strategy):
        def __init__(self):
            super().__init__()
            self.bars, self.fills = 0, 0

        def on_bar(self, ctx):
            self.bars += 1
            if self.bars == 2:
                self.submit_order(O.Market(side="buy", units=1.0), client_id="seed")

        def on_order_event(self, ctx, e):
            if e.kind == "order_filled":
                self.fills += 1
                self.submit_order(O.Market(side="sell" if self.fills % 2 else "buy", units=1.0),
                                  client_id=f"pp{self.fills}")

    node = LiveNode([P()], ["A"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close")
    assert node.health()["halted"] == []
    for i in range(1, 7):
        node.push_bar("A", i * MIN, 100 + i, 100 + i, 100 + i, 100 + i, 1000)
    assert node.health()["halted"] == [0]
