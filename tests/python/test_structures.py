"""Option structures: resolution at the event time, group submission, leg failure, exits, and
the same result in a batch run and a live node."""

import types

import numpy as np
import pytest

import raptorbt
from raptorbt import (
    BacktestConfig,
    InstrumentSpec,
    LiveNode,
    MarginModel,
    OptionChain,
    Strategy,
    run_strategies,
)
from raptorbt import structures as st
from raptorbt.errors import InvalidParameter, StateError

from _chain_fixture import (
    DAY, EXP1, EXP2, LOT, N_BARS, STEP, STRIKES, T0, U,
    option_spec, specs, symbol, universe,
)

CHAIN = OptionChain(specs())
NOW = T0 + 5 * STEP


def code_of(fn, *a, **kw):
    with pytest.raises((InvalidParameter, StateError)) as e:
        fn(*a, **kw)
    return e.value.code


def resolve(structure, spot=100.2, lots=1, now=NOW, chain=CHAIN):
    return structure.resolve(chain, U, spot, now, lots)


def picks(r):
    return [(l.instrument, l.side, l.lots) for l in r.legs]


# -- resolution ---------------------------------------------------------------------------------


def test_iron_condor_picks_contracts_sides_and_units():
    r = resolve(st.iron_condor(1, 2), lots=3)
    assert picks(r) == [
        (symbol(95, "put", EXP1), "sell", 3),
        (symbol(105, "call", EXP1), "sell", 3),
        (symbol(110, "call", EXP1), "buy", 3),
        (symbol(90, "put", EXP1), "buy", 3),
    ]
    assert [l.quantity for l in r.legs] == [3 * LOT] * 4
    assert r.anchor_expiry_ns == EXP1 and r.name == "iron_condor"


def test_submission_order_is_protective_first():
    r = resolve(st.iron_condor(1, 2))
    sides = [r.legs[i].side for i in r.submission_order()]
    assert sides == ["buy", "buy", "sell", "sell"]


def test_ratio_and_back_spreads_scale_lots_per_leg():
    r = resolve(st.ratio_spread("call", 0, 1), lots=2)
    assert [(l.side, l.lots, l.quantity) for l in r.legs] == [("buy", 2, 100.0), ("sell", 4, 200.0)]
    r = resolve(st.ratio_spread("call", 0, 1, long_ratio=2, short_ratio=1))
    assert [(l.side, l.lots) for l in r.legs] == [("buy", 2), ("sell", 1)]
    r = resolve(st.butterfly("put", 1))
    assert [(l.side, l.lots) for l in r.legs] == [("buy", 1), ("sell", 2), ("buy", 1)]


def test_a_sell_straddle_and_a_buy_iron_condor_flip_every_leg():
    assert {l.side for l in resolve(st.straddle("sell")).legs} == {"sell"}
    longc = resolve(st.iron_condor(1, 2, side="buy"))
    assert [l.side for l in longc.legs] == ["buy", "buy", "sell", "sell"]


def test_named_vertical_spreads():
    assert picks(resolve(st.bull_call_spread(0, 1))) == [
        (symbol(100, "call", EXP1), "buy", 1), (symbol(105, "call", EXP1), "sell", 1)]
    assert picks(resolve(st.bear_put_spread(0, 1))) == [
        (symbol(100, "put", EXP1), "buy", 1), (symbol(95, "put", EXP1), "sell", 1)]
    assert picks(resolve(st.bull_put_spread(0, 1))) == [
        (symbol(95, "put", EXP1), "buy", 1), (symbol(100, "put", EXP1), "sell", 1)]
    assert picks(resolve(st.bear_call_spread(0, 1))) == [
        (symbol(105, "call", EXP1), "buy", 1), (symbol(100, "call", EXP1), "sell", 1)]
    assert code_of(st.bull_call_spread, 1, 1) == "invalid_structure"


def test_calendar_sells_near_and_buys_far_at_the_same_strike():
    r = resolve(st.calendar("call", 1))
    assert picks(r) == [(symbol(105, "call", EXP2), "buy", 1), (symbol(105, "call", EXP1), "sell", 1)]


def test_generic_structure_with_selectors():
    s = st.Structure(
        [
            st.Leg("call", st.Strike(110.0), st.Expiry.at(EXP2), "sell"),
            st.Leg("put", st.Delta(0.30, vol=0.2), st.Expiry.nth(0), "buy", ratio=2),
            st.Leg("call", st.Premium(2.0, lambda c: {105.0: 2.1, 110.0: 0.4}.get(c.strike)),
                   st.Expiry.nearest(), "buy"),
        ]
    )
    r = resolve(s)
    assert r.legs[0].instrument == symbol(110, "call", EXP2)
    assert r.legs[1].right == "put" and r.legs[1].strike < 100.2 and r.legs[1].lots == 2
    assert r.legs[2].instrument == symbol(105, "call", EXP1)
    # Strikes of one expiry are not those of another unless listed: the anchor is the earliest.
    assert r.anchor_expiry_ns == EXP1


def test_expiry_selectors():
    pick = lambda e: e.resolve(CHAIN, U, NOW)  # noqa: E731
    assert pick(st.Expiry.nearest()) == EXP1
    assert pick(st.Expiry.nearest(min_days=0)) == EXP1
    assert pick(st.Expiry.nth(1)) == EXP2
    assert pick(st.Expiry.at(EXP2)) == EXP2
    assert pick(st.Expiry.on(EXP1 + 3600 * 10**9)) == EXP1
    assert code_of(pick, st.Expiry.at(EXP2 + 1)) == "structure_contract_missing"
    assert code_of(pick, st.Expiry.nth(5)) == "structure_contract_missing"


def test_nothing_unlisted_is_visible_at_the_event_time():
    late = T0 + 30 * STEP
    new_strike = option_spec(105.0, "call", EXP1)
    listed = [s for s in specs() if s.symbol != new_strike.symbol]
    late_spec = InstrumentSpec.option(
        new_strike.symbol, 105.0, "call", EXP1, LOT, underlying=U, activation_ns=late
    )
    chain = OptionChain(listed + [late_spec])
    s = st.strangle(1)
    assert code_of(lambda: s.resolve(chain, U, 100.0, late - 1)) == "structure_contract_missing"
    assert s.resolve(chain, U, 100.0, late).legs[0].instrument == new_strike.symbol


def test_off_the_ladder_and_bad_arguments_are_refused():
    assert code_of(resolve, st.strangle(4)) == "structure_contract_missing"
    assert code_of(resolve, st.Structure([st.Leg("call", st.Strike(102.0))])) == "structure_contract_missing"
    assert code_of(resolve, st.straddle(), lots=0) == "invalid_structure"
    assert code_of(resolve, st.straddle(), spot=float("nan")) == "invalid_structure"
    assert code_of(st.Leg, "future") == "invalid_structure"
    assert code_of(st.Leg, "call", side="hold") == "invalid_structure"
    assert code_of(st.Leg, "call", ratio=0) == "invalid_structure"
    assert code_of(st.Structure, []) == "invalid_structure"
    assert code_of(st.iron_condor, 2, 2) == "invalid_structure"
    assert code_of(st.strangle, 0) == "invalid_structure"
    assert code_of(resolve, st.covered_call()) == "invalid_structure"  # no underlying_spec


def test_covered_call_sizes_shares_from_the_option_lot():
    chain = OptionChain([
        InstrumentSpec.option(f"EQ{int(k)}{r[0]}", k, r, EXP1, 50.0, underlying="EQ")
        for k in STRIKES for r in ("call", "put")
    ])
    eq = InstrumentSpec.equity("EQ", lot_size=1.0)
    r = st.covered_call(1).resolve(chain, "EQ", 100.0, NOW, 2, underlying_spec=eq)
    assert [(l.instrument, l.side, l.quantity) for l in r.legs] == [("EQ", "buy", 100.0), ("EQ105c", "sell", 100.0)]
    eq75 = InstrumentSpec.equity("EQ", lot_size=75.0)
    assert code_of(st.covered_call(1).resolve, chain, "EQ", 100.0, NOW, 1, underlying_spec=eq75) == "invalid_structure"


def test_net_premium_and_defined_risk_margin():
    r = resolve(st.iron_condor(1, 2))
    px = {l.instrument: p for l, p in zip(r.legs, (3.0, 2.5, 1.0, 1.5))}
    assert r.net_premium(px) == pytest.approx(-3.0 - 2.5 + 1.0 + 1.5)
    # Each wing is 5 wide: the worst loss is (5 - credit) per unit on the losing side.
    margin = r.margin(MarginModel.defined_risk(), px)
    # both short strikes sit one step inside wings 5 wide: the worst loss is one wing less the credit
    assert margin == pytest.approx((5.0 - 3.0) * LOT)
    assert code_of(r.margin, MarginModel.defined_risk(), {}) == "structure_missing_mark"


# -- submission, without an engine ----------------------------------------------------------------


def ctx_at(ts):
    return types.SimpleNamespace(bar_close_ts=ts, timestamp=ts)


def queued(s):
    return s.drain_commands()


def test_nothing_is_queued_when_a_pre_flight_check_fails():
    s = Strategy()
    ic = st.iron_condor(1, 2)
    kw = dict(chain=CHAIN, underlying=U, spot=100.2)
    r = ic.resolve(CHAIN, U, 100.2, NOW)
    px = {l.instrument: 2.0 for l in r.legs}
    with pytest.raises(InvalidParameter) as e:
        s.submit_structure(ctx_at(NOW), ic, prices=px, net_limit=-5.0, **kw)  # credit of 0 < 5 asked
    assert e.value.code == "structure_limit_breached"
    with pytest.raises(InvalidParameter) as e:
        s.submit_structure(ctx_at(NOW), ic, prices=px, margin_model=MarginModel.defined_risk(),
                           available=1.0, **kw)
    assert e.value.code == "structure_margin_exceeded"
    with pytest.raises(InvalidParameter) as e:
        s.submit_structure(ctx_at(NOW), ic, order_type="limit", **kw)  # no prices
    assert e.value.code == "invalid_structure"
    with pytest.raises(InvalidParameter) as e:
        s.submit_structure(ctx_at(NOW), ic, order_type="stop", **kw)
    assert e.value.code == "invalid_structure"
    with pytest.raises(InvalidParameter) as e:
        s.submit_structure(ctx_at(NOW), ic, on_leg_failure="ignore", **kw)
    assert e.value.code == "invalid_structure"
    assert queued(s) == []


def test_legs_are_queued_as_one_tagged_group_protective_first():
    s = Strategy()
    g = s.submit_structure(ctx_at(NOW), st.iron_condor(1, 2), chain=CHAIN, underlying=U, spot=100.2)
    commands = queued(s)
    # hedge-first: the two bought wings go out; the sold legs wait for the wings that cover them
    assert [c[0] for c in commands] == ["submit"] * 2
    bought = [i for i, ls in enumerate(g.legs) if ls.leg.side == "buy"]
    assert [c[1] for c in commands] == [f"{g.id}:{i}" for i in bought]
    assert g.status(s.cache) == "submitted"
    assert [ls.client_id for ls in g.legs] == [f"{g.id}:{i}" for i in range(4)]
    with pytest.raises(InvalidParameter) as e:
        s.submit_structure(ctx_at(NOW), st.straddle(), chain=CHAIN, underlying=U, spot=100.0, group_id=g.id)
    assert e.value.code == "invalid_structure"


def test_limit_legs_carry_the_given_prices():
    s = Strategy()
    ic = st.straddle()
    r = ic.resolve(CHAIN, U, 100.0, NOW)
    px = {l.instrument: 4.25 for l in r.legs}
    s.submit_structure(ctx_at(NOW), ic, chain=CHAIN, underlying=U, spot=100.0, order_type="limit", prices=px)
    assert len(queued(s)) == 2  # two sold limit legs, nothing covers them


# -- runs ---------------------------------------------------------------------------------------


class Base(Strategy):
    """Opens one structure on the underlying's third bar and manages it every bar."""

    structure = None
    exit = None
    on_leg_failure = "unwind"
    lots = 1

    def __init__(self):
        super().__init__()
        self.marks = {}
        self.group = None

    def on_bar(self, ctx):
        self.marks[ctx.symbol] = ctx.bar.close
        self.update_structures(ctx, self.marks)
        if ctx.symbol == U and self.group is None and ctx.idx == 3:
            self.group = self.submit_structure(
                ctx, self.structure, chain=CHAIN, underlying=U, spot=ctx.bar.close,
                lots=self.lots, exit=self.exit, on_leg_failure=self.on_leg_failure,
            )


def make(structure, **attrs):
    return type("S", (Base,), {"structure": structure, **attrs})


def run(strategy, drift=0.0, **config):
    data, inst = universe(drift=drift)
    cfg = BacktestConfig(initial_capital=2_000_000.0, fees=0.0, **config)
    result = run_strategies([strategy], data, cfg, instruments=inst, account_type="margin",
                            leverage=1.0, bar_stamp="close")[0]
    return result


def order_rows(result):
    return [(o.client_id, o.symbol, o.side, o.status, o.reject_reason, o.requested_qty, o.filled_qty,
             o.avg_fill_price, o.submitted_idx) for o in result.result.orders()]


def test_an_iron_condor_runs_as_four_linked_orders_and_stays_open():
    s = make(st.iron_condor(1, 2))()
    res = run(s)
    rows = order_rows(res)
    assert len(rows) == 4 and all(r[3] == "filled" for r in rows)
    assert {r[0].split(":")[0] for r in rows} == {s.group.id}
    assert s.structure_status(s.group) == "filled"
    # Sold legs carry the credit, bought legs the debit.
    assert {r[1]: r[2] for r in rows} == {
        symbol(95, "put", EXP1): "sell", symbol(105, "call", EXP1): "sell",
        symbol(110, "call", EXP1): "buy", symbol(90, "put", EXP1): "buy"}
    assert s.group.pnl(s.cache, s.marks) != 0.0


def test_a_straddle_with_a_time_exit_closes_every_leg():
    close_at = T0 + 20 * STEP
    s = make(st.straddle("sell"), exit=st.Exit(close_at_ns=close_at))()
    res = run(s)
    assert s.structure_status(s.group) == "closed" and s.group.reason == "close_at"
    trades = res.result.trades()
    assert len(trades) == 2 and all(t.exit_idx >= 19 for t in trades)
    closes = [r for r in order_rows(res) if ":x" in r[0]]
    assert len(closes) == 2 and all(r[3] == "filled" for r in closes)


SHORT_CALL = st.Structure([st.Leg("call", side="sell")], "short_call")
LONG_CALL = st.Structure([st.Leg("call", side="buy")], "long_call")


def test_stop_loss_on_net_pnl_closes_the_group():
    # A rally hurts a short call: the stop fires well before the end.
    s = make(SHORT_CALL, exit=st.Exit(stop_loss=150.0))()
    res = run(s, drift=0.4)
    assert s.group.reason == "stop_loss" and s.structure_status(s.group) == "closed"
    (trade,) = res.result.trades()
    assert trade.pnl < -150.0 and trade.exit_idx < N_BARS - 2
    # A stop that is never reached leaves the group open.
    s2 = make(SHORT_CALL, exit=st.Exit(take_profit=1e9, stop_loss=1e9))()
    run(s2, drift=0.4)
    assert s2.structure_status(s2.group) == "filled"


def test_take_profit_on_net_pnl():
    s = make(LONG_CALL, exit=st.Exit(take_profit=100.0))()
    run(s, drift=0.4)
    assert s.group.reason == "take_profit"


def test_premium_exit_uses_the_marks():
    s = make(LONG_CALL, exit=st.Exit(premium_at_or_above=12.0))()
    run(s, drift=0.4)
    assert s.group.reason == "premium_at_or_above"
    s = make(LONG_CALL, exit=st.Exit(premium_at_or_below=0.5))()
    run(s)
    assert s.structure_status(s.group) == "filled"


def test_days_before_expiry_exit():
    s = make(st.straddle(), exit=st.Exit(days_before_expiry=(EXP1 - (T0 + 25 * STEP)) / DAY))()
    run(s)
    assert s.group.reason == "days_before_expiry"


def test_pnl_exits_need_marks_and_never_skip_silently():
    class NoMarks(Base):
        structure = st.straddle()
        exit = st.Exit(stop_loss=10.0)

        def on_bar(self, ctx):
            self.marks[ctx.symbol] = ctx.bar.close
            if ctx.symbol == U and self.group is None and ctx.idx == 3:
                self.group = self.submit_structure(ctx, self.structure, chain=CHAIN, underlying=U,
                                                   spot=ctx.bar.close, exit=self.exit)
            elif self.group is not None:
                self.update_structures(ctx)

    with pytest.raises(StateError) as e:
        run(NoMarks())
    assert e.value.code == "structure_missing_mark"


def test_manual_close():
    class Manual(Base):
        structure = st.straddle()

        def on_bar(self, ctx):
            super().on_bar(ctx)
            if ctx.symbol == U and ctx.idx == 12:
                self.close_structure(self.group)

    s = Manual()
    run(s)
    assert s.group.reason == "manual" and s.structure_status(s.group) == "closed"


# -- leg failure ----------------------------------------------------------------------------------


def notional_cut():
    """An order-notional limit that admits the bought wings and refuses the sold body."""
    data, _ = universe()
    at = lambda strike, right: data[symbol(strike, right, EXP1)]["close"][3] * LOT  # noqa: E731
    lo = max(at(90, "put"), at(110, "call"))
    hi = min(at(95, "put"), at(105, "call"))
    assert lo < hi, (lo, hi)
    return (lo + hi) / 2.0


def test_a_rejected_sold_leg_unwinds_what_filled_and_cancels_the_rest():
    s = make(st.iron_condor(1, 2))()
    res = run(s, max_order_notional=notional_cut())
    rows = {r[0]: r for r in order_rows(res)}
    g = s.group
    states = [g.leg_status(s.cache, i) for i in range(4)]
    refused = [i for i, x in enumerate(states) if x == "rejected"]
    assert refused and all(rows[f"{g.id}:{i}"][4] == "order_size" for i in refused)
    assert all(g.legs[i].leg.side == "sell" for i in refused), "only a sold leg is over the limit"
    assert g.failure[1:] == ("rejected", "order_size")
    assert set(states) <= {"rejected", "canceled", "filled", "waiting"}, states
    # a sold leg is sent when its hedge has filled, so how many legs had filled when the refusal
    # was heard depends on bar order; whatever filled was closed, the rest was cancelled or never sent
    filled = [i for i, x in enumerate(states) if x == "filled"]
    assert filled, "the test needs at least one leg to have filled before the failure"
    assert g.reason == "leg_failure" and s.structure_status(g) == "closed"
    closes = {k: r for k, r in rows.items() if ":x" in k}
    assert sorted(k.split(":x")[1].split(".")[0] for k in closes) == sorted(str(i) for i in filled)
    assert all(r[3] == "filled" and r[2] == "sell" for r in closes.values())  # wings were bought
    trades = res.result.trades()
    assert len(trades) == len(filled)  # each filled leg round-tripped, nothing is left open
    assert not res.halted


def test_keep_leaves_the_filled_legs_and_reports_broken():
    s = make(st.iron_condor(1, 2), on_leg_failure="keep")()
    res = run(s, max_order_notional=notional_cut())
    assert s.structure_status(s.group) == "broken"
    assert not [r for r in order_rows(res) if ":x" in r[0]]


class UnderlyingOnly(Base):
    """Looks at its group only on the underlying's bars, as a strategy that returns early would."""

    structure = st.iron_condor(1, 2)

    def on_bar(self, ctx):
        self.marks[ctx.symbol] = ctx.bar.close
        if ctx.symbol != U:
            return
        self.update_structures(ctx, self.marks)
        if self.group is None and ctx.idx == 3:
            self.group = self.submit_structure(
                ctx, self.structure, chain=CHAIN, underlying=U, spot=ctx.bar.close)


def test_a_strategy_that_looks_at_its_group_only_on_one_instruments_bars_still_unwinds_at_once():
    """The strategy base class forwards order events to the groups: no mixin, no update call on
    the failing leg's bar."""
    plain = UnderlyingOnly()
    res = run(plain, max_order_notional=notional_cut())
    closes = [r for r in order_rows(res) if ":x" in r[0]]
    assert plain.structure_status(plain.group) == "closed"
    assert plain.group.reason == "leg_failure" and plain.group.failure is not None
    assert all(r[3] == "filled" for r in closes), closes


def test_an_entry_that_does_not_fill_in_time_is_unwound():
    class Limited(Base):
        structure = st.straddle()
        exit = st.Exit(entry_timeout_ns=4 * STEP)

        def on_bar(self, ctx):
            self.marks[ctx.symbol] = ctx.bar.close
            self.update_structures(ctx, self.marks)
            if ctx.symbol == U and self.group is None and ctx.idx == 3:
                r = self.structure.resolve(CHAIN, U, ctx.bar.close, ctx.bar_close_ts)
                px = {l.instrument: 4.0 for l in r.legs}
                px[r.legs[1].instrument] = 0.05  # the put never trades that low
                self.group = self.submit_structure(
                    ctx, self.structure, chain=CHAIN, underlying=U, spot=ctx.bar.close,
                    order_type="limit", prices=px, exit=self.exit)

    s = Limited()
    res = run(s)
    g = s.group
    assert g.reason == "entry_timeout" and s.structure_status(g) == "closed"
    assert g.leg_status(s.cache, 1) == "canceled"
    assert len(res.result.trades()) <= 1


# -- batch run and live node agree ----------------------------------------------------------


def digest(result):
    return {
        "trades": [(t.symbol, t.entry_idx, t.exit_idx, t.entry_price, t.exit_price, t.size, t.pnl)
                   for t in result.result.trades()],
        "orders": order_rows(result),
        "equity": np.asarray(result.result.equity_curve()).tolist(),
    }


@pytest.mark.parametrize(
    "structure,attrs,config",
    [
        (st.iron_condor(1, 2), {"exit": st.Exit(close_at_ns=T0 + 25 * STEP)}, {}),
        (SHORT_CALL, {"exit": st.Exit(stop_loss=150.0)}, {}),
        (st.iron_condor(1, 2), {}, {"max_order_notional": "cut"}),
    ],
    ids=["iron_condor_time_exit", "straddle_stop", "condor_leg_failure"],
)
def test_run_strategies_and_live_node_give_identical_results(structure, attrs, config):
    if config.get("max_order_notional") == "cut":
        config = {"max_order_notional": notional_cut()}
    cls = make(structure, **attrs)
    batch_strategy, live_strategy = cls(), cls()
    drift = 0.4 if "stop_loss" in str(attrs) else 0.0
    data, inst = universe(drift=drift)
    cfg = BacktestConfig(initial_capital=2_000_000.0, fees=0.0, **config)
    batch = run_strategies([batch_strategy], data, cfg, instruments=inst, account_type="margin",
                           leverage=1.0, bar_stamp="close")[0]
    symbols = list(data)
    node = LiveNode([live_strategy], symbols, cfg, instruments=inst, account_type="margin",
                    leverage=1.0, bar_stamp="close")
    for i in range(N_BARS):
        # the bars of one time are one group, as the batch run delivers them
        node.begin_group()
        for sym in symbols:
            b = data[sym]
            node.push_bar(sym, int(b["timestamps"][i]), float(b["open"][i]), float(b["high"][i]),
                          float(b["low"][i]), float(b["close"][i]), float(b["volume"][i]))
        node.flush()
    streamed = node.finish()[0]
    d1, d2 = digest(batch), digest(streamed)
    assert d1["orders"], "the structure placed nothing"
    assert d1 == d2
    assert batch_strategy.group.reason == live_strategy.group.reason
    assert batch_strategy.structure_status(batch_strategy.group) == live_strategy.structure_status(live_strategy.group)


def test_an_order_for_another_instrument_fills_on_its_next_bar():
    """Placed from one instrument's bar, a market order for another is stamped with that
    instrument's last stepped bar and fills on its next (it used to stay unacknowledged forever
    in bar runs)."""

    def flat(n=10):
        c = np.linspace(100.0, 110.0, n)
        return {"timestamps": np.arange(1, n + 1, dtype=np.int64) * STEP, "open": c, "high": c + 1,
                "low": c - 1, "close": c, "volume": np.full(n, 1000.0)}

    class Cross(Strategy):
        def on_bar(self, ctx):
            if ctx.symbol == "AAA" and ctx.idx == 3:
                self.submit_order(raptorbt.strategy.orders.Market(side="buy", instrument="BBB", quantity=2.0))

    res = run_strategies([Cross()], {"AAA": flat(), "BBB": flat()},
                         BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close")[0]
    (order,) = res.result.orders()
    assert (order.symbol, order.status, order.filled_qty) == ("BBB", "filled", 2.0)


# -- group funding at entry -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "capital"),
    [(MarginModel.defined_risk(), 4_000.0), (MarginModel.spread_credit(15, 5), 1_000.0)],
    ids=["defined_risk", "spread_credit"],
)
def test_an_iron_condor_enters_on_capital_that_only_suffices_with_the_group_credit(model, capital):
    """A sold call is filled before its bought wing in instrument order, so it is funded by the
    hedge already in its way: the hedge is pending, not yet held, and the sold leg waits for it."""
    s = make(st.iron_condor(1, 2))()
    data, inst = universe()
    cfg = BacktestConfig(initial_capital=capital, fees=0.0, margin_model=model)
    res = run_strategies([s], data, cfg, instruments=inst, account_type="margin", leverage=1.0,
                         bar_stamp="close")[0]
    rows = order_rows(res)
    assert [r[3] for r in rows] == ["filled"] * 4, rows
    assert s.structure_status(s.group) == "filled"
