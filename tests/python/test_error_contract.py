"""The errors contract: every registered code is raised through the public API.

One case per code (a code may have several). Each case calls a public entry
point with an input that breaks exactly one rule, and the matrix asserts what
an agent or a program branches on: the ``code``, the registered class, the
``operation``, the ``parameter`` the case names, the ``skill`` and the
alternatives (``valid``), and that ``str(error)`` carries the documented
blocks. A code with no case, a skill with no page, a code with no
troubleshooting row, or a code with no alternatives fails the build.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

import raptorbt
import raptorbt.stats
from raptorbt import (
    BacktestConfig,
    KernelSession,
    LiveNode,
    PortfolioSession,
    Strategy,
    errors,
    run_strategies,
    run_strategy_backtest,
)
from raptorbt.errors import CODES, SKILLS, RaptorBTError
from raptorbt.strategy import orders

ROOT = Path(__file__).resolve().parents[2]
MIN = 60 * 10**9

#: code -> [(case id, trigger, parameter or None, check)]
CASES: dict[str, list] = {}


def case(code, parameter=None, *, name=None):
    """Register ``fn`` as a public-API trigger of ``code``.

    ``parameter`` is what the error must name (``None`` when it names none).
    """

    def register(fn):
        CASES.setdefault(code, []).append((name or fn.__name__, fn, parameter))
        return fn

    return register


# ---------------------------------------------------------------------------
# Fixtures shared by the triggers
# ---------------------------------------------------------------------------


class Noop(Strategy):
    pass


def bars(n=20, start=100.0):
    close = start + np.arange(n, dtype=np.float64)
    return {
        "timestamps": (np.arange(n, dtype=np.int64) + 1) * MIN,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.ones(n),
    }


def run(strategy=None, data=None, **kw):
    kw.setdefault("bar_stamp", "close")
    return run_strategies([strategy or Noop()], data or {"A": bars()}, **kw)


def single(**override):
    b = {**bars(), **override}
    kw = {"bar_stamp": "close", **{k: b.pop(k) for k in ("bar_stamp", "bar_ns") if k in b}}
    return run_strategy_backtest(
        Noop(), b["timestamps"], b["open"], b["high"], b["low"], b["close"], b["volume"], **kw
    )


# ---------------------------------------------------------------------------
# Timeframes, bar stamps, strategies and accounts
# ---------------------------------------------------------------------------


@case("missing_bar_stamp", "bar_stamp")
def _():
    run_strategies([Noop()], {"A": bars()})


@case("invalid_bar_stamp", "bar_stamp")
def _():
    run(bar_stamp="start")


@case("missing_bar_ns", "bar_ns")
def _():
    run(bar_stamp="open")


@case("invalid_bar_ns", "bar_ns")
def _():
    run(bar_stamp="open", bar_ns=0)


@case("invalid_primary_bars", "primary_bars")
def _():
    run(data={"A": bars()}, primary_bars="5m")


@case("invalid_bar_spec", "unit")
def _():
    raptorbt.BarAggregator(5, "fortnight")


@case("invalid_bar_step", "step")
def _():
    Noop().subscribe_bars(0, "m")


@case("primary_bars_without_ticks", "primary_bars")
def _():
    run(primary_bars=(5, "m"))


@case("invalid_strategy", "strategies")
def _():
    run_strategies([object()], {"A": bars()}, bar_stamp="close")


@case("duplicate_strategy", "strategies")
def _():
    s = Noop()
    run_strategies([s, s], {"A": bars()}, bar_stamp="close")


@case("empty_strategies", "strategies")
def _():
    run_strategies([], {"A": bars()}, bar_stamp="close")


@case("invalid_strategy_entry", "strategies")
def _():
    run_strategies([(Noop(), ["A"], "main")], {"A": bars()}, bar_stamp="close")


@case("invalid_accounts", "accounts")
def _():
    run(accounts=[1])


@case("invalid_account_spec", "initial_capital")
def _():
    raptorbt.AccountSpec(initial_capital=-1.0)


@case("unknown_account", "account")
def _():
    run_strategies([{"strategy": Noop(), "account": "desk"}], {"A": bars()}, bar_stamp="close")


@case("account_too_large", "accounts")
def _():
    from raptorbt.strategy.accounts import MAX_ACCOUNT_STRATEGIES

    spec = raptorbt.AccountSpec(initial_capital=1e6)
    many = [{"strategy": Noop(), "account": "desk"} for _ in range(MAX_ACCOUNT_STRATEGIES + 1)]
    run_strategies(many, {"A": bars()}, bar_stamp="close", accounts={"desk": spec})


@case("invalid_batch_size", "batch_size")
def _():
    run(batch_size=0)


@case("invalid_history", "history", name="live_node_history")
def _():
    LiveNode([Noop()], ["A"], bar_stamp="close", history=0)


@case("invalid_curve_samples", "curve_samples")
def _():
    LiveNode([Noop()], ["A"], bar_stamp="close", curve_samples=0)


@case("no_symbols", "data")
def _():
    run_strategies([Noop()], {}, bar_stamp="close")


@case("no_symbols", "symbols", name="live_node_symbols")
def _():
    LiveNode([Noop()], [], bar_stamp="close")


@case("duplicate_symbols", "symbols")
def _():
    LiveNode([Noop()], ["A", "A"], bar_stamp="close")


@case("unknown_symbol", "symbols")
def _():
    run_strategies([(Noop(), ["ZZZ"])], {"A": bars()}, bar_stamp="close")


# -- input data ---------------------------------------------------------------


@case("invalid_input_type", "data")
def _():
    run_strategies([Noop()], [1, 2], bar_stamp="close")


@case("missing_field", "data['A']['close']")
def _():
    b = bars()
    del b["close"]
    run(data={"A": b})


@case("invalid_array", "data['A']['close']")
def _():
    run(data={"A": {**bars(), "close": np.ones((20, 2))}})


@case("length_mismatch", "volume")
def _():
    single(volume=np.ones(5))


@case("decreasing_timestamps", "timestamps")
def _():
    single(timestamps=np.arange(20, 0, -1, dtype=np.int64) * MIN)


@case("empty_data", "timestamps")
def _():
    single(**{k: v[:0] for k, v in bars().items()})


@case("depth_without_ticks", "depth")
def _():
    run_strategies([Noop()], {"A": bars()}, depth={"B": {}}, bar_stamp="close")


@case("non_finite_price", "close")
def _():
    close = bars()["close"].copy()
    close[3] = np.nan
    single(close=close)




# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


def kernel(**kw):
    kw.setdefault("symbol", "X")
    return KernelSession(**kw)


def submit(session=None, **kw):
    """``KernelSession.submit_order`` with one rule broken by ``kw``."""
    base = {"side": "buy", "kind": "market", "submitted_idx": 0, "submitted_ts": MIN, "client_id": "c1", "units": 1.0}
    base.update(kw)
    return (session or kernel()).submit_order(**base)


class OnBar(Strategy):
    """Runs ``fn(strategy, ctx)`` on the first bar of a run."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.n = 0

    def on_bar(self, ctx, bar):
        self.n += 1
        if self.n == 1:
            self.fn(self, ctx)


def during_run(fn, **kw):
    return run(OnBar(fn), **kw)


@case("invalid_order_side", "side")
def _():
    orders.Market(side="hold", units=1)


@case("invalid_order_side", "side", name="kernel")
def _():
    submit(side="hold")


@case("invalid_time_in_force", "tif")
def _():
    orders.Market(side="buy", units=1, tif="forever")


@case("order_validity_unsupported", "tif")
def _():
    future = raptorbt.InstrumentSpec.futures_contract("F", 10**18, 50.0, venue="NFO")
    submit(KernelSession(symbol="F", instrument=future), tif="gtc", units=50.0)


@case("invalid_order_kind", "kind")
def _():
    submit(kind="iceberg")


@case("invalid_offset_kind", "offset_kind")
def _():
    orders.TrailingStopMarket(side="sell", units=1, offset=1.0, offset_kind="pct")


@case("invalid_offset_kind", "offset_kind", name="kernel")
def _():
    submit(side="sell", kind="trailing_stop_market", offset=1.0, offset_kind="pct")


@case("invalid_order_value", "price")
def _():
    orders.Limit(side="buy", units=1, price=0.0)


@case("missing_order_field", "expire_ns")
def _():
    orders.Market(side="buy", units=1, tif="gtd")


@case("conflicting_order_args", "units, size_frac")
def _():
    orders.Market(side="buy", units=1, size_frac=0.5)


@case("incompatible_order_options", "tif")
def _():
    orders.Limit(side="buy", units=1, price=100.0, tif="at_open")


@case("invalid_order_link", "client_ids")
def _():
    Noop().link_oco("only-one")


@case("off_lot_quantity", "units")
def _():
    future = raptorbt.InstrumentSpec.futures_contract("F", 10**18, 50.0, price_increment=0.05)
    submit(KernelSession(symbol="F", instrument=future), units=7.0)


@case("off_tick_price", "limit_price")
def _():
    future = raptorbt.InstrumentSpec.futures_contract("F", 10**18, 50.0, price_increment=0.05)
    submit(KernelSession(symbol="F", instrument=future), kind="limit", limit_price=100.03, units=50.0)


@case("unknown_order", "client_id")
def _():
    during_run(lambda s, ctx: s.cancel_order("never-sent"))


@case("bracket_under_hedging", "client_id")
def _():
    submit(KernelSession(symbol="X", oms_type="hedging"), reduce_only=True, parent_id=1)


@case("invalid_order_intent", "intents")
def _():
    def twice(s, ctx):
        s.enter(size_frac=0.5)
        s.enter(size_frac=0.5)

    during_run(twice)


@case("invalid_order_reference", "position_id")
def _():
    KernelSession(symbol="X").set_stop_price(10.0, position_id=999)


@case("invalid_order_tags", "tags")
def _():
    during_run(lambda s, ctx: s.submit_order(orders.Market(side="buy", units=1, tags={"client_id": "x"})))


@case("roll_price_missing", "to")
def _():
    from raptorbt.strategy.continuous import RollEvents

    specs = {
        "IDX": raptorbt.InstrumentSpec.equity("IDX"),
        "FA": raptorbt.InstrumentSpec.futures_contract("FA", rg.EXPA, 1.0, underlying="IDX"),
        "FB": raptorbt.InstrumentSpec.futures_contract("FB", rg.EXPB, 1.0, underlying="IDX"),
    }
    data = {}
    for k, s in enumerate(("IDX", "FA", "FB")):
        first = 28 if s == "FB" else 0  # FB has no bar before the roll
        ts = rg.T0C + (np.arange(first, 30, dtype=np.int64) + 1) * MIN
        c = 100.0 + 10 * k + np.arange(first, 30) * 0.1
        data[s] = {"timestamps": ts, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": np.full(len(ts), 100.0)}

    class Notional(rg.Carry):
        roll_sizing = "notional"

    run_strategies([Notional()], data, NO_FEES, instruments=specs, bar_stamp="close", bindings={"index": "IDX"})


@case("invalid_roll_sizing", "roll_sizing")
def _():
    import test_venue_parity as pa

    pa._roll_run("exposure")


# ---------------------------------------------------------------------------
# Accounts, configuration, instruments, sessions, fees, margin
# ---------------------------------------------------------------------------

DAY_NS = 86_400 * 10**9


def legs(shift=0):
    """Two instruments for a basket; the second one is ``shift`` ns off the first."""
    n = 20
    ts = np.arange(n, dtype=np.int64) * MIN
    c = np.linspace(100.0, 110.0, n)
    flat = np.zeros(n, bool)
    one = (ts, c, c + 1, c - 1, c, np.ones(n), flat, flat, 1, 1.0, "A")
    return [one, (ts + shift, *one[1:10], "B")]


@case("invalid_account_type", "account_type")
def _():
    KernelSession(symbol="X", account_type="spot")


@case("invalid_account_type", "account_type", name="portfolio")
def _():
    PortfolioSession(account_type="spot")


@case("invalid_leverage", "leverage")
def _():
    KernelSession(symbol="X", account_type="margin", leverage=0.0)


@case("invalid_oms_type", "oms_type")
def _():
    KernelSession(symbol="X", oms_type="spread")


@case("invalid_direction", "direction")
def _():
    KernelSession(symbol="X", direction=2)


@case("invalid_combine_mode", "combine_mode")
def _():
    g = legs()[0]
    raptorbt.run_multi_backtest(g[0], g[1], g[2], g[3], g[4], g[5], [(g[6], g[7], 1, 1.0, "s")], combine_mode="most")


@case("invalid_fill_timing", "fill_timing")
def _():
    BacktestConfig(fill_timing="whenever")


@case("invalid_allocation", "allocation")
def _():
    raptorbt.run_portfolio_backtest(legs(), allocation="kelly")


def option_run(**kw):
    g = legs()[0]
    return raptorbt.run_options_backtest(g[0], g[1], g[2], g[3], g[4], g[5], g[4], g[6], g[7], 1, "O", **kw)


@case("invalid_option_type", "option_type")
def _():
    option_run(option_type="straddle")


@case("invalid_strike_selection", "strike_selection")
def _():
    option_run(strike_selection="otm9")


@case("invalid_size_type", "size_type")
def _():
    option_run(size_type="lots")


@case("unsupported_short_ticks", "quantity")
def _():
    g = legs()[0]
    raptorbt.run_tick_backtest(g[0], g[4], g[4], g[4], g[5], g[5], g[5], g[6], g[7], quantity=-1)


@case("invalid_squareoff_time", "squareoff_time")
def _():
    BacktestConfig(squareoff_time="25:99")


@case("untradable_instrument", "instrument")
def _():
    KernelSession(symbol="X", instrument=raptorbt.InstrumentSpec.index("NIFTY"))


@case("invalid_session_schedule", "windows")
def _():
    raptorbt.SessionSchedule({"funday": [(540, 900)]})


@case("invalid_corporate_action", "kind")
def _():
    raptorbt.CorporateAction("A", "merger", DAY_NS)


@case("invalid_adjustment_policy", "adjustment_policy")
def _():
    run(corporate_actions=[raptorbt.CorporateAction.split("A", 5 * MIN, 2.0)])


@case("unsupported_corporate_actions", "corporate_actions")
def _():
    raptorbt.run_basket_backtest(legs(), corporate_actions=[raptorbt.CorporateAction.split("A", 5 * MIN, 2.0)])


@case("invalid_settlement", "settlement_style")
def _():
    raptorbt.InstrumentSpec.futures_contract("F", DAY_NS * 30, 1.0).settlement_style = "barter"


@case("invalid_fee_schedule")
def _():
    raptorbt.FeeSchedule.from_json("{")


@case("fee_schedule_gap", "fee_schedule")
def _():
    b = bars()
    single_kw = dict(bar_stamp="close", config=BacktestConfig(fee_schedule=raptorbt.FeeSchedule.exchange_presets()))
    run_strategy_backtest(Noop(), b["timestamps"], b["open"], b["high"], b["low"], b["close"], b["volume"], **single_kw)


@case("squareoff_needs_timezone", "squareoff_time")
def _():
    run(config=BacktestConfig(squareoff_time="15:25"))


@case("mixed_holding", "requirements.holding")
def _():
    from raptorbt.strategy import Holding, Requirements

    class Early(Noop):
        requirements = Requirements(holding=Holding.intraday("15:00"))

    class Late(Noop):
        requirements = Requirements(holding=Holding.intraday("15:20"))

    run_strategies([Early(), Late()], {"A": bars()}, bar_stamp="close")


@case("session_not_covered", "session")
def _():
    spec = raptorbt.InstrumentSpec.equity("A", session=raptorbt.SessionSchedule.mcx())
    run(instruments={"A": spec})


@case("invalid_calendar", "documents")
def _():
    raptorbt.SessionSchedule.from_calendars("NSE", ["not json"])


@case("invalid_margin_model", "leverage")
def _():
    raptorbt.MarginModel.from_spec(leverage=0.0)


@case("money_overflow")
def _():
    KernelSession(symbol="X", config=BacktestConfig(initial_capital=1e300))


@case("invalid_money_decimals", "money_decimals")
def _():
    BacktestConfig(money_decimals=12)


@case("invalid_session_date", "date")
def _():
    raptorbt.SessionSchedule.nse_equity().session_close_ns(20251399)


@case("invalid_instrument_metadata", "venue")
def _():
    raptorbt.InstrumentSpec.equity("A", venue=" ")


@case("mcx_session_unknown", "session")
def _():
    raptorbt.InstrumentSpec.equity("ZZZ", venue="MCX")


@case("invalid_option_right", "right")
def _():
    raptorbt.InstrumentSpec.option("O", 100.0, "sideways", DAY_NS * 30, 1.0)


@case("invalid_expiry_pick", "which")
def _():
    raptorbt.OptionChain([]).pick_expiry("U", 0, which="fortnightly")


@case("invalid_parameter", "fsync")
def _():
    rj.Journal({"run": 1}, memory=True, fsync="sometimes")


@case("invalid_argument_type", "order")
def _():
    Noop().submit_order("buy")


@case("unknown_skill", "name")
def _():
    from raptorbt import skills

    skills.load("no-such-skill")


@case("invalid_config", "percent")
def _():
    BacktestConfig().set_fixed_stop(5.0)


@case("misaligned_legs", "instruments")
def _():
    raptorbt.run_basket_backtest(legs(shift=1))


@case("invalid_data", "timestamp")
def _():
    raptorbt.adjust_history("A", [], {"open": np.ones(3)})


# ---------------------------------------------------------------------------
# The strategy API: subscriptions, data views, indicators, roles
# ---------------------------------------------------------------------------


class Declares(Strategy):
    """Runs ``fn(strategy, ctx)`` in ``on_start``."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def on_start(self, ctx):
        self.fn(self, ctx)


def at_start(fn, **kw):
    return run(Declares(fn), **kw)


@case("invalid_timeframe", "timeframe")
def _():
    at_start(lambda s, ctx: s.subscribe("A", timeframe="1minute"))


@case("invalid_data_type", "data_type")
def _():
    at_start(lambda s, ctx: s.subscribe("A", data_type="candle"))


@case("invalid_history", "history")
def _():
    at_start(lambda s, ctx: s.subscribe("A", timeframe="5m", history=0))


@case("not_subscribed_timeframe", "timeframe")
def _():
    during_run(lambda s, ctx: ctx.data.latest("A", timeframe="5m"))


@case("not_subscribed_data", "instrument")
def _():
    during_run(lambda s, ctx: ctx.data.quote("A"))


@case("invalid_indicator_input", "field")
def _():
    at_start(lambda s, ctx: ctx.indicators.sma(instrument="A", period=3, field="high"))


@case("declaration_outside_run", "subscribe()")
def _():
    during_run(lambda s, ctx: s.subscribe("A", timeframe="5m"))


@case("portfolio_total_unavailable")
def _():
    class Reads(Strategy):
        def on_bar(self, ctx, bar):
            ctx.portfolio.gross_exposure

    b = bars()
    run_strategy_backtest(Reads(), b["timestamps"], b["open"], b["high"], b["low"], b["close"], b["volume"], bar_stamp="close")


@case("no_requirements", "name")
def _():
    during_run(lambda s, ctx: ctx.role("fut"))


def with_roles(**roles):
    from raptorbt.strategy import Holding, Requirements

    class Roled(OnBar):
        requirements = Requirements(roles=roles, timeframes=["1m"], holding=Holding.overnight(), data=["bar"])

    return Roled


@case("unknown_role", "name")
def _():
    from raptorbt.strategy import Instrument

    Roled = with_roles(index=Instrument(symbol="A"))
    run(Roled(lambda s, ctx: ctx.role("fut")))


@case("unresolved_role", "name")
def _():
    from raptorbt.strategy import Future, Underlying

    Roled = with_roles(index=Underlying(asset="index"), fut=Future(of="index"))
    run(Roled(lambda s, ctx: ctx.role("fut")))


@case("unknown_role_binding", "bindings")
def _():
    run(bindings={"fut": "A"})


@case("invalid_binding", "bindings")
def _():
    from raptorbt.strategy import Underlying

    run(with_roles(index=Underlying(asset="index"))(lambda s, ctx: None), bindings={"index": ["A", "B"]})


@case("invalid_binding", "bindings", name="universe_given_a_symbol")
def _():
    from raptorbt.strategy import Universe

    run(with_roles(basket=Universe(min=1, max=2))(lambda s, ctx: None), bindings={"basket": "A"})


@case("universe_size", "bindings")
def _():
    from raptorbt.strategy import Universe

    run(with_roles(basket=Universe(min=2, max=3))(lambda s, ctx: None), bindings={"basket": ["A"]})


@case("universe_instrument", "symbol_or_role")
def _():
    from raptorbt.strategy import Universe

    Roled = with_roles(basket=Universe(min=1, max=2))
    run(Roled(lambda s, ctx: ctx.instrument("basket")), bindings={"basket": ["A"]})


@case("invalid_trade_from", "trade_from")
def _():
    run(trade_from="09:30")


@case("no_instrument_spec", "symbol_or_role")
def _():
    during_run(lambda s, ctx: ctx.instrument("A"))


@case("invalid_adjust", "adjust")
def _():
    from raptorbt.strategy.continuous import continuous_series

    continuous_series({}, {}, roll_days=1.0, adjust="splice")


@case("no_continuous_bars", "contracts")
def _():
    from raptorbt.strategy.continuous import continuous_series

    spec = raptorbt.InstrumentSpec.futures_contract("FA", MIN // 2, 1.0)
    continuous_series({"FA": bars()}, {"FA": spec}, roll_days=0.0, adjust="raw")


@case("invalid_input")
def _():
    raptorbt.sma(np.ones(5), 0)


# -- expressions ----------------------------------------------------------------------------------


@case("expr_syntax", "expression")
def _():
    raptorbt.expr("close +")


@case("expr_unknown_function", "expression")
def _():
    raptorbt.expr("smaa(close, 5)")


@case("expr_type", "expression")
def _():
    raptorbt.expr("close and open")


@case("expr_type", "expression", name="truth_test_of_an_expression")
def _():
    bool(raptorbt.col.close > 1)


@case("expr_unbound_parameter", "expression")
def _():
    raptorbt.expr("sma(close, $n)").eval({"close": np.arange(5.0)})


@case("sweep_too_large", "max_combinations")
def _():
    raptorbt.expr("sma(close, $n)").sweep({"close": np.arange(20.0)}, n=[2, 3, 4], max_combinations=2)


@case("expr_unsupported_here", "expression")
def _():
    raptorbt.expr('at("1d", at("1d", close))')


# -- statistics -----------------------------------------------------------------------------------


@case("insufficient_observations")
def _():
    raptorbt.stats.hac_tstat(np.arange(5, dtype=np.float64))


@case("degenerate_series")
def _():
    raptorbt.stats.hac_tstat(np.full(60, 0.01))


@case("did_not_converge")
def _():
    # a variance past the double range: no start has a finite likelihood
    raptorbt.stats.markov_regimes(np.tile([1e170, -1e170, 2e170, -3e170], 100))


# -- analysis sessions ----------------------------------------------------------------------------


def _session(**kw):
    from raptorbt.analysis import Session

    s = Session(**kw)
    s.load({"timestamps": bars()["timestamps"], "close": bars()["close"]}, bar_stamp="close")
    return s


@case("unknown_handle", "handle")
def _():
    _session().eval("sma(h_0000000000, 3)")


@case("handle_expired", "handle")
def _():
    s = _session(max_handles=1)
    first = s.eval("sma(close, 3)")["handle"]
    s.eval("sma(close, 4)")
    s.series(first)


@case("session_memory_exceeded", "max_bytes")
def _():
    _session(max_bytes=400).eval("sma(close, 3)")


@case("below_one_lot", "lots")
def _():
    from raptorbt import sizing

    sizing.lots_for_percent_of_equity(1_000.0, 10.0, 100.0, 50.0)


@case("invalid_sizing_input", "capital")
def _():
    from raptorbt import sizing

    sizing.lots_for_risk(-1.0, 0.01, 100.0, 95.0, 50.0)


@case("invalid_order_check", "order_check")
def _():
    run(order_check="not callable")


@case("invalid_order_refusal", "order_check")
def _():
    run(OnBar(lambda s, ctx: s.submit_order(orders.Market(side="buy", units=1))), order_check=lambda order, ctx: 7)


@case("invalid_order_check_scope", "order_check_scope")
def _():
    run(order_check=lambda order, ctx: None, order_check_scope="intents")


@case("order_check_intent", "order_check")
def _():
    run(OnBar(lambda s, ctx: s.enter(size_frac=0.5)), order_check=lambda order, ctx: None)


@case("unsupported_on_bars", "order_latency_ns")
def _():
    run(config=BacktestConfig(order_latency_ns=1))


# ---------------------------------------------------------------------------
# Live nodes: journal, recovery, fencing, venues
# ---------------------------------------------------------------------------

from raptorbt import journal as rj  # noqa: E402
from raptorbt.errors import InvalidData, InvalidParameter, StateError  # noqa: E402
from raptorbt.execution import ExecutionReport  # noqa: E402
from test_journal import (  # noqa: E402
    CONFIG,
    HEADER,
    SYMBOLS,
    MockVenue,
    Trader,
    build,
    lost_order_story,
    midway,
    script,
)

NO_FEES = BacktestConfig(initial_capital=1_000_000.0, fees=0.0)


def restore_node(records, execution, strategy=Trader, symbols=SYMBOLS, **kw):
    return LiveNode.restore(
        None, [strategy()], symbols, CONFIG, journal=rj.Journal(HEADER, memory=True),
        records=list(records), execution=execution, bar_stamp="close", **kw,
    )


@case("recovering")
def _():
    venue, store = lost_order_story()
    node = restore_node(store, venue, on_discrepancy=lambda d: None)
    node.push_bar("A", 99 * MIN, 1.0, 1.0, 1.0, 1.0, 1.0)


@case("no_reconcile")
def _():
    class Plain:
        namespace = "P1"
        keeps_own_records = True

        def submit(self, r):
            pass

        def cancel(self, c):
            pass

        def modify(self, c, **k):
            pass

    plain = Plain()
    j = rj.Journal(HEADER, memory=True)
    node = LiveNode([Trader()], SYMBOLS, CONFIG, bar_stamp="close", execution=plain, journal=j)
    for step in script(3):
        if step[0] == "bar" and step[1] == "A":
            node.push_bar(step[1], step[2], *step[3:8])
    restore_node(j.since(0), plain)


@case("fenced")
def _():
    venue = MockVenue()
    node = build(venue, rj.Journal(HEADER, memory=True), epoch=1)
    node.heartbeat()
    build(venue, rj.Journal(HEADER, memory=True), epoch=2).heartbeat()
    node.heartbeat()


@case("journal_failed")
def _():
    def refuse(_):
        raise OSError("disk full")

    build(MockVenue(), rj.Journal(HEADER, sink=refuse))


@case("journal_evicted", "seq")
def _():
    j = rj.Journal(HEADER, memory=5)
    node = build(MockVenue(), j)
    for step in script(6):
        if step[0] == "bar" and step[1] == "A":
            node.push_bar(step[1], step[2], *step[3:8])
    j.since(0)


@case("invalid_checkpoint", "checkpoint")
def _():
    node, venue, j, steps = midway()
    rj.decode_checkpoint(node.checkpoint()[:-5])


@case("replay_diverged")
def _():
    node, venue, j, steps = midway()
    body = rj.decode_checkpoint(node.checkpoint())
    body["digest"] = "0" * 64
    LiveNode.restore(rj.encode_checkpoint(body), [Trader()], SYMBOLS, CONFIG, execution=venue, bar_stamp="close")


def epoch_store(epoch=4):
    venue = MockVenue()
    store = []
    node = build(venue, rj.Journal(HEADER, sink=lambda r: store.append(r.canonical())), epoch=epoch)
    for step in script(4):
        if step[0] == "bar" and step[1] == "A":
            node.push_bar(step[1], step[2], *step[3:8], arrival_ns=step[2] + 7)
    return venue, store


@case("epoch_required")
def _():
    venue, store = epoch_store()
    restore_node(store, venue)


@case("stale_epoch")
def _():
    venue, store = epoch_store()
    restore_node(store, venue, epoch=4)


import test_live_node_instruments as lni  # noqa: E402
import test_recovery_gate as rg  # noqa: E402
import test_venue_correctness as vc  # noqa: E402
from raptorbt.errors import InvalidOrder  # noqa: E402
from raptorbt.strategy.exits import MAX_CLOSE_ATTEMPTS, ExitRule  # noqa: E402,F401
from test_execution import Hooked, Venue, bar as exec_bar, node_with  # noqa: E402


class Reader(Strategy):
    pass


def grouped_node():
    return LiveNode([Reader()], ["A", "B"], NO_FEES, bar_stamp="close")


@case("group_open")
def _():
    node = grouped_node()
    node.begin_group()
    node.begin_group()


@case("group_timestamp_mismatch", "timestamp")
def _():
    node = grouped_node()
    node.push_group(MIN, {"A": (100.0,) * 4 + (1000,), "B": (50.0,) * 4 + (1000,)})
    node.begin_group(2 * MIN)
    node.push_bar("A", 2 * MIN, 101.0, 101.0, 101.0, 101.0, 1000)
    node.push_bar("B", 3 * MIN, 51.0, 51.0, 51.0, 51.0, 1000)


@case("duplicate_bar", "timestamp")
def _():
    node = grouped_node()
    node.push_bar("A", MIN, 100, 100, 100, 100, 1)
    node.push_bar("A", MIN, 100, 100, 100, 100, 1)


@case("conflicting_bar", "timestamp")
def _():
    node = grouped_node()
    node.push_bar("A", MIN, 100, 100, 100, 100, 1)
    node.push_bar("A", MIN, 101, 101, 101, 101, 1)


@case("instrument_in_use", "symbol")
def _():
    node, mover = lni.build()
    node.add_instrument("B", lni.spec("B"))
    lni.run(node, lni.feed(["A", "B"], 0, 3))
    node.retire_instrument("B")


@case("invalid_strategies", "strategies")
def _():
    node, _ = lni.build()
    node.add_instrument("C", lni.spec("C"), strategies=[5])


@case("invalid_instrument_spec", "spec")
def _():
    node, _ = lni.build()
    node.add_instrument("C", lni.spec("Z"))


@case("adoption_refused", "quantity")
def _():
    LiveNode([Noop()], ["A"], BacktestConfig(), bar_stamp="close", initial_positions={0: {"A": {"quantity": -5.0, "avg_price": 100.0}}})


@case("invalid_initial_positions", "initial_positions[0]['A']['margin_minor']")
def _():
    LiveNode([Noop()], ["A"], BacktestConfig(), bar_stamp="close", initial_positions={0: {"A": {"quantity": 5.0, "avg_price": 100.0, "margin_minor": 1.5}}})


@case("invalid_warmup", "warmup")
def _():
    LiveNode([Noop()], ["A"], NO_FEES, bar_stamp="close", warmup="cold")


@case("invalid_execution", "execution")
def _():
    LiveNode([Hooked()], ["A"], execution=object(), bar_stamp="close")


@case("unknown_client_order_id", "client_order_id")
def _():
    from raptorbt.execution import ExecutionReport

    venue = Venue()
    node = node_with(venue, [Hooked(lambda s, c, n: s.submit_order(orders.Market(side="buy", units=1.0)) if n == 1 else None)], symbols=("A",))
    exec_bar(node, "A", 0)
    node.push_report("s9-9", ExecutionReport.accepted("V", 1), arrival_ns=1)


@case("invalid_execution_report")
def _():
    from raptorbt.execution import ExecutionReport

    venue = Venue()
    node = node_with(venue, [Hooked(lambda s, c, n: s.submit_order(orders.Market(side="buy", units=1.0)) if n == 1 else None)], symbols=("A",))
    exec_bar(node, "A", 0)
    cid = venue.requests[0].client_order_id
    node.push_report(cid, ExecutionReport.fill("F", 5.0, 1.0, 1), arrival_ns=1)


@case("duplicate_client_order_id", "client_id")
def _():
    venue = Venue()

    def script(s, ctx, n):
        s.submit_order(orders.Limit(side="buy", price=90.0 + n, units=5.0), client_id="same")

    node = node_with(venue, [Hooked(script)], symbols=("A",))
    exec_bar(node, "A", 0)
    exec_bar(node, "A", 1)


@case("node_poisoned")
def _():
    from raptorbt.execution import ExecutionReport

    venue = Venue()
    node = node_with(venue, [Hooked(lambda s, c, n: s.submit_order(orders.Market(side="buy", units=1.0)) if n == 1 else None)], symbols=("A",))
    exec_bar(node, "A", 0)
    cid = venue.requests[0].client_order_id
    try:
        node.push_report(cid, ExecutionReport.fill("E1", 1.0, 1e300, MIN), arrival_ns=MIN)
    except Exception:
        pass
    exec_bar(node, "A", 1)


@case("fence_unsupported", "epoch")
def _():
    class Bare:
        namespace = "B1"
        keeps_own_records = True

        def submit(self, r):
            pass

        def cancel(self, c):
            pass

        def modify(self, c, **k):
            pass

    LiveNode([Trader()], SYMBOLS, CONFIG, bar_stamp="close", execution=Bare(), journal=rj.Journal(HEADER, memory=True), epoch=3)


@case("adapter_namespace_missing", "execution")
def _():
    class NoNamespace:
        def submit(self, r):
            pass

        def cancel(self, c):
            pass

        def modify(self, c, **k):
            pass

    LiveNode([Trader()], SYMBOLS, CONFIG, bar_stamp="close", execution=NoNamespace())


@case("adapter_namespace_mismatch", "execution")
def _():
    from raptorbt.adapters.conformance import ScriptedVenue

    venue = ScriptedVenue()
    store = []
    rg.drive_buyer(rg.broker_adapter(venue), venue, store)
    LiveNode.restore(
        None, [rg.Buyer()], ["A"], NO_FEES, journal=rj.Journal(HEADER, memory=True), records=list(store),
        execution=rg.broker_adapter(venue, namespace="ZZ"), bar_stamp="close",
    )


@case("adapter_cannot_adopt", "execution")
def _():
    from raptorbt.adapters import Session
    from raptorbt.adapters.conformance import ScriptedVenue

    venue = ScriptedVenue()
    store = []
    rg.drive_buyer(rg.broker_adapter(venue), venue, store)
    blind = type(rg.broker_adapter(venue))(Session("t"), lambda symbol: None, sleep=lambda s: None, namespace="CF", trusted=True)
    LiveNode.restore(
        None, [rg.Buyer()], ["A"], NO_FEES, journal=rj.Journal(HEADER, memory=True), records=list(store),
        execution=blind, bar_stamp="close",
    )


@case("hedging_unsupported_by_venue", "oms_type")
def _():
    from raptorbt.adapters.conformance import ScriptedVenue

    LiveNode([rg.Buyer()], ["A"], NO_FEES, bar_stamp="close", execution=rg.broker_adapter(ScriptedVenue()), oms_type="hedging")


@case("correction_not_reversible")
def _():
    from raptorbt.execution import ExecutionReport

    v = MockVenue()
    node = LiveNode([vc.Stack({1: ("buy", 5.0, False), 2: ("sell", 8.0, False)})], ["A"], NO_FEES,
                    bar_stamp="close", execution=v, journal=rj.Journal(HEADER, memory=True))
    for i in range(1, 4):
        node.push_bar("A", i * MIN, 100, 100, 100, 100, 1)
        v.fill_all()
        v.deliver(node, i * MIN + 5)
    second = list(v.orders)[1]
    node.push_report(second, ExecutionReport.bust(f"X-{second}", 4 * MIN), arrival_ns=4 * MIN)


@case("exit_rule_not_cancellable")
def _():
    v = rg.HoldingVenue()
    node = vc.trim_node(v)
    vc.run_bars(node, v, 1, 4)
    s = node._strategies[0]
    s.cancel_exit(s.handle)


@case("exit_rule_unconfirmed")
def _():
    class Silent(rg.HoldingVenue):
        def place_exit_rule(self, request):
            self.rules.append(request)
            raise RuntimeError("connection reset after the request left")

    v = Silent()
    node = vc.trim_node(v)
    vc.run_bars(node, v, 1, 3)


@case("exit_close_failed")
def _():
    v = rg.RefuseReduceOnly()
    node = LiveNode([rg.Guard()], ["A"], NO_FEES, bar_stamp="close", execution=v)
    for i, p in enumerate([100, 100] + [80] * 40, start=1):
        node.push_bar("A", i * MIN, p, p, p, p, 1)
        v.fill_all()
        v.deliver(node, i * MIN + 5)


@case("roll_close_failed")
def _():
    class Strict(rg.Carry):
        on_roll_failed = rg.RollEvents.on_roll_failed

    v = rg.RefuseReduceOnly()
    specs = {
        "IDX": raptorbt.InstrumentSpec.equity("IDX"),
        "FA": raptorbt.InstrumentSpec.futures_contract("FA", rg.EXPA, 1.0, underlying="IDX"),
        "FB": raptorbt.InstrumentSpec.futures_contract("FB", rg.EXPB, 1.0, underlying="IDX"),
    }
    node = LiveNode([Strict()], ["IDX", "FA", "FB"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close",
                    execution=v, instruments=specs, bindings={"index": "IDX"})
    rg.drive_roll(node, v)


@case("roll_open_failed")
def _():
    from raptorbt.execution import OrderRefused

    class RefuseOpen(MockVenue):
        def submit(self, request):
            if request.symbol == "FB":
                raise OrderRefused("no_new_positions")
            super().submit(request)

    class Strict(rg.Carry):
        on_roll_failed = rg.RollEvents.on_roll_failed

    specs = {
        "IDX": raptorbt.InstrumentSpec.equity("IDX"),
        "FA": raptorbt.InstrumentSpec.futures_contract("FA", rg.EXPA, 1.0, underlying="IDX"),
        "FB": raptorbt.InstrumentSpec.futures_contract("FB", rg.EXPB, 1.0, underlying="IDX"),
    }
    v = RefuseOpen()
    node = LiveNode([Strict()], ["IDX", "FA", "FB"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close",
                    execution=v, instruments=specs, bindings={"index": "IDX"})
    rg.drive_roll(node, v)


# -- option structures --------------------------------------------------------------------------

import test_structures as tst  # noqa: E402

st = tst.st


@case("invalid_structure", "side")
def _():
    st.Leg("call", side="hold")


@case("structure_contract_missing", "strike")
def _():
    st.strangle(4).resolve(tst.CHAIN, tst.U, 100.2, tst.NOW)


def condor_with_prices():
    ic = st.iron_condor(1, 2)
    resolved = ic.resolve(tst.CHAIN, tst.U, 100.2, tst.NOW)
    return ic, {leg.instrument: 2.0 for leg in resolved.legs}


@case("structure_limit_breached", "net_limit")
def _():
    ic, px = condor_with_prices()
    Strategy().submit_structure(
        tst.ctx_at(tst.NOW), ic, prices=px, net_limit=-5.0, chain=tst.CHAIN, underlying=tst.U, spot=100.2
    )


@case("structure_margin_exceeded", "available")
def _():
    ic, px = condor_with_prices()
    Strategy().submit_structure(
        tst.ctx_at(tst.NOW), ic, prices=px, margin_model=raptorbt.MarginModel.defined_risk(), available=1.0,
        chain=tst.CHAIN, underlying=tst.U, spot=100.2,
    )


@case("structure_missing_mark", "prices")
def _():
    resolved = st.iron_condor(1, 2).resolve(tst.CHAIN, tst.U, 100.2, tst.NOW)
    resolved.margin(raptorbt.MarginModel.defined_risk(), {})


# -- lifecycle of sessions and nodes, the runtime -----------------------------------------------


@case("session_finished")
def _():
    s = KernelSession(symbol="X")
    s.step(0, 0, 100.0, 101.0, 99.0, 100.0, 1000.0)
    s.finish()
    s.finish()


@case("node_finished")
def _():
    node = LiveNode([Noop()], ["A"], bar_stamp="close")
    node.finish()
    node.finish()


@case("run_finished")
def _():
    s = KernelSession(symbol="X")
    s.step(0, 0, 100.0, 101.0, 99.0, 100.0, 1000.0)
    s.finish()
    s.halt()


@case("runtime_busy")
def _():
    holder = {}

    class Reenters(Strategy):
        def on_bar(self, ctx, bar):
            holder["node"].push_bar("A", 2 * MIN, 1.0, 1.0, 1.0, 1.0, 1.0)

    holder["node"] = LiveNode([Reenters()], ["A"], bar_stamp="close")
    holder["node"].push_bar("A", MIN, 1.0, 1.0, 1.0, 1.0, 1.0)


@case("history_exhausted")
def _():
    class Keeps(Strategy):
        def on_start(self, ctx):
            self.subscribe("A", history=50)

    node = LiveNode([Keeps()], ["A"], bar_stamp="close", history=3)
    for i in range(1, 10):
        node.push_bar("A", i * MIN, 1.0, 1.0, 1.0, 1.0, 1.0)


@case("invalid_exit_rule", "rule")
def _():
    ExitRule()


@case("roll_target_not_subscribed", "symbols")
def _():
    specs = {
        "IDX": raptorbt.InstrumentSpec.equity("IDX"),
        "FA": raptorbt.InstrumentSpec.futures_contract("FA", rg.EXPA, 1.0, underlying="IDX"),
        "FB": raptorbt.InstrumentSpec.futures_contract("FB", rg.EXPB, 1.0, underlying="IDX"),
    }
    data = {}
    for k, s in enumerate(("IDX", "FA")):
        c = 100.0 + 10 * k + np.arange(30) * 0.1
        data[s] = {
            "timestamps": rg.T0C + (np.arange(30, dtype=np.int64) + 1) * MIN,
            "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": np.full(30, 100.0),
        }
    run_strategies([rg.Carry()], data, NO_FEES, instruments=specs, bar_stamp="close", bindings={"index": "IDX"})


# -- venues ---------------------------------------------------------------------------------------


@case("venue_requires_units", "units")
def _():
    class Modifies(Strategy):
        n = 0

        def on_bar(self, ctx, bar):
            self.n += 1
            if self.n == 1:
                self.submit_order(orders.Limit(side="buy", units=1.0, price=50.0), client_id="x")
            if self.n == 2:
                self.modify_order("x", size_frac=0.5)

    node = LiveNode([Modifies()], ["A"], NO_FEES, bar_stamp="close", execution=MockVenue())
    for i in range(1, 4):
        node.push_bar("A", i * MIN, 100.0, 100.0, 100.0, 100.0, 1.0)


@case("unsupported_in_venue_run", "warmup")
def _():
    class Buys(Strategy):
        def on_bar(self, ctx, bar):
            self.submit_order(orders.Market(side="buy", units=1.0))

    LiveNode([Buys()], ["A"], NO_FEES, bar_stamp="close", execution=MockVenue(), warmup="live", warmup_bars={"A": bars(5)})


@case("venue_unavailable")
def _():
    from raptorbt.execution import OrderRefused

    venue, store = lost_order_story()
    node = restore_node(store, venue, on_discrepancy=lambda d: None)
    venue.fail_next = OrderRefused("market closed")
    node.resend("s0-0")


@case("venue_fill_not_booked")
def _():
    from raptorbt.execution import ExecutionReport

    venue = Venue()
    node = node_with(
        venue,
        [Hooked(lambda s, c, n: s.submit_order(orders.Market(side="sell", units=1.0, reduce_only=True)))],
        symbols=("A",),
    )
    for i in range(1100):
        exec_bar(node, "A", i)
        cid = venue.requests[-1].client_order_id
        node.push_report(cid, ExecutionReport.accepted(f"V{i}", (i + 1) * MIN), arrival_ns=(i + 1) * MIN)
        node.push_report(cid, ExecutionReport.fill(f"E{i}", 1.0, 100.0, (i + 1) * MIN), arrival_ns=(i + 1) * MIN)


# ===========================================================================
# The matrix
# ===========================================================================

SKILLS_DIR = ROOT / "skills"
TROUBLESHOOTING = (SKILLS_DIR / "troubleshooting" / "SKILL.md").read_text()
CASE_IDS = [(code, name) for code in sorted(CASES) for name, _, _ in CASES[code]]


def _case(code, name):
    return next(c for c in CASES[code] if c[0] == name)


@pytest.mark.parametrize(("code", "name"), CASE_IDS, ids=[f"{c}:{n}" for c, n in CASE_IDS])
def test_the_public_api_raises_the_code(code, name):
    _, trigger, parameter = _case(code, name)
    entry = CODES[code]
    with pytest.raises(RaptorBTError) as info:
        trigger()
    e = info.value
    assert e.code == code
    assert isinstance(e, entry.cls)
    assert e.skill == entry.skill
    assert e.parameter == parameter
    assert e.operation, "an error names the call that failed"
    assert e.valid, "an error lists the alternatives, or states the rule when the set is open"
    text = str(e)
    assert text.splitlines()[0] == e.message
    assert f"Operation:\n    {e.operation}" in text
    assert "Supported:\n    " in text
    assert f"Related skill:\n    raptorbt-{entry.skill}" in text
    if parameter is not None:
        assert f"Requested:\n    {parameter}" in text
    assert errors.from_dict(json.loads(json.dumps(e.to_dict()))).to_dict() == e.to_dict()


def test_every_code_has_a_public_api_case():
    assert sorted(set(CODES) - set(CASES)) == []


def test_no_case_names_a_code_that_is_not_registered():
    assert sorted(set(CASES) - set(CODES)) == []


def test_every_code_lists_alternatives_or_the_rule_in_the_registry():
    assert sorted(c for c, e in CODES.items() if not e.valid) == []


def test_every_code_has_a_skill_page():
    for code, entry in CODES.items():
        assert entry.skill in SKILLS, code
        assert (SKILLS_DIR / entry.skill / "SKILL.md").is_file(), (code, entry.skill)


def test_every_code_has_a_troubleshooting_row():
    missing = [c for c in CODES if f"`{c}`" not in TROUBLESHOOTING]
    assert missing == []


# ---------------------------------------------------------------------------
# The alternatives an error lists are accepted by the call that refused them
# ---------------------------------------------------------------------------


def _accept_bar_stamp(v):
    run(bar_stamp=v, bar_ns=MIN)


ACCEPTS = {
    "invalid_bar_stamp": _accept_bar_stamp,
    "invalid_account_type": lambda v: KernelSession(symbol="X", account_type=v, leverage=2.0),
    "invalid_oms_type": lambda v: KernelSession(symbol="X", oms_type=v),
    "invalid_direction": lambda v: KernelSession(symbol="X", direction=v),
    "invalid_fill_timing": lambda v: BacktestConfig(fill_timing=v),
    "invalid_order_side": lambda v: orders.Market(side=v, units=1),
    "invalid_time_in_force": lambda v: orders.Market(side="buy", units=1, tif=v, expire_ns=10 * MIN),
    "invalid_offset_kind": lambda v: orders.TrailingStopMarket(side="sell", units=1, offset=1.0, offset_kind=v),
    "invalid_option_type": lambda v: option_run(option_type=v),
    "invalid_strike_selection": lambda v: option_run(strike_selection=v),
    "invalid_size_type": lambda v: option_run(size_type=v),
    "invalid_allocation": lambda v: raptorbt.run_portfolio_backtest(legs(), allocation=v),
    "invalid_order_check_scope": lambda v: run(order_check=lambda order, ctx: None, order_check_scope=v),
    "invalid_warmup": lambda v: LiveNode([Noop()], ["A"], NO_FEES, bar_stamp="close", warmup=v),
    "invalid_timeframe": lambda v: at_start(lambda s, ctx: s.subscribe("A", timeframe=v)),
    "invalid_data_type": lambda v: at_start(lambda s, ctx: s.subscribe("A", data_type=v)),
    "invalid_adjust": lambda v: _accept_adjust(v),
    "invalid_adjustment_policy": lambda v: run(
        corporate_actions=[raptorbt.CorporateAction.split("A", 5 * MIN, 2.0)], adjustment_policy=v
    ),
}


def _accept_adjust(v):
    from raptorbt.strategy.continuous import continuous_series

    spec = raptorbt.InstrumentSpec.futures_contract("FA", 10**18, 1.0)
    continuous_series({"FA": bars()}, {"FA": spec}, roll_days=0.0, adjust=v)


def _enumerated(code):
    return [v for v in CODES[code].valid if isinstance(v, (str, int)) and not isinstance(v, bool)]


@pytest.mark.parametrize("code", sorted(ACCEPTS))
def test_every_alternative_an_error_lists_is_accepted(code):
    values = [v for v in _enumerated(code) if not (isinstance(v, str) and (" " in v or v == "None"))]
    assert values, code
    for value in values:
        try:
            ACCEPTS[code](value)
        except RaptorBTError as e:
            raise AssertionError(f"{code}: the listed alternative {value!r} was refused: {e.code}") from e
