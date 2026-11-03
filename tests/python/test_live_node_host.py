"""What an embedding host needs of a LiveNode.

Each behaviour runs the same strategy through ``run_strategies``, a
``LiveNode`` that matches in-process and a ``LiveNode`` on
``PaperExecution`` wherever all three apply, and through an adapter of your
own where only a venue can show the effect.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from raptorbt import BacktestConfig, LiveNode, Strategy, run_strategies
from raptorbt.errors import InvalidOrder, InvalidParameter
from raptorbt.execution import PaperExecution, SimulatedExecution
from raptorbt.strategy import orders
from raptorbt.strategy.order_check import OrderProposal, OrderRefusal

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from test_execution import (  # noqa: E402
    MIN,
    Hooked,
    Venue,
    bar,
    make_bars,
    node_with,
    push_bars,
    trades_of,
)

SYMBOLS = ["AAA", "BBB"]


def config():
    return BacktestConfig(initial_capital=500_000.0, fees=0.0005, slippage=0.0003)


def three_ways(strategy_factory, symbols=SYMBOLS, data=None, **kw):
    """The same strategy through the batch runtime, an in-process node and a
    paper node. Returns the three ``(strategy, result)`` pairs."""
    data = data or make_bars(symbols)
    out = []
    s = strategy_factory()
    out.append((s, run_strategies([s], data, config(), bar_stamp="close", **kw)[0]))
    for execution in (None, PaperExecution()):
        s = strategy_factory()
        node = LiveNode([s], symbols, config(), bar_stamp="close", execution=execution, **kw)
        push_bars(node, data, symbols)
        out.append((s, node.finish()[0]))
    return out


# -- order tags --------------------------------------------------------------


def test_order_tags_reach_the_venue_request_with_the_engines_own_client_id():
    venue = Venue()

    def script(s, ctx, n):
        if n == 1:
            s.submit_order(
                orders.Limit(side="buy", price=99.0, units=7.0, tags={"basket": "b-1", "leg": "2"})
            )
        if n == 2:
            s.submit_order(orders.Limit(side="buy", price=98.0, units=1.0), client_id="plain")

    node = node_with(venue, [Hooked(script)])
    bar(node, "A", 0)
    bar(node, "A", 1)
    first, second = venue.requests
    assert first.tags == {"client_id": "O-0", "basket": "b-1", "leg": "2"}
    assert second.tags == {"client_id": "plain"}


@pytest.mark.parametrize(
    "tags, why",
    [
        ({"client_id": "x"}, "reserved"),
        ({str(i): "v" for i in range(17)}, "at most"),
        ({"k": 1}, "must be str"),
        ({"": "v"}, "bytes"),
        ({"k": "v" * 257}, "bytes"),
    ],
)
def test_order_tags_are_bounded_and_string_only(tags, why):
    class Bad(Strategy):
        def on_bar(self, ctx, bar):
            self.submit_order(orders.Market(side="buy", units=1.0, tags=tags))

    data = make_bars(["AAA"], n=3)
    with pytest.raises(Exception) as e:
        run_strategies([Bad()], data, config(), bar_stamp="close")
    assert e.value.code == "invalid_order_tags"
    assert why in str(e.value)


def test_tagged_orders_run_the_same_in_every_way():
    class Tagged(Strategy):
        def on_bar(self, ctx, bar):
            if ctx.idx == 3:
                self.submit_order(orders.Market(side="buy", units=5.0, tags={"a": "b"}))
            if ctx.idx == 9:
                self.submit_order(orders.Market(side="sell", units=5.0, reduce_only=True))

    a, b, c = (trades_of(r) for _, r in three_ways(Tagged))
    assert a and a == b == c


# -- the host's order check --------------------------------------------------


class Sizer(Strategy):
    """Small and large market orders on a rhythm."""

    def __init__(self):
        super().__init__()
        self.rejected = []
        self.n = {}

    def on_bar(self, ctx, bar):
        n = self.n.get(ctx.symbol, 0)
        self.n[ctx.symbol] = n + 1
        if n % 10 == 2:
            self.submit_order(orders.Market(side="buy", units=3.0), client_id=f"s-{ctx.symbol}-{n}")
        if n % 10 == 4:
            self.submit_order(orders.Market(side="buy", units=40.0), client_id=f"b-{ctx.symbol}-{n}")
        if n % 10 == 7:
            self.submit_order(
                orders.Market(side="sell", units=3.0, reduce_only=True), client_id=f"x-{ctx.symbol}-{n}"
            )

    def on_order_rejected(self, ctx, event):
        self.rejected.append((event.client_order_id, event.reject_reason))


def lot_check(order, ctx):
    assert isinstance(order, OrderProposal)
    assert ctx.symbol == order.symbol
    if order.units is not None and order.units > 10 and not order.reduce_only:
        return "lot_limit"
    return None


def test_one_order_check_refuses_the_same_orders_in_every_way():
    ways = three_ways(Sizer, order_check=lot_check)
    refused = [sorted(s.rejected) for s, _ in ways]
    assert refused[0], "the check refused nothing"
    assert all(code == "lot_limit" for _, code in refused[0])
    assert all(cid.startswith("b-") for cid, _ in refused[0])
    assert refused[0] == refused[1] == refused[2]
    a, b, c = (trades_of(r) for _, r in ways)
    assert a and a == b == c
    # A refused order never traded.
    for _, result in ways:
        assert all(t.size == 3.0 for t in result.result.trades())


def test_a_refusal_never_reaches_the_venue_and_carries_the_hosts_code():
    venue = Venue()
    seen = []

    def script(s, ctx, n):
        if n == 1:
            s.submit_order(orders.Limit(side="buy", price=99.0, units=7.0, tags={"t": "1"}), client_id="no")
        if n == 2:
            s.submit_order(orders.Limit(side="buy", price=98.0, units=1.0), client_id="yes")

    def check(order, ctx):
        seen.append(order)
        return OrderRefusal("term_rule_7") if order.client_id == "no" else None

    class Reasons(Hooked):
        reasons = []

        def on_order_rejected(self, ctx, event):
            self.reasons.append((event.client_order_id, event.reject_reason))

    Reasons.reasons = []
    node = node_with(venue, [Reasons(script)], order_check=check)
    bar(node, "A", 0)
    bar(node, "A", 1)
    assert [r.tags["client_id"] for r in venue.requests] == ["yes"]
    strategy = node.strategies[0]
    rejects = [e for e in strategy.seen if e[0] == "order_rejected"]
    assert [e[1] for e in rejects] == ["no"]
    assert strategy.reasons == [("no", "term_rule_7")]
    assert seen[0].tags == {"t": "1"} and seen[0].limit_price == 99.0 and seen[0].strategy_index == 0


def test_the_check_must_answer_with_a_code_and_its_errors_stop_the_run():
    class One(Strategy):
        def on_bar(self, ctx, bar):
            self.submit_order(orders.Market(side="buy", units=1.0))

    data = make_bars(["AAA"], n=3)
    with pytest.raises(Exception) as e:
        run_strategies([One()], data, config(), bar_stamp="close", order_check=lambda o, c: 42)
    assert e.value.code == "invalid_order_refusal"

    def boom(o, c):
        raise KeyError("host bug")

    with pytest.raises(KeyError):
        run_strategies([One()], data, config(), bar_stamp="close", order_check=boom)
    with pytest.raises(InvalidParameter) as e:
        run_strategies([One()], data, config(), bar_stamp="close", order_check="nope")
    assert e.value.code == "invalid_order_check"


def test_intents_are_refused_when_a_check_is_installed():
    class Legacy(Strategy):
        def on_bar(self, ctx, bar):
            if ctx.idx == 2:
                self.enter()

    data = make_bars(["AAA"], n=5)
    with pytest.raises(Exception) as e:
        run_strategies([Legacy()], data, config(), bar_stamp="close", order_check=lambda o, c: None)
    assert e.value.code == "order_check_intent"


# -- muted warm-up -----------------------------------------------------------


class WarmTrader(Strategy):
    def __init__(self):
        super().__init__()
        self.flags = []
        self.sma_n = 0

    def on_bar(self, ctx, bar):
        self.flags.append(ctx.is_warmup)
        self.sma_n += 1
        self.submit_order(orders.Market(side="buy", units=1.0), client_id=f"w-{self.sma_n}")


def _warm(n=20, symbols=("AAA",)):
    data = make_bars(list(symbols), n=n + 5)
    cut = n
    hist = {s: {k: v[:cut] for k, v in a.items()} for s, a in data.items()}
    live = {s: {k: v[cut:] for k, v in a.items()} for s, a in data.items()}
    return hist, live


@pytest.mark.parametrize("execution", [None, PaperExecution()])
def test_muted_warmup_primes_state_but_trades_nothing(execution):
    hist, live = _warm()
    s = WarmTrader()
    node = LiveNode(
        [s], ["AAA"], config(), bar_stamp="close", execution=execution,
        warmup_bars=hist, warmup="muted",
    )
    assert s.sma_n == 20 and all(s.flags)
    ctx = node.ctx(0)
    assert ctx.is_warmup is False
    assert node.positions(0, "AAA") == []
    push_bars(node, live, ["AAA"])
    assert s.flags[20:] == [False] * 5
    result = node.finish()[0]
    assert all(t.entry_time > int(hist["AAA"]["timestamps"][-1]) for t in result.result.trades())
    assert [o.client_id for o in result.result.orders()][0] == "w-21"


def test_live_warmup_still_trades_as_before_and_muted_is_the_default_for_your_own_venue():
    hist, _ = _warm()
    s = WarmTrader()
    node = LiveNode([s], ["AAA"], config(), bar_stamp="close", warmup_bars=hist)
    assert node.positions(0, "AAA")  # in-process default: warm-up trades
    node.finish()
    venue = Venue()
    s = WarmTrader()
    node = LiveNode([s], ["AAA"], config(), bar_stamp="close", execution=venue, warmup_bars=hist)
    assert s.sma_n == 20 and venue.requests == []
    node.finish()


def test_history_pushed_after_construction_warms_the_node_muted():
    hist, live = _warm()
    s = WarmTrader()
    node = LiveNode([s], ["AAA"], config(), bar_stamp="close")
    node.warm_up({"AAA": hist["AAA"]})
    assert s.sma_n == 20 and all(s.flags) and node.positions(0, "AAA") == []
    push_bars(node, live, ["AAA"])
    assert s.flags[20:] == [False] * 5
    node.finish()
    with pytest.raises(InvalidParameter):
        LiveNode([WarmTrader()], ["AAA"], warmup="warm")


# -- signed seeds on margin accounts -----------------------------------------


def _flat_bars(n=6, price=100.0, symbol="AAA"):
    ts = (np.arange(n, dtype=np.int64) + 1) * MIN
    closes = np.array([price - i for i in range(n)], dtype=np.float64)  # a falling market
    return {
        symbol: {
            "timestamps": ts, "open": closes, "high": closes + 0.5, "low": closes - 0.5,
            "close": closes, "volume": np.full(n, 1000.0),
        }
    }


class Idle(Strategy):
    def on_bar(self, ctx, bar):
        pass


def _seeded(seed, account_type="margin", leverage=1.0, execution=None, **kw):
    data = _flat_bars()
    node = LiveNode(
        [Idle()], ["AAA"],
        BacktestConfig(initial_capital=100_000.0, fees=0.0, slippage=0.0),
        bar_stamp="close", account_type=account_type, leverage=leverage,
        initial_positions={0: {"AAA": seed}}, execution=execution, **kw,
    )
    push_bars(node, data, ["AAA"])
    return node


@pytest.mark.parametrize("execution", [None, PaperExecution()])
def test_a_short_seed_gains_as_the_price_falls_and_costs_nothing_to_adopt(execution):
    node = _seeded({"quantity": -100.0, "avg_price": 100.0}, execution=execution)
    [pos] = node.positions(0, "AAA")
    assert pos.size == 100.0 and pos.direction == -1
    # Falling 100, 99, ..., 95: the short is up (100 - 95) * 100 = 500.
    assert node.equity(0) == pytest.approx(100_000.0 + 500.0)
    result = node.finish()[0]
    assert len(result.result.trades()) == 1  # closed at the end of the run
    assert result.result.orders() == []  # adopting made no order


def test_a_posted_margin_is_what_is_locked_on_a_leveraged_account():
    base = {"quantity": -100.0, "avg_price": 100.0}
    with pytest.raises(ValueError, match="fully funded|cash or fully funded"):
        _seeded(base, leverage=5.0)
    node = _seeded({**base, "margin_minor": 2_500_000}, leverage=5.0)  # 25,000.00 posted
    ctx = node.ctx(0)
    # Equity is balance plus the short's gain; free capital is the rest of the balance.
    assert node.equity(0) == pytest.approx(100_000.0 + 500.0)
    assert ctx.free_capital == pytest.approx(75_000.0)
    node.finish()


def test_a_cash_account_refuses_a_short_and_a_cash_basis_is_exact():
    with pytest.raises(ValueError, match="cash account"):
        _seeded({"quantity": -10.0, "avg_price": 100.0}, account_type="cash")
    with pytest.raises(ValueError, match="margin accounts only"):
        _seeded({"quantity": 10.0, "avg_price": 100.0, "margin_minor": 5}, account_type="cash")
    node = _seeded({"quantity": 3.0, "avg_price": 33.333333, "cost_minor": 10_000}, account_type="cash")
    # The broker's own figure (100.00) was debited, not 3 * 33.333333 rounded.
    assert node.ctx(0).cash == pytest.approx(99_900.0)
    node.finish()


def test_seeds_are_validated():
    for bad in (
        {"quantity": 0.0, "avg_price": 100.0},
        {"quantity": 5.0, "avg_price": 0.0},
        {"quantity": 5.0, "avg_price": 100.0, "margin_minor": 1.5},
        {"quantity": 5.0, "avg_price": 100.0, "margin_minor": -1},
        {"quantity": 5.0, "avg_price": 100.0, "cost_minor": True},
    ):
        with pytest.raises(InvalidParameter) as e:
            _seeded(bad)
        assert e.value.code == "invalid_initial_positions"


# -- instruments, chains and roles from the context ----------------------------

import raptorbt  # noqa: E402
from raptorbt.errors import UnknownSymbol  # noqa: E402
from raptorbt.strategy import Chain, Future, Holding, Requirements, StrikeWindow, Underlying  # noqa: E402
from raptorbt.structures import Expiry  # noqa: E402

DAY = 86_400 * 10**9
EXP1 = T0_EXP = (1_700_000_000 + 10 * 86_400) * 10**9
EXP2 = EXP1 + 7 * DAY


def _universe():
    """A spot index, two futures, and calls and puts at three strikes over two expiries,
    the second expiry listed only from bar 10."""
    t0 = (1_700_000_000) * 10**9
    listed_late = t0 + 10 * MIN
    specs = {"IDX": raptorbt.InstrumentSpec.equity("IDX")}
    specs["FUT1"] = raptorbt.InstrumentSpec.futures_contract("FUT1", EXP1, 50.0, underlying="IDX")
    specs["FUT2"] = raptorbt.InstrumentSpec.futures_contract("FUT2", EXP2, 50.0, underlying="IDX")
    for expiry, activation, tag in ((EXP1, None, "A"), (EXP2, listed_late, "B")):
        for k in (95.0, 100.0, 105.0):
            for right in ("call", "put"):
                sym = f"IDX-{tag}-{k:g}-{right[0]}"
                specs[sym] = raptorbt.InstrumentSpec.option(
                    sym, k, right, expiry, 50.0, underlying="IDX", activation_ns=activation
                )
    return specs


class Asker(Strategy):
    requirements = Requirements(
        roles={
            "index": Underlying(asset="index"),
            "hedge": Future(of="index", contract="front"),
            "options": Chain(of="index", expiry=Expiry.nearest(), strikes=StrikeWindow.around(1)),
        },
        timeframes=["1m"],
        holding=Holding.overnight(),
        data=["bar"],
    )

    def __init__(self):
        super().__init__()
        self.rows = []

    def on_bar(self, ctx, bar):
        if ctx.symbol != "IDX":
            return
        chain = ctx.option_chain()
        now = ctx.event_ns
        self.rows.append(
            (
                ctx.instrument().symbol,
                ctx.instrument("index").symbol,
                ctx.role("index"),
                ctx.role("hedge"),
                ctx.role("options"),
                len(chain),
                tuple(chain.expiries("IDX", now)),
                ctx.instrument(ctx.role("hedge")).expiration_ns,
            )
        )


def _asker_ways(n=14):
    specs = _universe()
    t0 = 1_700_000_000 * 10**9
    ts = t0 + (np.arange(n, dtype=np.int64) + 1) * MIN
    px = np.full(n, 100.0)
    data = {"IDX": {"timestamps": ts, "open": px, "high": px + 1, "low": px - 1, "close": px, "volume": np.full(n, 10.0)}}
    kw = dict(instruments=specs, bindings={"index": "IDX"})
    out = []
    s = Asker()
    run_strategies([s], data, config(), bar_stamp="close", **kw)
    out.append(s.rows)
    for execution in (None, PaperExecution()):
        s = Asker()
        node = LiveNode([s], ["IDX"], config(), bar_stamp="close", execution=execution, **kw)
        push_bars(node, data, ["IDX"])
        node.finish()
        out.append(s.rows)
    return out


def test_the_context_answers_identically_in_a_batch_and_on_both_nodes():
    batch, live, paper = _asker_ways()
    assert len(batch) == 14 and batch == live == paper
    first, last = batch[0], batch[-1]
    # Roles: the bound underlying, the front future, the nearest expiry's calls and puts
    # one strike either side of the money.
    assert first[:3] == ("IDX", "IDX", "IDX")
    assert first[3] == "FUT1"
    assert first[4] == tuple(f"IDX-A-{k:g}-{r}" for k in (95.0, 100.0, 105.0) for r in ("c", "p"))
    assert first[7] == EXP1


def test_a_chain_holds_only_the_contracts_listed_at_the_context_time():
    batch = _asker_ways()[0]
    # Bars 1-9: only the first expiry (6 contracts) is listed; from bar 10 the second joins.
    sizes = [row[5] for row in batch]
    assert sizes[:9] == [6] * 9 and sizes[9:] == [12] * 5
    assert [len(row[6]) for row in batch][:9] == [1] * 9
    assert [len(row[6]) for row in batch][9:] == [2] * 5


def test_unknown_things_raise_with_a_code():
    class Bad(Strategy):
        def on_bar(self, ctx, bar):
            self.what(ctx)

    data = make_bars(["AAA"], n=2)

    def run(what, **kw):
        s = Bad()
        s.what = what
        run_strategies([s], data, config(), bar_stamp="close", **kw)

    with pytest.raises(UnknownSymbol) as e:
        run(lambda c: c.instrument("AAA"))
    assert e.value.code == "no_instrument_spec"
    with pytest.raises(InvalidParameter) as e:
        run(lambda c: c.role("index"))
    assert e.value.code == "no_requirements"
    with pytest.raises(InvalidParameter) as e:
        run(lambda c: None, bindings={"index": "AAA"})
    assert e.value.code == "unknown_role_binding"
    with pytest.raises(UnknownSymbol) as e:
        run(lambda c: None, instruments={"AAA": raptorbt.InstrumentSpec.equity("AAA")}, bindings={})
        run(lambda c: c.instrument("ZZZ"), instruments={"AAA": raptorbt.InstrumentSpec.equity("AAA")})
    assert e.value.code == "no_instrument_spec"


# -- percent exits ---------------------------------------------------------------

from raptorbt.errors import InvalidParameter as _IP  # noqa: E402
from raptorbt.execution import OrderRefused, Unsupported  # noqa: E402
from raptorbt.strategy.exits import ExitEvents, ExitRule, ExitRuleRequest  # noqa: E402


def _path_bars(closes_by_symbol, start=1):
    n = len(next(iter(closes_by_symbol.values())))
    ts = (np.arange(n, dtype=np.int64) + start) * MIN
    out = {}
    for s, closes in closes_by_symbol.items():
        c = np.array(closes, dtype=np.float64)
        out[s] = {"timestamps": ts, "open": c, "high": c + 0.1, "low": c - 0.1, "close": c, "volume": np.full(n, 1000.0)}
    return out


class Holder(ExitEvents, Strategy):
    """Buys 10 of each symbol on the first bar and arms the rule it is given."""

    def __init__(self, arm):
        super().__init__()
        self.arm, self.handles, self.n, self.seen = arm, [], 0, []

    def on_bar(self, ctx, bar):
        if ctx.symbol != ctx.symbols[0]:
            return
        self.n += 1
        if self.n == 1:
            for s in ctx.symbols:
                self.submit_order(orders.Market(side="buy", units=10.0, instrument=s), client_id=f"in-{s}")
        if self.n == 2:
            self.handles = self.arm(self, ctx)

    def on_order_event(self, ctx, event):
        self.seen.append((event.kind, event.client_order_id))


def _exit_ways(arm, closes, symbols):
    data = _path_bars(closes)
    out = []
    s = Holder(arm)
    run_strategies([s], data, BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close")
    out.append(s)
    for execution in (None, PaperExecution()):
        s = Holder(arm)
        node = LiveNode([s], symbols, BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close", execution=execution)
        push_bars(node, data, symbols)
        node.finish()
        out.append(s)
    return out


def test_a_position_stop_in_percent_closes_with_a_reduce_only_order_everywhere():
    # Entry ~100 on bar 2's fill; a 5% stop on a 10-lot costing ~1000 fires at -50.
    closes = {"AAA": [100, 100, 100, 99, 97, 94, 94, 94]}
    ways = _exit_ways(lambda s, ctx: [s.set_exit(ctx, ExitRule(stop_loss_pct=5), symbol="AAA")], closes, ["AAA"])
    for s in ways:
        [h] = s.handles
        assert h.state in ("fired", "done") and h.reason == "stop_loss_pct"
        assert h.close_ids == [f"{h.id}:x0"]
        assert ("order_filled", f"{h.id}:x0") in s.seen
    assert len({tuple(s.seen) for s in ways}) == 1


def test_a_group_take_profit_in_currency_closes_every_leg_together():
    closes = {"AAA": [100, 100, 100, 101, 102, 103, 103, 103], "BBB": [50, 50, 50, 50, 51, 52, 52, 52]}
    arm = lambda s, ctx: [s.set_exit(ctx, ExitRule(take_profit=25.0), symbols=["AAA", "BBB"], name="pair")]
    ways = _exit_ways(arm, closes, ["AAA", "BBB"])
    for s in ways:
        [h] = s.handles
        assert h.reason == "take_profit" and len(h.close_ids) == 2
    assert len({tuple(s.seen) for s in ways}) == 1


def test_a_rule_that_never_holds_stays_armed_and_a_canceled_one_does_not_fire():
    closes = {"AAA": [100] * 8}
    ways = _exit_ways(lambda s, ctx: [s.set_exit(ctx, ExitRule(stop_loss_pct=50), symbol="AAA")], closes, ["AAA"])
    assert all(s.handles[0].state == "armed" for s in ways)

    def arm(s, ctx):
        h = s.set_exit(ctx, ExitRule(take_profit_pct=1), symbol="AAA")
        s.cancel_exit(h)
        return [h]

    ways = _exit_ways(arm, {"AAA": [100, 100, 100, 110, 120, 130, 130, 130]}, ["AAA"])
    assert all(s.handles[0].state == "canceled" and not s.handles[0].close_ids for s in ways)


def test_exit_rules_are_validated():
    for bad in ({}, {"stop_loss_pct": 0}, {"stop_loss_pct": 101}, {"take_profit": -1.0}, {"premium_at_or_above": float("nan")}):
        with pytest.raises(_IP) as e:
            ExitRule(**bad)
        assert e.value.code == "invalid_exit_rule"


class HoldingVenue(Venue):
    """A venue that holds exit rules (or refuses to)."""

    def __init__(self, refuse=None):
        super().__init__()
        self.rules, self.refuse_rules = [], refuse

    def place_exit_rule(self, request):
        if self.refuse_rules:
            raise self.refuse_rules
        self.rules.append(request)


def test_a_venue_that_holds_the_rule_takes_it_and_the_engine_stops_watching():
    venue = HoldingVenue()
    s = Hooked()
    node = node_with(venue, [s], symbols=("A",))
    bar(node, "A", 0)
    handle = s.set_exit(node.ctx(0), ExitRule(stop_loss_pct=10), symbol="A")
    assert handle.state == "venue" and len(venue.rules) == 1
    assert isinstance(venue.rules[0], ExitRuleRequest) and venue.rules[0].rule == {"stop_loss_pct": 10}


@pytest.mark.parametrize("why", [Unsupported("no"), OrderRefused("no")])
def test_a_venue_that_refuses_the_rule_leaves_it_with_the_engine(why):
    venue = HoldingVenue(refuse=why)
    s = Hooked()
    node = node_with(venue, [s], symbols=("A",))
    bar(node, "A", 0)
    handle = s.set_exit(node.ctx(0), ExitRule(take_profit=1.0), symbol="A")
    assert handle.state == "armed" and venue.rules == []


# -- continuous futures -----------------------------------------------------

from raptorbt.strategy.continuous import RollEvents, continuous_series, front  # noqa: E402

T0C = 1_700_000_000 * 10**9
EXPA, EXPB = T0C + 20 * MIN, T0C + 40 * MIN  # expiries in minutes; the roll rule is 5 minutes (5/1440 days)


def _fut_specs():
    return {
        "IDX": raptorbt.InstrumentSpec.equity("IDX"),
        "FA": raptorbt.InstrumentSpec.futures_contract("FA", EXPA, 1.0, underlying="IDX"),
        "FB": raptorbt.InstrumentSpec.futures_contract("FB", EXPB, 1.0, underlying="IDX"),
    }


def test_a_continuous_role_names_the_contract_that_is_front_under_its_roll_rule():
    specs = list(_fut_specs().values())
    assert front(specs, 0.0, EXPA - 1) == ("FA", EXPA)
    assert front(specs, 0.0, EXPA) == ("FB", EXPB)
    assert front(specs, 5 / 1440, EXPA - 6 * MIN) == ("FA", EXPA - 5 * MIN)  # roll_days is days: 5/1440 is 5 minutes
    assert front(specs, 0.0, EXPB) == (None, None)


def test_future_roll_days_resolves_and_round_trips():
    from raptorbt.strategy.requirements import Future as F, Requirements as R, resolve

    req = R(roles={"i": Underlying(asset="index"), "f": F(of="i", contract="continuous", roll_days=1)},
            timeframes=["1m"], holding=Holding.overnight())
    assert R.from_dict(req.to_dict()) == req
    with pytest.raises(Exception):
        F(of="i", contract="front", roll_days=1)
    day = 86_400 * 10**9
    a = raptorbt.InstrumentSpec.futures_contract("FA", 10 * day, 1.0, underlying="IDX")
    b = raptorbt.InstrumentSpec.futures_contract("FB", 40 * day, 1.0, underlying="IDX")
    sym = lambda now: resolve(req, now_ns=now, bindings={"i": "IDX"}, specs=[a, b]).roles["f"].symbols
    assert sym(8 * day) == ("FA",) and sym(9 * day) == ("FB",) and sym(9 * day + 1) == ("FB",)


class Carry(RollEvents, Strategy):
    requirements = Requirements(
        roles={"index": Underlying(asset="index"),
               "fut": Future(of="index", contract="continuous", roll_days=5 / 1440)},
        timeframes=["1m"], holding=Holding.overnight(), data=["bar"],
    )
    auto_roll = True

    def __init__(self):
        super().__init__()
        self.rolls, self.seen, self.n = [], [], 0

    def on_bar(self, ctx, bar):
        if ctx.symbol != "IDX":
            return
        self.n += 1
        if self.n == 1:
            self.submit_order(orders.Market(side="buy", units=3.0, instrument=ctx.role("fut")), client_id="open")

    def on_roll(self, ctx, roll):
        self.rolls.append(roll)

    def on_order_event(self, ctx, event):
        self.seen.append((event.kind, event.client_order_id))


def _carry_ways():
    specs = _fut_specs()
    n = 35  # ends before FB's expiry: no settlement
    data = {}
    for k, s in enumerate(("IDX", "FA", "FB")):
        ts = T0C + (np.arange(n, dtype=np.int64) + 1) * MIN
        c = np.full(n, 100.0 + 5 * k)
        data[s] = {"timestamps": ts, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": np.full(n, 100.0)}
    symbols = ["IDX", "FA", "FB"]
    cfg = lambda: BacktestConfig(initial_capital=1e6, fees=0.0)
    kw = dict(instruments=specs, bindings={"index": "IDX"})
    out = []
    s = Carry()
    run_strategies([s], data, cfg(), bar_stamp="close", **kw)
    out.append(s)
    for execution in (None, PaperExecution()):
        s = Carry()
        node = LiveNode([s], symbols, cfg(), bar_stamp="close", execution=execution, **kw)
        push_bars(node, data, symbols)
        node.finish()
        out.append(s)
    return out


def test_a_roll_is_an_event_and_auto_roll_moves_the_position_with_real_orders():
    # The rule rolls 5 minutes before FA's expiry (EXPA = bar 20 + T0C), i.e. at bar 15.
    ways = _carry_ways()
    for s in ways:
        assert [(r.role, r.from_symbol, r.to_symbol) for r in s.rolls] == [("fut", "FA", "FB")]
        assert s.rolls[0].ts_ns == EXPA - 5 * MIN
        assert ("order_filled", "fut:roll0:c0") in s.seen and ("order_filled", "fut:roll0:o0") in s.seen
    assert len({tuple(s.seen) for s in ways}) == 1
    assert len({tuple(s.rolls) for s in ways}) == 1


def test_continuous_series_adjusts_history_under_an_explicit_policy():
    specs = _fut_specs()
    day = 86_400 * 10**9
    # Contract A trades 100.., B 110.. ; the roll falls at EXPA.
    ts = T0C + (np.arange(30, dtype=np.int64) + 1) * MIN
    a = {"timestamps": ts, "open": np.full(30, 100.0), "high": np.full(30, 101.0), "low": np.full(30, 99.0), "close": np.full(30, 100.0), "volume": np.full(30, 5.0)}
    b = {k: (v + 10.0 if k in ("open", "high", "low", "close") else v) for k, v in a.items()}
    raw = continuous_series({"FA": a, "FB": b}, specs, roll_days=0.0, adjust="raw")
    assert raw.contract[:19] == ["FA"] * 19 and raw.contract[19:] == ["FB"] * 11
    assert raw.bars["close"][18] == 100.0 and raw.bars["close"][19] == 110.0
    assert raw.rolls == [(int(ts[19]), "FA", "FB", 10.0)]
    back = continuous_series({"FA": a, "FB": b}, specs, roll_days=0.0, adjust="back")
    assert set(back.bars["close"]) == {110.0}, "earlier bars shifted onto the latest basis"
    ratio = continuous_series({"FA": a, "FB": b}, specs, roll_days=0.0, adjust="ratio")
    assert set(np.round(ratio.bars["close"], 9)) == {110.0}
    assert back.bars["volume"].tolist() == raw.bars["volume"].tolist()
    with pytest.raises(InvalidParameter) as e:
        continuous_series({"FA": a, "FB": b}, specs, roll_days=0.0, adjust="none")
    assert e.value.code == "invalid_adjust"


# -- journal: a refused order and an exit's orders are recorded -----------------------


def test_an_order_check_refusal_and_an_exits_orders_are_journaled():
    from raptorbt import journal as rj

    j = rj.Journal({"run": "host"}, memory=True)

    class Mixed(ExitEvents, Strategy):
        def __init__(self):
            super().__init__()
            self.n = 0

        def on_bar(self, ctx, bar):
            self.n += 1
            if self.n == 1:
                self.submit_order(orders.Market(side="buy", units=999.0), client_id="big")
                self.submit_order(orders.Market(side="buy", units=5.0, tags={"k": "v"}), client_id="in")
            if self.n == 2:
                self.set_exit(ctx, ExitRule(stop_loss_pct=1), symbol="A")

    check = lambda o, c: "lot_limit" if o.units and o.units > 100 else None
    node = LiveNode(
        [Mixed()], ["A"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close",
        execution=PaperExecution(), journal=j, order_check=check,
    )
    for i, px in enumerate([100, 100, 100, 90, 90]):
        node.push_bar("A", (i + 1) * MIN, px, px + 1, px - 1, px, 10.0)
    node.finish()
    records = list(j.since(0))
    refusals = [r for r in records if r.kind == "refusal"]
    assert [(r.client_order_id, r.payload["code"]) for r in refusals] == [("big", "lot_limit")]
    cmds = [c for r in records if r.kind == "command" for c in r.payload["commands"]]
    submitted = {c["client_id"]: c for c in cmds if c["command"] == "submit"}
    assert submitted["in"]["order"]["tags"] == {"k": "v"}
    exit_ids = [cid for cid in submitted if ":x" in cid]
    assert exit_ids and submitted[exit_ids[0]]["order"]["tags"]["close"] == "stop_loss_pct"
    assert submitted[exit_ids[0]]["order"]["reduce_only"] is True
