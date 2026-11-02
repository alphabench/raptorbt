"""Contract expiry, settlement and corporate actions, in every runner.

Pins: the settlement price is declared or stated (never silently guessed),
physical delivery lands on the underlying, settlement charges come from the
fee schedule's settlement row, a fill on day D is charged by the row in force
on D, a split never looks like a crash under ``adjusted`` and never moves
equity under ``raw``, and dividends are credited as declared.
"""

import numpy as np
import pytest

import raptorbt
from raptorbt import (
    BacktestConfig,
    CorporateAction,
    FeeSchedule,
    Indicator,
    InstrumentSpec,
    RateCard,
    Strategy,
    adjust_history,
    run_portfolio_strategy,
    run_strategies,
    run_strategy_backtest,
)
from raptorbt.errors import InvalidParameter, UnknownSymbol
from raptorbt.strategy import orders

DAY = 86_400_000_000_000


def bars(closes, start=0, step=1, volume=1_000.0):
    c = np.asarray(closes, dtype=np.float64)
    n = len(c)
    return {
        "timestamps": np.arange(start, start + n * step, step, dtype=np.int64),
        "open": c.copy(),
        "high": c + 1.0,
        "low": c - 1.0,
        "close": c,
        "volume": np.full(n, volume),
    }


def config(**kw):
    cfg = BacktestConfig(**kw)
    cfg.fees = 0.0
    return cfg


class Holder(Strategy):
    """Enters ``sym`` once (``units`` if given) and records lifecycle events."""

    sym = None
    units = None
    side = "buy"

    def __init__(self):
        super().__init__()
        self.lifecycle = []
        self.positions = []
        self.rejections = []
        self.done = False

    def on_bar(self, ctx):
        if self.done or getattr(ctx, "symbol", self.sym) != self.sym:
            return
        self.done = True
        if self.units is None:
            self.enter() if self.side == "buy" else self.enter_short()
        else:
            self.submit_order(orders.Market(side=self.side, units=self.units, instrument=self.sym))

    def on_contract_event(self, ctx, e):
        self.lifecycle.append(e)

    def on_corporate_action(self, ctx, e):
        self.lifecycle.append(e)

    def on_position_event(self, ctx, e):
        self.positions.append(e.kind)

    def on_order_rejected(self, ctx, e):
        self.rejections.append(e.reject_reason)


def holder_cls(sym, units=None, side="buy"):
    return type("Held", (Holder,), {"sym": sym, "units": units, "side": side})


def holder(sym, units=None, side="buy"):
    return holder_cls(sym, units, side)()


def run(strategy, data, symbols=None, **kw):
    kw.setdefault("bar_stamp", "close")
    kw.setdefault("config", config())
    out = run_strategies([(strategy, symbols or list(data))], data, **kw)
    return out[0]


def trades(result):
    return result.result.trades()


# -- the settlement price is stated, never guessed -----------------------------


class TestSettlementPrice:
    def test_last_price_is_stated_when_the_contract_has_no_bar_at_expiry(self):
        fut = InstrumentSpec.futures_contract("FUT", expiration_ns=5, lot_size=1.0, underlying="U")
        s = holder("FUT")
        r = run(
            s,
            {"FUT": bars([100, 101, 102]), "U": bars([50, 51, 52, 53, 54, 55, 56, 57])},
            instruments={"FUT": fut},
        )
        (t,) = trades(r)
        assert (t.exit_reason, t.exit_price, t.exit_time) == ("Settlement", 102.0, 5)
        (e,) = s.lifecycle
        assert (e.kind, e.outcome, e.style, e.source) == (
            "contract_settled",
            "settled",
            "cash",
            "last_price",
        )
        assert e.price == 102.0

    def test_a_declared_price_settles_the_contract_and_says_so(self):
        fut = InstrumentSpec.futures_contract("FUT", expiration_ns=2, lot_size=1.0)
        fut.settlement_price = 105.0
        assert fut.settlement_price == 105.0
        s = holder("FUT")
        r = run_strategy_backtest(
            holder_cls('FUT'), **bars([100, 101, 102, 103]), config=config(), instrument=fut
        )
        (t,) = r.trades()
        assert t.exit_price == 105.0 and t.exit_reason == "Settlement"

    def test_an_option_settles_to_intrinsic_against_a_registered_underlying(self):
        opt = InstrumentSpec.option(
            "OPT", strike=100.0, right="call", expiration_ns=4, lot_size=1.0, underlying="U"
        )
        opt.settlement_price = "underlying"
        s = holder("OPT")
        r = run(
            s,
            {"OPT": bars([7, 8]), "U": bars([105, 108, 110, 111, 112, 113])},
            instruments={"OPT": opt},
        )
        (t,) = trades(r)
        # U prints 112 at the expiry instant (ts 4): the contract waits for that
        # print and settles on it, intrinsic 12.
        assert t.exit_price == 12.0
        (e,) = s.lifecycle
        assert (e.outcome, e.source) == ("settled", "underlying")

    def test_a_missing_underlying_price_refuses_the_settlement_and_keeps_the_position(self):
        opt = InstrumentSpec.option(
            "OPT", strike=100.0, right="call", expiration_ns=2, lot_size=1.0, underlying="NOPE"
        )
        opt.settlement_price = "underlying"
        s = holder("OPT")
        r = run_strategy_backtest(
            holder_cls('FUT'), **bars([7, 8, 9, 10]), config=config(), instrument=opt
        )
        (t,) = r.trades()
        # the position was not closed at a guess: the run's own end closed it
        assert t.exit_reason == "EndOfData"

    def test_the_refusal_is_reported_once_with_a_stable_reason(self):
        opt = InstrumentSpec.option(
            "OPT", strike=100.0, right="call", expiration_ns=2, lot_size=1.0, underlying="NOPE"
        )
        opt.settlement_price = "underlying"
        s = holder("OPT")
        run(s, {"OPT": bars([7, 8, 9, 10])}, instruments={"OPT": opt})
        refused = [e for e in s.lifecycle if e.kind == "lifecycle_refused"]
        assert len(refused) == 1
        assert refused[0].reject_reason == "underlying_price_missing"
        assert refused[0].outcome == "settlement"

    def test_settlement_terms_validate(self):
        spec = InstrumentSpec.futures_contract("F", expiration_ns=2, lot_size=1.0)
        with pytest.raises(InvalidParameter):
            spec.settlement_style = "swap"
        with pytest.raises(InvalidParameter):
            spec.settlement_price = "mid"
        with pytest.raises(InvalidParameter):
            spec.settlement_price = -1.0
        assert spec.settlement_style == "cash" and spec.settlement_price == "auto"


class TestOrdersOnAnExpiredContract:
    def test_an_order_is_rejected_with_a_stable_reason(self):
        fut = InstrumentSpec.futures_contract("FUT", expiration_ns=2, lot_size=1.0)

        class Late(Holder):
            def on_bar(self, ctx):
                if ctx.idx == 3:
                    self.submit_order(orders.Market(side="buy", units=1.0))

        s = Late()
        s.sym = "FUT"
        run_strategy_backtest(holder_cls('FUT'), **bars([100, 101, 102, 103, 104]), config=config(),
                              instrument=fut)
        s2 = Late()
        s2.sym = "FUT"
        run(s2, {"FUT": bars([100, 101, 102, 103, 104])}, instruments={"FUT": fut})
        assert s2.rejections == ["expired"]


# -- physical settlement --------------------------------------------------------


class TestPhysicalSettlement:
    def run_option(self, right, spot, side="buy", strike=100.0):
        opt = InstrumentSpec.option(
            "OPT", strike=strike, right=right, expiration_ns=4, lot_size=1.0,
            multiplier=10.0, underlying="U",
        )
        opt.settlement_style = "physical"
        s = holder("OPT", units=2.0, side=side)
        # the option trades at ts 0..1 only; the underlying carries the clock
        r = run(
            s,
            {"OPT": bars([5, 5]), "U": bars([100, 100, 100, spot, spot, spot])},
            instruments={"OPT": opt, "U": InstrumentSpec.equity("U")},
        )
        return s, r

    def test_a_long_call_in_the_money_is_exercised_and_delivers_at_the_strike(self):
        s, r = self.run_option("call", 110.0)
        opt_trade = next(t for t in trades(r) if t.symbol == "OPT")
        # settles at nothing: the 2 contracts cost 2 * 5 * 10 = 100
        assert opt_trade.exit_price == 0.0 and opt_trade.pnl == -100.0
        delivered = [t for t in trades(r) if t.symbol == "U"]
        assert len(delivered) == 1
        # taken in at the strike: 2 contracts * multiplier 10 = 20 units at 100,
        # closed at the end of the run at the last mark (110)
        assert delivered[0].entry_price == 100.0 and delivered[0].size == 20.0
        (e,) = [x for x in s.lifecycle if x.kind == "contract_settled"]
        assert (e.outcome, e.style, e.quantity, e.amount) == ("exercised", "physical", 20.0, 100.0)

    def test_a_short_put_in_the_money_is_assigned_and_takes_the_underlying_in(self):
        s, r = self.run_option("put", 90.0, side="sell")
        (e,) = [x for x in s.lifecycle if x.kind == "contract_settled"]
        # a short put is obliged to buy: +20 units at the strike
        assert (e.outcome, e.quantity, e.amount) == ("assigned", 20.0, 100.0)
        delivered = [t for t in trades(r) if t.symbol == "U"]
        assert delivered[0].direction == 1 and delivered[0].entry_price == 100.0

    def test_out_of_the_money_expires_without_delivery(self):
        s, r = self.run_option("call", 90.0)
        (e,) = [x for x in s.lifecycle if x.kind == "contract_settled"]
        assert (e.outcome, e.quantity) == ("expired", 0.0)
        assert [t for t in trades(r) if t.symbol == "U"] == []

    def test_physical_settlement_without_the_underlying_registered_is_refused(self):
        opt = InstrumentSpec.option(
            "OPT", strike=100.0, right="call", expiration_ns=2, lot_size=1.0, underlying="U"
        )
        opt.settlement_style = "physical"
        s = holder("OPT")
        run(s, {"OPT": bars([5, 5, 5, 5])}, instruments={"OPT": opt})
        refused = [e for e in s.lifecycle if e.kind == "lifecycle_refused"]
        assert refused and refused[0].reject_reason == "physical_settlement_needs_underlying_price"

    def test_a_physical_future_delivers_its_underlying_at_the_settlement_price(self):
        fut = InstrumentSpec.futures_contract(
            "FUT", expiration_ns=3, lot_size=1.0, multiplier=2.0, underlying="U"
        )
        fut.settlement_style = "physical"
        fut.settlement_price = 120.0
        s = holder("FUT", units=3.0)
        r = run(
            s,
            {"FUT": bars([100, 101]), "U": bars([100, 100, 100, 100, 100])},
            instruments={"FUT": fut, "U": InstrumentSpec.equity("U")},
        )
        (e,) = [x for x in s.lifecycle if x.kind == "contract_settled"]
        assert (e.outcome, e.price, e.quantity, e.amount) == ("delivered", 120.0, 6.0, 120.0)
        assert [t.size for t in trades(r) if t.symbol == "U"] == [6.0]


# -- charges ---------------------------------------------------------------------


def priced(schedule, segment):
    cfg = config()
    cfg.fee_segment = segment
    cfg.fee_schedule = schedule
    return cfg


def card(effective_from, brokerage, settle=None, name="c"):
    rates = [
        {
            "exchange": "NFO",
            "segment": "FUTURES",
            "underlying": None,
            "brokerage": {"per_order": brokerage},
            "transaction_tax": {"kind": "STT", "percent": "0", "sides": ["SELL"]},
            "exchange_turnover_percent": "0",
            "sebi_percent": "0",
            "stamp_duty_percent": "0",
            "gst_percent": "0",
        }
    ]
    if settle is not None:
        rates.append(
            {
                "exchange": "NFO",
                "segment": "FUTURES",
                "event": "SETTLEMENT",
                "underlying": None,
                "brokerage": {"per_order": "0"},
                "transaction_tax": {"kind": "STT", "percent": settle, "sides": ["SELL"]},
                "exchange_turnover_percent": "0",
                "sebi_percent": "0",
                "stamp_duty_percent": "0",
                "gst_percent": "0",
            }
        )
    return RateCard.from_dict(
        {"name": name, "effective_from": effective_from, "source": "test", "rates": rates}
    )


class TestCharges:
    def test_a_settlement_is_charged_on_the_schedules_settlement_row(self):
        sched = FeeSchedule.from_cards([card("2026-01-01T00:00:00+00:00", "20", settle="0.1")])
        cfg = priced(sched, "NFO-FUT")
        fut = InstrumentSpec.futures_contract("FUT", expiration_ns=2, lot_size=1.0, multiplier=10.0)
        r = run_strategy_backtest(
            holder_cls('FUT'), **bars([100, 101, 102, 103]), config=cfg, instrument=fut
        )
        (t,) = r.trades()
        assert t.exit_reason == "Settlement"
        # exit side: STT 0.1% of 102 * 10 * size on the closing (sell) side, no brokerage
        expected = 0.001 * 102.0 * 10.0 * t.size
        assert t.exit_fees == pytest.approx(expected, abs=0.01)
        assert t.fees == pytest.approx(t.entry_fees + t.exit_fees, abs=1e-9)

    def test_a_schedule_without_a_settlement_row_falls_back_to_the_spec_fraction(self):
        sched = FeeSchedule.from_cards([card("2026-01-01T00:00:00+00:00", "20")])
        cfg = priced(sched, "NFO-FUT")
        fut = InstrumentSpec.futures_contract("FUT", expiration_ns=2, lot_size=1.0)
        fut.settlement_fee = 0.001
        r = run_strategy_backtest(
            holder_cls('FUT'), **bars([100, 101, 102, 103]), config=cfg, instrument=fut
        )
        (t,) = r.trades()
        assert t.exit_fees == pytest.approx(0.001 * 102.0 * t.size, abs=0.01)

    def test_a_fill_on_day_d_is_charged_by_the_row_in_force_on_d(self):
        old = card("2026-01-01T00:00:00+00:00", "20", name="old")
        new = card("2026-01-03T00:00:00+00:00", "50", name="new")
        sched = FeeSchedule.from_cards([old, new])
        cfg = priced(sched, "NFO-FUT")
        day0 = 1_767_225_600 * 10**9  # 2026-01-01T00:00:00Z

        class RoundTrips(Strategy):
            def on_bar(self, ctx):
                if ctx.idx % 2 == 0:
                    self.submit_order(orders.Market(side="buy", units=1.0))
                else:
                    self.submit_order(orders.Market(side="sell", units=1.0))

        data = bars([100.0] * 6, start=day0, step=DAY)
        r = run_strategy_backtest(RoundTrips, **data, config=cfg)
        fees = [(t.entry_time, t.entry_fees, t.exit_fees) for t in r.trades()]
        by_day = {int((et - day0) // DAY): (ef, xf) for et, ef, xf in fees}
        # day 0 and 1 fill under the old row (20 per order); the entry on day 2
        # is the first under the new row (50)
        assert by_day[0][0] == pytest.approx(20.0)
        assert by_day[2][0] == pytest.approx(50.0)
        assert by_day[4][0] == pytest.approx(50.0)

    def test_the_per_instrument_day_fee_is_owed_once_per_trading_date(self):
        sched = FeeSchedule.from_cards(
            [
                RateCard.from_dict(
                    {
                        "name": "dp",
                        "effective_from": "2026-01-01T00:00:00+00:00",
                        "source": "test",
                        "rates": [
                            {
                                "exchange": "NSE",
                                "segment": "DELIVERY",
                                "underlying": None,
                                "brokerage": {"per_order": "0"},
                                "transaction_tax": {"kind": "STT", "percent": "0", "sides": ["SELL"]},
                                "exchange_turnover_percent": "0",
                                "sebi_percent": "0",
                                "stamp_duty_percent": "0",
                                "gst_percent": "0",
                                "dp_charge": {"per_scrip_day": "13", "gst": False},
                            }
                        ],
                    }
                )
            ]
        )
        cfg = priced(sched, "NSE-DELIVERY")
        day0 = 1_767_225_600 * 10**9 + 4 * 3600 * 10**9

        class TwoSellsADay(Strategy):
            def on_bar(self, ctx):
                # buy 10, sell 4, sell 4 on day 0 (bars an hour apart), then flat
                step = {0: ("buy", 10.0), 1: ("sell", 4.0), 2: ("sell", 4.0)}.get(ctx.idx)
                if step:
                    self.submit_order(orders.Market(side=step[0], units=step[1]))

        data = bars([100.0] * 4, start=day0, step=3600 * 10**9)
        r = run_strategy_backtest(TwoSellsADay, **data, config=cfg)
        exits = sorted(r.trades(), key=lambda t: t.exit_time)
        assert [t.exit_fees for t in exits[:2]] == [pytest.approx(13.0), pytest.approx(0.0)]


# -- corporate actions ------------------------------------------------------------


def equity_at(result, i):
    """The equity sampled at the i-th distinct timestamp."""
    return result.result.equity_curve()[i]


class TestSplits:
    def raw_run(self, ratio=2.0, units=10.0, closes=None, **kw):
        closes = closes or [100.0, 100.0, 50.0, 50.0, 55.0]
        s = holder("A", units=units)
        r = run(
            s,
            {"A": bars(closes)},
            corporate_actions=[CorporateAction.split("A", 2, ratio)],
            adjustment_policy="raw",
            **kw,
        )
        return s, r

    def test_a_split_moves_the_quantity_and_the_price_and_leaves_equity_alone(self):
        s, r = self.raw_run()
        assert equity_at(r, 1) == pytest.approx(equity_at(r, 2))
        (t,) = trades(r)
        assert t.size == 20.0 and t.entry_price == 50.0
        assert t.pnl == pytest.approx(20.0 * (55.0 - 50.0))
        (e,) = [x for x in s.lifecycle if x.kind == "corporate_action"]
        assert (e.outcome, e.quantity, e.cash) == ("split", 2.0, 0.0)

    def test_an_event_is_delivered_before_the_bar_at_the_ex_date(self):
        order = []

        class Order(Holder):
            def on_bar(self, ctx):
                order.append(("bar", ctx.timestamp))
                super().on_bar(ctx)

            def on_corporate_action(self, ctx, e):
                order.append(("action", None))

        s = Order()
        s.sym, s.units = "A", 10.0
        run(
            s,
            {"A": bars([100.0, 100.0, 50.0, 50.0])},
            corporate_actions=[CorporateAction.split("A", 2, 2.0)],
            adjustment_policy="raw",
        )
        i = order.index(("action", None))
        assert order[i + 1] == ("bar", 2)

    def test_a_fraction_of_a_share_is_paid_in_cash(self):
        spec = InstrumentSpec.equity("A", size_increment=1.0)
        s = holder("A", units=3.0)
        r = run(
            s,
            {"A": bars([100.0, 100.0, 40.0, 40.0])},
            instruments={"A": spec},
            corporate_actions=[CorporateAction.split("A", 2, 1.5)],
            adjustment_policy="raw",
        )
        # 3 x 1.5 = 4.5 shares: 4 kept, half a share paid at 100 / 1.5
        kept = [t for t in trades(r) if t.exit_reason != "CorporateAction"]
        lieu = [t for t in trades(r) if t.exit_reason == "CorporateAction"]
        assert [t.size for t in kept] == [4.0]
        assert len(lieu) == 1 and lieu[0].size == pytest.approx(0.5)
        assert lieu[0].exit_price == pytest.approx(100.0 / 1.5)
        (e,) = [x for x in s.lifecycle if x.kind == "corporate_action"]
        assert e.cash == pytest.approx(0.5 * 100.0 / 1.5, abs=0.01)

    def test_a_split_without_a_policy_is_refused(self):
        with pytest.raises(InvalidParameter) as err:
            run(
                holder("A"),
                {"A": bars([100.0, 50.0])},
                corporate_actions=[CorporateAction.split("A", 1, 2.0)],
            )
        assert err.value.code == "invalid_adjustment_policy"

    def test_an_action_on_a_symbol_without_data_is_refused(self):
        with pytest.raises(UnknownSymbol):
            run(
                holder("A"),
                {"A": bars([100.0, 50.0])},
                corporate_actions=[CorporateAction.split("B", 1, 2.0)],
                adjustment_policy="raw",
            )

    def test_malformed_actions_are_refused(self):
        with pytest.raises(InvalidParameter):
            CorporateAction.split("A", 1, 0.0)
        with pytest.raises(InvalidParameter):
            CorporateAction.dividend("A", 5, -1.0)
        with pytest.raises(InvalidParameter):
            CorporateAction.dividend("A", 5, 1.0, pay_ts=4)
        with pytest.raises(InvalidParameter):
            CorporateAction("A", "merger", 1, ratio=2.0)


class TestAdjustedHistory:
    DATA = [100.0, 102.0, 104.0, 53.0, 54.0, 55.0]  # a 2:1 split at ts 3

    def test_adjust_history_puts_earlier_bars_on_the_later_basis(self):
        out = adjust_history("A", [CorporateAction.split("A", 3, 2.0)], bars(self.DATA))
        assert list(out["close"][:3]) == [50.0, 51.0, 52.0]
        assert list(out["close"][3:]) == [53.0, 54.0, 55.0]
        assert list(out["volume"][:3]) == [2000.0] * 3
        # the input is untouched
        assert bars(self.DATA)["close"][0] == 100.0

    def test_a_split_does_not_look_like_a_crash_to_an_indicator(self):
        seen = {}

        class Watch(Holder):
            def on_start(self, ctx):
                self.fast = self.register_indicator(Indicator.sma(2), symbol="A")

            def on_bar(self, ctx):
                if self.fast.value is not None:
                    seen.setdefault(self.tag, []).append(float(self.fast.value))

        for policy in ("raw", "adjusted"):
            s = Watch()
            s.tag = policy
            s.sym = "A"
            run(
                s,
                {"A": bars(self.DATA)},
                corporate_actions=[CorporateAction.split("A", 3, 2.0)],
                adjustment_policy=policy,
            )
        raw = np.array(seen["raw"])
        adj = np.array(seen["adjusted"])
        assert raw.min() < 0.7 * raw.max()  # the raw series crashes at the split
        assert adj.min() > 0.9 * adj.max() / 1.1  # adjusted does not
        assert np.all(np.diff(adj) > 0)

    def test_adjusted_and_raw_agree_on_the_money(self):
        raw = run(
            holder("A"),
            {"A": bars(self.DATA)},
            corporate_actions=[CorporateAction.split("A", 3, 2.0)],
            adjustment_policy="raw",
        )
        adj = run(
            holder("A"),
            {"A": bars(self.DATA)},
            corporate_actions=[CorporateAction.split("A", 3, 2.0)],
            adjustment_policy="adjusted",
        )
        assert raw.result.equity_curve()[-1] == pytest.approx(adj.result.equity_curve()[-1])
        np.testing.assert_allclose(raw.result.equity_curve(), adj.result.equity_curve())

    def test_adjusted_refuses_ticks(self):
        ticks = {
            "A": {
                "timestamps": np.arange(3, dtype=np.int64),
                "ltp": np.array([1.0, 2.0, 3.0]),
            }
        }
        with pytest.raises(InvalidParameter):
            run_strategies(
                [holder("A")],
                ticks=ticks,
                corporate_actions=[CorporateAction.split("A", 1, 2.0)],
                adjustment_policy="adjusted",
            )


class TestDividends:
    def test_a_long_is_credited_on_the_ex_date_and_a_short_pays(self):
        for side, sign in (("buy", 1.0), ("sell", -1.0)):
            s = holder("A", units=10.0, side=side)
            r = run(
                s,
                {"A": bars([100.0] * 5)},
                corporate_actions=[CorporateAction.dividend("A", 2, 3.0)],
                adjustment_policy="raw",
                account_type="margin",
            )
            (e,) = [x for x in s.lifecycle if x.kind == "corporate_action"]
            assert (e.outcome, e.amount, e.cash) == ("dividend", 3.0, sign * 30.0)
            assert equity_at(r, 2) - equity_at(r, 1) == pytest.approx(sign * 30.0)

    def test_the_cash_arrives_on_the_pay_date_for_the_holder_at_the_ex_date(self):
        class SellAtExDay(Holder):
            def on_bar(self, ctx):
                super().on_bar(ctx)
                if ctx.idx == 2:
                    self.submit_order(orders.Market(side="sell", units=10.0, instrument="A"))

        s = SellAtExDay()
        s.sym, s.units = "A", 10.0
        r = run(
            s,
            {"A": bars([100.0] * 6)},
            corporate_actions=[CorporateAction.dividend("A", 2, 3.0, pay_ts=4)],
            adjustment_policy="raw",
        )
        # held at the ex-date (sold during it): still entitled, credited at ts 4
        (e,) = [x for x in s.lifecycle if x.kind == "corporate_action"]
        assert e.cash == 30.0
        assert equity_at(r, 4) - equity_at(r, 3) == pytest.approx(30.0)

    def test_an_adjusted_run_restates_the_amount_to_the_final_basis(self):
        s = holder("A", units=10.0)
        run(
            s,
            {"A": bars([100.0, 100.0, 100.0, 50.0, 50.0])},
            corporate_actions=[
                CorporateAction.dividend("A", 1, 6.0),
                CorporateAction.split("A", 3, 2.0),
            ],
            adjustment_policy="adjusted",
        )
        (div,) = [x for x in s.lifecycle if x.outcome == "dividend"]
        # 6 per old share is 3 per share on the adjusted basis
        # 10 units bought on the adjusted basis (price 50): 3.0 each
        assert div.amount == 3.0
        assert div.cash == pytest.approx(30.0)


# -- the runners agree -------------------------------------------------------------


class TestRunnersAgree:
    def test_the_single_runner_the_portfolio_runner_and_run_strategies_settle_alike(self):
        fut = InstrumentSpec.futures_contract("FUT", expiration_ns=3, lot_size=1.0, multiplier=5.0)
        data = bars([100, 101, 102, 103, 104])
        single = run_strategy_backtest(
            holder_cls("FUT"), **data, config=config(), instrument=fut
        )
        port = run_portfolio_strategy(
            holder("FUT"), {"FUT": data}, config=config(), instruments={"FUT": fut},
            bar_stamp="close",
        )
        many = run_strategies(
            [holder("FUT")], {"FUT": data}, config=config(), instruments={"FUT": fut},
            bar_stamp="close",
        )[0]
        key = lambda t: (t.exit_reason, t.exit_price, t.exit_time, t.size, t.pnl)  # noqa: E731
        assert [key(t) for t in single.trades()] == [key(t) for t in trades(port)]
        assert [key(t) for t in trades(port)] == [key(t) for t in trades(many)]

    def test_corporate_actions_agree_between_the_portfolio_runner_and_run_strategies(self):
        acts = [CorporateAction.split("A", 2, 2.0), CorporateAction.dividend("A", 3, 1.0)]
        kw = dict(corporate_actions=acts, adjustment_policy="raw", config=config(),
                  bar_stamp="close")
        data = {"A": bars([100.0, 100.0, 50.0, 50.0, 55.0])}
        a = run_portfolio_strategy(holder("A", units=10.0), data, **kw)
        b = run_strategies([holder("A", units=10.0)], data, **kw)[0]
        np.testing.assert_array_equal(a.result.equity_curve(), b.result.equity_curve())
        assert [t.pnl for t in trades(a)] == [t.pnl for t in trades(b)]
