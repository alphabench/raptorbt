"""Expiry, settlement, delivery and corporate actions against the reference implementation,
replayed from fixtures.

The fixture holds the numbers a reference implementation produced for these inputs. These tests replay the same market and holding
through this engine and compare cash, positions and closed trades.

Differences that are intended, and where they show:

* The reference engine settles a contract at its underlying's mark whenever it has
  one (a future at the mark, an option at intrinsic value). Here the settlement
  price comes from the terms declared on the instrument and the source is
  stated on the event; these scenarios declare ``"underlying"``, which is the
  reference behaviour, or a final price, which is the reference ``final_prices``.
* A fraction of a share that a split leaves over is paid in cash in both
  engines; here it also closes as a trade slice (reason ``CorporateAction``,
  no fee), which the reference trade list does not show. Its cash is the same.
* The reference engine reports a physical delivery as a ``Lifecycle`` event and
  records a ``SETTLEMENT`` decision; here the strategy hears
  ``on_contract_event`` and the trade list carries the settlement.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import raptorbt as r
from raptorbt._raptorbt import PortfolioSession

HERE = Path(__file__).parent
FIXTURE = json.loads((HERE / "lifecycle_fixture.json").read_text())
T0, MIN = FIXTURE["t0"], FIXTURE["minute"]
DER, UND = "NFO:DER", "NSE:UND"


def spec_for(s):
    """The instrument held, and the underlying, in this engine."""
    if s["kind"] == "SHARE":
        return None, r.InstrumentSpec.equity(UND, size_increment=1.0)
    # a contract with no expiry in the scenario expires far beyond its data
    expiry = T0 + (s["expiry_min"] if s["expiry_min"] is not None else 10**6) * MIN
    under = r.InstrumentSpec.equity(UND)
    if s["kind"] == "FUTURE":
        der = r.InstrumentSpec.futures_contract(
            DER, expiration_ns=expiry, lot_size=1.0, multiplier=float(s["multiplier"]),
            underlying=UND,
        )
        # reference: a final price when one is given, else the underlying's mark
        der.settlement_price = s["declared"] if "declared" in s else "underlying"
    else:
        der = r.InstrumentSpec.option(
            DER, strike=float(s["strike"]), right="call" if s["kind"] == "CALL" else "put",
            expiration_ns=expiry, lot_size=1.0, underlying=UND,
        )
        der.settlement_price = "underlying"
        if s.get("settlement") == "PHYSICAL":
            der.settlement_style = "physical"
    return der, under


def replay(s):
    config = r.BacktestConfig(initial_capital=FIXTURE["capital"])
    config.fees = 0.0
    config.slippage = 0.0
    if "cards" in s:
        config.fee_segment = "NFO-FUT"
        config.fee_schedule = r.FeeSchedule.from_cards([r.RateCard.from_dict(c) for c in s["cards"]])
    step = s.get("step", MIN)
    session = PortfolioSession(config=config, account_type="margin", leverage=1.0)
    der_spec, und_spec = spec_for(s)
    ids = {}
    if der_spec is not None:
        ids[DER] = session.add_instrument(DER, direction=1, instrument=der_spec)
    ids[UND] = session.add_instrument(UND, direction=1, instrument=und_spec)

    def feed(symbol, closes):
        c = np.asarray(closes, dtype=np.float64)
        ts = T0 + np.arange(len(c), dtype=np.int64) * step
        session.set_bars(ids[symbol], ts, c, c, c, c, np.full(len(c), 100_000.0))

    if der_spec is not None:
        feed(DER, s["der_closes"])
    feed(UND, s["und_closes"])
    if "action" in s:
        a = s["action"]
        kind = a["kind"].lower()
        ts = T0 + a["minute"] * MIN
        action = (
            r.CorporateAction.dividend(UND, ts, a["amount"])
            if kind == "dividend"
            else r.CorporateAction(UND, kind, ts, ratio=a["ratio"])
        )
        session.set_corporate_actions([action], "raw")
    session.seal()
    held = DER if der_spec is not None else UND
    plan = {i: (side, qty) for i, side, qty in (s.get("plan") or [[0, s["side"], s["qty"]]])}
    seen = -1
    trades = []
    while (ev := session.current_event()) is not None:
        if ev[1] == ids[held]:
            seen += 1
            if seen in plan:
                side, qty = plan[seen]
                session.submit_order(
                    ids[held], side.lower(), "market", ev[2], ev[3], f"h-{seen}", units=float(qty)
                )
        for e in session.apply_current():
            if e.kind == "exited":
                trades.append(e.trade)
    positions = sorted(
        [sym, p.size * p.direction, p.entry_price]
        for sym, i in ids.items()
        for p in session.positions(i)
    )
    return session.cash(), positions, trades


@pytest.mark.parametrize("scenario", FIXTURE["scenarios"], ids=lambda s: s["name"])
def test_the_account_and_the_holding_match_the_1x_engine(scenario):
    old = scenario["old"]
    cash, positions, trades = replay(scenario)
    assert cash == pytest.approx(old["cash"], abs=1e-6)
    assert len(positions) == len(old["positions"])
    for (sym, qty, avg), (osym, oqty, oavg) in zip(positions, old["positions"]):
        assert (sym, qty) == (osym, oqty)
        assert avg == pytest.approx(oavg, rel=1e-12)


# trades the lifecycle closed or a plan's own order closed, not the run's end
LIFECYCLE = ("Settlement", "Order")


@pytest.mark.parametrize("scenario", FIXTURE["scenarios"], ids=lambda s: s["name"])
def test_the_closed_trades_match_the_1x_engine(scenario):
    old = scenario["old"]
    _, _, trades = replay(scenario)
    # closed by the lifecycle, not by the end of the run
    ours = [
        [t.symbol, t.exit_reason, t.exit_price, t.size, t.pnl, t.direction, t.entry_fees, t.exit_fees]
        for t in trades
        if t.exit_reason in LIFECYCLE
    ]
    theirs = [t for t in old["trades"] if t[1] in LIFECYCLE]
    assert len(ours) == len(theirs)
    for a, b in zip(ours, theirs):
        assert a[:2] == b[:2]
        assert a[2:] == pytest.approx(b[2:], abs=1e-9)
