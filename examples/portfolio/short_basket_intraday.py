"""A basket sold short at the open and bought back before the close, over members that change by day.

Shows a point-in-time basket. The strategy declares a ``Universe`` role like any basket, and the
run binds it to ``PointInTime(union, steps)``: bars are fed for every symbol that was ever a
member, and ``ctx.role("basket")`` answers the members **on the day of the event**, so a name is
ranked and traded only on the days it was in the basket (an index's constituents, the stocks
eligible that day). The manifest is the same ``Universe`` role a plain list is bound to.

Each session it ranks the day's members and, on the close of the first one-minute bar, sells short
the ``HOLD`` with the strongest prior-day return (a reversal bet), each sized so a ``STOP_PCT``
move against it loses ``RISK`` of the equity (``raptorbt.sizing.lots_for_risk``, whole shares:
the lot size is 1). The order is filled at the next bar's open. The holding is intraday, so the run
squares every position off at ``FLATTEN`` (the engine reads ``Holding.intraday``); the strategy
sends no exit.

The lookahead guard. The rank reads only the days before today: ``prior_closes`` cuts each
member's one-minute series at the start of the session, so not one of today's bars, the first
included, is in it. The decision is made on the close of the first bar and filled at the next
bar's open, so no price after the decision is used. A member with fewer than ``DAYS + 1`` earlier
sessions is not ranked.

Skills: raptorbt-portfolio, raptorbt-strategy, raptorbt-risk.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _data  # noqa: E402
from raptorbt import BacktestConfig, Strategy, run_strategies  # noqa: E402
from raptorbt.errors import InvalidParameter  # noqa: E402
from raptorbt.sizing import lots_for_risk  # noqa: E402
from raptorbt.strategy import (  # noqa: E402
    Holding,
    MaxPosition,
    PointInTime,
    Requirements,
    Universe,
    check_requirements,
    requirements_hash,
)
from raptorbt.strategy.orders import Market  # noqa: E402

DAYS = 1  # the prior-day return: the last close against the close DAYS sessions earlier
HOLD = 2  # names sold short each session
RISK = 0.004  # the share of the equity one name may lose to a STOP_PCT move
STOP_PCT = 0.02
FLATTEN = "15:15"
DAY_NS = 86_400 * 10**9


class ShortTheStrongest(Strategy):
    """Short the ``HOLD`` members with the best prior-day return, flat by ``FLATTEN``."""

    requirements = Requirements(
        roles={"basket": Universe(asset="equity", min=3, max=10, data=("bar",))},
        timeframes=["1m"],
        holding=Holding.intraday(FLATTEN),
        data=["bar"],
        max_position={"basket": MaxPosition(fraction=0.25)},
    )

    def __init__(self):
        super().__init__()
        self.day = None  # the IST day the last decision was made on
        self.decided = []  # (day, the members that day, the names sold short)

    def on_bar(self, ctx):
        now = ctx.event_ns
        day = (now + _data.IST_OFFSET_NS) // DAY_NS
        if day == self.day:
            return  # one decision a session, on its first bar
        self.day = day
        members = ctx.role("basket")  # the members on this day, not the union the run feeds
        ranked = sorted(
            ((prior_return(ctx.series(s), day), s) for s in members),
            key=lambda pair: (pair[0] is None, -(pair[0] or 0.0), pair[1]),
        )
        picks = [s for r, s in ranked if r is not None][:HOLD]
        for symbol in picks:
            entry = float(ctx.series(symbol)["close"][-1])  # the first bar's close: the decision
            try:
                shares = lots_for_risk(ctx.equity, RISK, entry, entry * (1.0 + STOP_PCT), 1)
            except InvalidParameter:  # no equity left, or a zero risk: no entry
                continue
            if shares:
                self.submit_order(Market(side="sell", quantity=shares), symbol=symbol)
        self.decided.append((day, tuple(members), tuple(picks)))


def prior_closes(series, day):
    """The closes of the sessions before ``day`` (an IST day number), one per session: the last
    one-minute close of each. Nothing of ``day`` itself is read."""
    stamps = (np.asarray(series["timestamps"]) + _data.IST_OFFSET_NS) // DAY_NS
    before = stamps < day
    days, closes = stamps[before], np.asarray(series["close"])[before]
    if not len(days):
        return closes
    last = np.flatnonzero(np.diff(days)) if len(days) > 1 else np.array([], dtype=int)
    return closes[np.append(last, len(days) - 1)]


def prior_return(series, day):
    """The return of the last prior close over the close ``DAYS`` sessions before it; None when
    there are not that many sessions."""
    closes = prior_closes(series, day)
    if len(closes) < DAYS + 1:
        return None
    return float(closes[-1] / closes[-1 - DAYS] - 1.0)


def sessions(n):
    """The first ``n`` IST session days of the synthetic data, as dates."""
    out, day = [], _data.FIRST_DAY
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += dt.timedelta(days=1)
    return out


def main():
    assert check_requirements(ShortTheStrongest) == []
    print("manifest", requirements_hash(ShortTheStrongest)[:12], ShortTheStrongest.requirements.roles["basket"])

    days = sessions(6)
    union = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    data = {s: _data.bars(days=6, seed=60 + i, start=100.0 + 15 * i, step=0.25) for i, s in enumerate(union)}
    # AAA is a member of the first three sessions, EEE of the last three; the others throughout
    steps = [
        (_data.ist_ns(days[0]), ["AAA", "BBB", "CCC", "DDD"]),
        (_data.ist_ns(days[3]), ["BBB", "CCC", "DDD", "EEE"]),
    ]
    strategy = ShortTheStrongest()
    (result,) = run_strategies(
        [(strategy, union)], data,
        BacktestConfig(initial_capital=1_000_000.0, fees=0.0003, session_tz_offset_ns=_data.IST_OFFSET_NS),
        bar_stamp="close",
        bindings={"basket": PointInTime(union, steps)},
        account_type="margin",
    )

    day_of = {(_data.ist_ns(d) + _data.IST_OFFSET_NS) // DAY_NS: d for d in days}
    for day, members, picks in strategy.decided:
        print(day_of[day], "members", "/".join(members), " short", "/".join(picks) or "-")
    first_member = {s: (_data.ist_ns(days[3]) if s == "EEE" else 0) for s in union}
    last_member = {"AAA": _data.ist_ns(days[3])}
    trades = result.result.trades()
    for t in trades:
        assert t.direction == -1, "the strategy only sells short"
        assert t.entry_time >= first_member[t.symbol], f"{t.symbol} traded before it was a member"
        assert t.entry_time < last_member.get(t.symbol, 1 << 62), f"{t.symbol} traded after it left"
        assert (t.exit_time + _data.IST_OFFSET_NS) // DAY_NS == (t.entry_time + _data.IST_OFFSET_NS) // DAY_NS, (
            "flat before the close of the session it opened in"
        )
    assert all(len(members) == 4 for _, members, _ in strategy.decided)
    assert trades, "the strategy never traded"
    m = result.result.metrics
    print(f"trades {m.total_trades}  return {m.total_return_pct:.2f}%")


if __name__ == "__main__":
    main()
