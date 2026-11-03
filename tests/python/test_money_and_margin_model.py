"""Booked money, and the margin model a config can choose.

Cash, fees and realised P&L are integer minor units (``money_decimals``,
default 2); ``pnl`` and ``fees`` are their float views and ``pnl_minor`` /
``fees_minor`` the exact companions. ``BacktestConfig(margin_model=...)`` funds
and refuses entries on a margin account through one path; unset, the built-in
rule applies exactly as before.
"""

import numpy as np
import pytest

from raptorbt import (
    BacktestConfig,
    InstrumentSpec,
    LiveNode,
    MarginModel,
    Strategy,
    run_portfolio_strategy,
)
from raptorbt.errors import InvalidParameter

SYM = "FUT"
DAY = 86_400 * 10**9
CAPITAL = 1_000_000.0


def _bars(closes):
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    return {
        "timestamps": np.arange(n, dtype=np.int64) * 60 * 10**9,
        "open": closes.copy(),
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
        "volume": np.full(n, 1_000.0),
    }


def _fut(**kw):
    return InstrumentSpec.futures_contract(SYM, 20_000 * DAY, 1.0, **kw)


class _EnterOnce(Strategy):
    def __init__(self, config=None, frac=1.0):
        super().__init__(config)
        self.rejects = []
        self.done = False
        self.frac = frac

    def on_bar(self, ctx):
        if self.done:
            return
        self.done = True
        self.enter(size_frac=self.frac)

    def on_order_rejected(self, ctx, event):
        self.rejects.append(event.reject_reason)


def _run(model=None, direction=1, spec=None, leverage=5.0, account="margin", closes=None, **cfg):
    config = BacktestConfig(initial_capital=CAPITAL, fees=0.0, margin_model=model, **cfg)
    strategy = _EnterOnce()
    result = run_portfolio_strategy(
        strategy,
        {SYM: _bars(closes if closes is not None else [100.0, 102.0, 104.0, 103.0])},
        config=config,
        directions={SYM: direction},
        instruments={SYM: spec or _fut()},
        account_type=account,
        leverage=leverage,
        bar_stamp="close",
    )
    return result, strategy


def test_the_built_in_rule_is_the_from_spec_model_and_unset_is_unchanged():
    spec = _fut(margin_init=0.2, margin_maint=0.1)
    base, _ = _run(None, spec=spec)
    chosen, _ = _run(MarginModel.from_spec(5), spec=spec)
    assert chosen.result.trades()[0].size == base.result.trades()[0].size
    assert list(chosen.result.equity_curve()) == list(base.result.equity_curve())
    # 20% initial margin: five times capital in notional.
    assert base.result.trades()[0].size == pytest.approx(CAPITAL / (100.0 * 0.2))


def test_a_percent_of_notional_model_sizes_and_locks_by_its_own_figure():
    result, _ = _run(MarginModel.fraction_of_notional(0.5))
    assert result.result.trades()[0].size == pytest.approx(CAPITAL / 50.0)


def test_no_margin_locks_nothing_and_sizes_at_full_notional():
    """The sizing rule for a model that requires nothing: a fraction of capital
    buys what the same fraction buys in cash, never more."""
    result, _ = _run(MarginModel.no_margin())
    assert result.result.trades()[0].size == pytest.approx(CAPITAL / 100.0)


def test_cash_only_refuses_a_short_with_a_stable_reason_and_funds_a_long():
    long_run, _ = _run(MarginModel.cash_only(), direction=1)
    assert long_run.result.trades()[0].size == pytest.approx(CAPITAL / 100.0)
    short_run, strategy = _run(MarginModel.cash_only(), direction=-1)
    assert short_run.result.trades() == []
    assert strategy.rejects == ["insufficient_margin"]


def test_a_model_without_a_rule_for_the_instrument_refuses_and_says_so():
    result, strategy = _run(MarginModel.leverage({"CASH": 5}))
    assert result.result.trades() == []
    assert strategy.rejects == ["margin_unpriced"]


def test_a_model_on_a_cash_account_is_refused_not_ignored():
    with pytest.raises(InvalidParameter) as e:
        _run(MarginModel.cash_only(), account="cash", leverage=1.0)
    assert e.value.code == "invalid_margin_model"


def test_trades_carry_exact_minor_units_that_the_floats_are_views_of():
    result, _ = _run_fees()
    for t in result.result.trades():
        assert t.money_decimals == 2
        assert t.pnl == t.pnl_minor / 100
        assert t.fees == t.fees_minor / 100
        gross = (
            t.exit_notional_minor - t.entry_notional_minor
            if t.direction == 1
            else t.entry_notional_minor - t.exit_notional_minor
        )
        assert t.pnl_minor == gross - t.fees_minor


def _run_fees():
    config = BacktestConfig(initial_capital=CAPITAL, fees=0.0003)
    result = run_portfolio_strategy(
        _EnterOnce(),
        {SYM: _bars([100.0, 101.37, 99.91, 103.33])},
        config=config,
        directions={SYM: 1},
        instruments={SYM: _fut()},
        bar_stamp="close",
    )
    return result, None


def test_money_decimals_is_a_scale_per_currency_and_bounded():
    config = BacktestConfig(initial_capital=CAPITAL, fees=0.0003, money_decimals=0)
    result = run_portfolio_strategy(
        _EnterOnce(),
        {SYM: _bars([100.0, 101.37, 99.91, 103.33])},
        config=config,
        directions={SYM: 1},
        instruments={SYM: _fut()},
        bar_stamp="close",
    )
    for t in result.result.trades():
        assert t.money_decimals == 0
        assert t.pnl == float(t.pnl_minor)
    with pytest.raises(InvalidParameter) as e:
        BacktestConfig(money_decimals=10)
    assert e.value.code == "invalid_money_decimals"


def test_a_live_node_takes_the_model_on_a_margin_account_and_refuses_it_on_cash():
    config = BacktestConfig(initial_capital=CAPITAL, margin_model=MarginModel.cash_only())
    common = dict(
        symbols=[SYM], directions={SYM: 1}, instruments={SYM: _fut()}, config=config
    )
    LiveNode([_EnterOnce()], account_type="margin", leverage=5.0, **common)
    with pytest.raises(InvalidParameter) as e:
        LiveNode([_EnterOnce()], account_type="cash", **common)
    assert e.value.code == "invalid_margin_model"
