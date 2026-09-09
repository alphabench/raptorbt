"""The parallel batch path must equal the serial one, exactly.

`batch_single_backtest` exists to run a parameter sweep across cores instead
of down a Python loop. That is only worth having if the answers do not change:
a sweep whose numbers shift with the thread count would make every comparison
between two parameter sets meaningless. So the gate here is bit-equality with
`run_single_backtest`, not approximate agreement, and it is checked over a
corpus large enough that Rayon actually splits the work.

Ordering is part of the contract too. Results come back in input order, so a
caller can zip them against the parameters that produced them.
"""

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig, BatchSingleItem, InstrumentConfig


def make_data(n=600, seed=5):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.011, n)))
    ts = np.arange(n, dtype=np.int64) * 86_400_000_000_000
    return (
        ts,
        close * 0.999,
        close * 1.004,
        close * 0.996,
        close,
        np.abs(rng.normal(1e5, 1e4, n)),
    )


def sma_cross(close, fast, slow):
    def sma(a, w):
        cs = np.cumsum(np.insert(a, 0, 0.0))
        out = np.full(len(a), np.nan)
        out[w - 1 :] = (cs[w:] - cs[:-w]) / w
        return out

    f, s = sma(close, fast), sma(close, slow)
    up = (f > s) & ~np.isnan(f) & ~np.isnan(s)
    entries = np.zeros(len(close), bool)
    exits = np.zeros(len(close), bool)
    entries[1:] = up[1:] & ~up[:-1]
    exits[1:] = ~up[1:] & up[:-1]
    return entries, exits


def assert_identical(res_a, res_b, label):
    ta, tb = res_a.trades(), res_b.trades()
    assert len(ta) == len(tb), label
    for a, b in zip(ta, tb):
        assert a.entry_idx == b.entry_idx, label
        assert a.exit_idx == b.exit_idx, label
        assert a.entry_price == b.entry_price, label
        assert a.exit_price == b.exit_price, label
        assert a.size == b.size, label
        assert a.pnl == b.pnl, label
        assert a.fees == b.fees, label
        assert a.exit_reason == b.exit_reason, label
    assert np.array_equal(res_a.equity_curve(), res_b.equity_curve()), label
    assert np.array_equal(res_a.drawdown_curve(), res_b.drawdown_curve()), label
    assert np.array_equal(res_a.returns(), res_b.returns()), label
    ma, mb = res_a.metrics, res_b.metrics
    for field in (
        "total_return_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "omega_ratio",
        "max_drawdown_pct",
        "ulcer_index",
        "time_under_water_pct",
        "win_rate_pct",
        "profit_factor",
        "total_trades",
        "total_fees_paid",
        "expectancy",
        "exposure_pct",
    ):
        assert getattr(ma, field) == getattr(mb, field), f"{label}: {field}"


def test_batch_matches_serial_across_a_sweep():
    """The reason this function exists: many signal sets, one price series."""
    ts, o, h, l, c, v = make_data()
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0002)
    combos = [(f, s) for f in (5, 10, 15, 20) for s in (30, 50, 80, 120)]

    signals = [sma_cross(c, f, s) for f, s in combos]
    serial = [
        raptorbt.run_single_backtest(
            ts, o, h, l, c, v, e, x, direction=1, symbol="B", config=cfg
        )
        for e, x in signals
    ]
    items = [
        BatchSingleItem(f"{f}_{s}", e, x, 1, 1.0, "B", cfg)
        for (f, s), (e, x) in zip(combos, signals)
    ]
    batch = raptorbt.batch_single_backtest(ts, o, h, l, c, v, items, cfg)

    assert [bid for bid, _ in batch] == [f"{f}_{s}" for f, s in combos]
    assert len(batch) == len(serial)
    for (bid, got), want in zip(batch, serial):
        assert_identical(want, got, bid)


def test_per_item_config_overrides_the_batch_config():
    """An item's own config wins; items without one fall back to the batch."""
    ts, o, h, l, c, v = make_data()
    e, x = sma_cross(c, 10, 50)
    base = BacktestConfig(initial_capital=100_000.0, fees=0.0002)

    stopped = BacktestConfig(initial_capital=100_000.0, fees=0.0002)
    stopped.set_fixed_stop(0.02)

    batch = raptorbt.batch_single_backtest(
        ts,
        o,
        h,
        l,
        c,
        v,
        [
            BatchSingleItem("plain", e, x, 1, 1.0, "B"),
            BatchSingleItem("stopped", e, x, 1, 1.0, "B", stopped),
        ],
        base,
    )
    results = dict(batch)

    assert_identical(
        raptorbt.run_single_backtest(
            ts, o, h, l, c, v, e, x, direction=1, symbol="B", config=base
        ),
        results["plain"],
        "plain",
    )
    assert_identical(
        raptorbt.run_single_backtest(
            ts, o, h, l, c, v, e, x, direction=1, symbol="B", config=stopped
        ),
        results["stopped"],
        "stopped",
    )
    # The override has to actually bite, or the test above proves nothing.
    reasons = {t.exit_reason for t in results["stopped"].trades()}
    assert "StopLoss" in reasons


def test_direction_and_instrument_config_travel_with_the_item():
    ts, o, h, l, c, v = make_data()
    e, x = sma_cross(c, 10, 50)
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0002)
    ic = InstrumentConfig(lot_size=25.0, alloted_capital=40_000.0)

    batch = dict(
        raptorbt.batch_single_backtest(
            ts,
            o,
            h,
            l,
            c,
            v,
            [
                BatchSingleItem("short", e, x, -1, 1.0, "B", cfg),
                BatchSingleItem("lots", e, x, 1, 1.0, "B", cfg, None, ic),
            ],
            cfg,
        )
    )

    assert_identical(
        raptorbt.run_single_backtest(
            ts, o, h, l, c, v, e, x, direction=-1, symbol="B", config=cfg
        ),
        batch["short"],
        "short",
    )
    assert_identical(
        raptorbt.run_single_backtest(
            ts,
            o,
            h,
            l,
            c,
            v,
            e,
            x,
            direction=1,
            symbol="B",
            config=cfg,
            instrument_config=ic,
        ),
        batch["lots"],
        "lots",
    )


def test_repeated_runs_agree():
    """Rayon may schedule differently run to run; the numbers may not move."""
    ts, o, h, l, c, v = make_data()
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0002)
    combos = [(f, s) for f in (5, 10, 15) for s in (40, 70, 110)]
    items = [
        BatchSingleItem(f"{f}_{s}", *sma_cross(c, f, s), 1, 1.0, "B", cfg)
        for f, s in combos
    ]

    first = raptorbt.batch_single_backtest(ts, o, h, l, c, v, items, cfg)
    second = raptorbt.batch_single_backtest(ts, o, h, l, c, v, items, cfg)

    assert [i for i, _ in first] == [i for i, _ in second]
    for (aid, a), (_, b) in zip(first, second):
        assert_identical(a, b, aid)


# ---------------------------------------------------------------------------
# Refusals. A bad item must name itself, and must do so before any worker
# starts -- a panic on a Rayon thread crosses PyO3 as PanicException, which a
# caller can neither catch as ValueError nor trace to the argument at fault.
# ---------------------------------------------------------------------------


def test_mismatched_signal_length_names_the_item():
    ts, o, h, l, c, v = make_data()
    e, x = sma_cross(c, 10, 50)
    with pytest.raises(ValueError, match="bad_one"):
        raptorbt.batch_single_backtest(
            ts,
            o,
            h,
            l,
            c,
            v,
            [
                BatchSingleItem("fine", e, x, 1, 1.0, "B"),
                BatchSingleItem("bad_one", e[:-1], x[:-1], 1, 1.0, "B"),
            ],
        )


def test_mismatched_position_sizes_are_refused():
    ts, o, h, l, c, v = make_data()
    e, x = sma_cross(c, 10, 50)
    with pytest.raises(ValueError, match="position_sizes"):
        raptorbt.batch_single_backtest(
            ts,
            o,
            h,
            l,
            c,
            v,
            [
                BatchSingleItem(
                    "short_sizes", e, x, 1, 1.0, "B", None, np.full(len(c) - 3, 0.5)
                )
            ],
        )


def test_bad_direction_is_refused():
    ts, o, h, l, c, v = make_data()
    e, x = sma_cross(c, 10, 50)
    with pytest.raises(ValueError, match="direction"):
        raptorbt.batch_single_backtest(
            ts, o, h, l, c, v, [BatchSingleItem("sideways", e, x, 0, 1.0, "B")]
        )


def test_empty_batch_returns_empty():
    ts, o, h, l, c, v = make_data()
    assert raptorbt.batch_single_backtest(ts, o, h, l, c, v, []) == []
