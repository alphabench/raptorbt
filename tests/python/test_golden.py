"""Golden-result regression gate.

Replays the fixture corpus (``golden/generate.py``) and asserts bit-exact
equality with the committed baselines. Any diff here means the execution
core's numbers changed — which requires a deliberate fixture regeneration,
version bump, and changelog entry, never an incidental refactor.
"""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "golden"))

from generate import (  # noqa: E402
    GoldenSma,
    config_variants,
    make_data,
    make_signals,
    result_digest,
)

import raptorbt  # noqa: E402
from raptorbt import PyBacktestConfig  # noqa: E402

FIXTURES = json.loads((HERE / "golden" / "fixtures.json").read_text())


def assert_digest_equal(actual: dict, expected: dict, name: str) -> None:
    assert actual.keys() == expected.keys(), f"{name}: digest shape changed"
    for key in expected:
        assert actual[key] == expected[key], f"{name}: {key} diverged from golden baseline"


@pytest.mark.parametrize("variant", [v[0] for v in config_variants()])
def test_single_path_matches_golden(variant):
    ts, o, h, l, c, v = make_data()
    entries, exits = make_signals(c)
    name, config, ic, direction = next(x for x in config_variants() if x[0] == variant)
    result = raptorbt.run_single_backtest(
        ts, o, h, l, c, v, entries, exits,
        direction=direction, config=config, instrument_config=ic,
    )
    assert_digest_equal(result_digest(result), FIXTURES[f"single/{name}"], name)


def test_class_path_matches_golden():
    ts, o, h, l, c, v = make_data()
    result = raptorbt.run_strategy_backtest(GoldenSma, ts, o, h, l, c, v)
    assert_digest_equal(result_digest(result), FIXTURES["class/sma_cross"], "class/sma_cross")


def test_portfolio_shared_pool_matches_golden():
    instruments = []
    for seed in (11, 12, 13):
        pts, po, ph, pl, pc, pv = make_data(300, seed=seed)
        pe, px = make_signals(pc)
        instruments.append((pts, po, ph, pl, pc, pv, pe, px, 1, 1.0, f"SYM{seed}"))
    portfolio = raptorbt.run_portfolio_backtest(
        instruments, config=PyBacktestConfig(), allocation="equal_weight"
    )
    expected = FIXTURES["portfolio/shared_pool"]
    actual = {
        "equity_curve": [float.hex(float(x)) for x in portfolio.result.equity_curve()],
        "total_return_pct": float.hex(portfolio.metrics.total_return_pct),
        "per_instrument": {
            s.symbol: {"trades": s.trades, "pnl": float.hex(s.pnl)}
            for s in portfolio.per_instrument
        },
    }
    assert_digest_equal(actual, expected, "portfolio/shared_pool")
