"""The native command path against the Python one.

``RAPTORBT_PY_COMMANDS=1`` keeps the original Python command path reachable.
Both must produce the same results, hook for hook, error for error: the frozen
runner scenarios and the ``run_strategies`` differential cases run under each
and are compared.
"""

import json
import sys
from pathlib import Path

import pytest

import raptorbt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "golden"))

import runner_fixtures as rf  # noqa: E402
import test_run_strategies as trs  # noqa: E402

FIXTURES = rf.load()
DATASETS = FIXTURES["datasets"]
SCENARIOS = {sc["name"]: sc for sc in rf.scenario_table()}
N_DIFFERENTIAL = 60


def _with_python_path(monkeypatch, on):
    if on:
        monkeypatch.setenv("RAPTORBT_PY_COMMANDS", "1")
    else:
        monkeypatch.delenv("RAPTORBT_PY_COMMANDS", raising=False)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario_is_identical_under_both_paths(name, monkeypatch):
    runs = []
    for python_path in (True, False):
        _with_python_path(monkeypatch, python_path)
        sc = json.loads(json.dumps(SCENARIOS[name]))
        runs.append(json.loads(json.dumps(rf.run_scenario(sc, DATASETS, backend="runtime"))))
    assert runs[0] == runs[1]


@pytest.mark.parametrize("case", range(N_DIFFERENTIAL))
def test_run_strategies_case_is_identical_under_both_paths(case, monkeypatch):
    digests = []
    for python_path in (True, False):
        _with_python_path(monkeypatch, python_path)
        data, kw, specs = trs._case(case)
        strats, items = trs._entries(specs)
        results = raptorbt.run_strategies(items, data, **kw)
        digests.append(trs._digest(results, strats))
    assert digests[0] == digests[1]


# ---------------------------------------------------------------------------
# refusals: the same exception type and message under both paths
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402

from raptorbt import Strategy  # noqa: E402
from raptorbt.strategy import orders  # noqa: E402


def _bars(n=30):
    t0 = 1_700_000_000 * 10**9
    ts = t0 + np.arange(n, dtype=np.int64) * 60 * 10**9
    close = 100.0 + np.sin(np.arange(n) / 3.0)
    return {
        "timestamps": ts,
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(n, 1000.0),
    }


class _Once(Strategy):
    """Runs ``act(strategy, ctx)`` on the third bar."""

    def __init__(self, act):
        super().__init__()
        self.act = act

    def on_bar(self, ctx):
        if ctx.idx == 2:
            self.act(self, ctx)


def _two_enters(s, ctx):
    s.enter()
    s.enter()


def _two_closes(s, ctx):
    s.close_position()
    s.close_position()


def _unknown_parent(s, ctx):
    s.submit_order(orders.Market(side="buy", units=1), parent="ghost")


def _parent_on_other_symbol(s, ctx):
    p = s.submit_order(orders.Limit(side="buy", price=50.0, units=1), symbol="AAA")
    s.submit_order(orders.Limit(side="buy", price=50.0, units=1), parent=p, symbol="BBB")


def _oco_across_symbols(s, ctx):
    a = s.submit_order(orders.Limit(side="buy", price=50.0, units=1), symbol="AAA")
    b = s.submit_order(orders.Limit(side="buy", price=50.0, units=1), symbol="BBB")
    s.link_oco(a, b)


def _unknown_symbol(s, ctx):
    s.submit_order(orders.Market(side="buy", units=1), symbol="NOPE")


def _units_and_frac_on_modify(s, ctx):
    a = s.submit_order(orders.Limit(side="buy", price=50.0, units=1))
    s._queue_drain_probe = a
    s.modify_order(a, units=1, size_frac=0.5)


def _bad_size(s, ctx):
    s.submit_order(orders.Market(side="buy", size_frac=7.0))


REFUSALS = [
    _two_enters,
    _two_closes,
    _unknown_parent,
    _parent_on_other_symbol,
    _oco_across_symbols,
    _unknown_symbol,
    _bad_size,
]


@pytest.mark.parametrize("act", REFUSALS, ids=lambda f: f.__name__)
def test_refusals_are_identical_under_both_paths(act, monkeypatch):
    seen = []
    for python_path in (True, False):
        _with_python_path(monkeypatch, python_path)
        data = {"AAA": _bars(), "BBB": _bars()}
        try:
            raptorbt.run_strategies([_Once(act)], data)
        except Exception as ex:  # noqa: BLE001 - the type and message are the subject
            seen.append((type(ex).__name__, str(ex)))
        else:
            seen.append(None)
    assert seen[0] == seen[1]
    assert seen[0] is not None


class _Cached(Strategy):
    """Trades through typed orders, then reports what its cache holds."""

    def __init__(self):
        super().__init__()
        self.report = None

    def on_bar(self, ctx):
        if ctx.idx == 2:
            self.submit_order(orders.Market(side="buy", units=3), symbol="AAA")
            self.submit_order(orders.Limit(side="buy", price=1.0, units=1), symbol="BBB")
            self.submit_order(orders.Limit(side="sell", price=500.0, units=1, reduce_only=True))
        if ctx.idx == 8:
            self.close_position(symbol="AAA")
            self.cancel_all_orders()

    def on_stop(self, ctx):
        c = self.cache
        self.report = (
            [
                (o.client_id, o.status, o.symbol, o.last_price, o.last_size, o.reject_reason)
                for o in c.orders()
            ],
            [(t.symbol, t.pnl) for t in c.closed_trades()],
            c.realized_pnl(),
        )


def test_cache_is_identical_under_both_paths(monkeypatch):
    reports = []
    for python_path in (True, False):
        _with_python_path(monkeypatch, python_path)
        strategy = _Cached()
        raptorbt.run_strategies([strategy], {"AAA": _bars(), "BBB": _bars()})
        reports.append(strategy.report)
    assert reports[0] == reports[1]
    assert reports[0][0] and reports[0][1]


# ---------------------------------------------------------------------------
# run_strategy_backtest
# ---------------------------------------------------------------------------
class _Single(Strategy):
    """Typed orders, intents and a same-bar refusal on a single instrument."""

    def __init__(self, clash=False):
        super().__init__()
        self.clash = clash
        self.seen = []

    def on_bar(self, ctx):
        i = ctx.idx
        if i == 3:
            self.enter(size_frac=0.5)
        if i == 6:
            self.submit_order(orders.Limit(side="sell", price=500.0, units=1, reduce_only=True))
        if i == 10:
            self.close_position()
            if self.clash:
                self.enter()
        if i == 14:
            self.submit_bracket(orders.Market(side="buy", units=2), 90.0, 120.0)

    def on_order_event(self, ctx, event):
        self.seen.append((ctx.idx, event.kind, event.client_order_id, event.price))

    def on_position_closed(self, ctx, event):
        self.seen.append((ctx.idx, "closed", event.trade.pnl))


def _single(monkeypatch, python_path, clash):
    _with_python_path(monkeypatch, python_path)
    bars = _bars(60)
    strategy = _Single(clash)
    try:
        result = raptorbt.run_strategy_backtest(
            strategy, bars["timestamps"], bars["open"], bars["high"], bars["low"],
            bars["close"], bars["volume"], bar_stamp="close",
        )
    except ValueError as ex:
        return str(ex), strategy.seen
    return [(t.entry_idx, t.exit_idx, t.pnl) for t in result.trades()], strategy.seen


@pytest.mark.parametrize("clash", [False, True])
def test_single_instrument_runner_is_identical_under_both_paths(monkeypatch, clash):
    python = _single(monkeypatch, True, clash)
    native = _single(monkeypatch, False, clash)
    assert python == native
    assert python[1]
    if clash:
        assert "same bar" in python[0]


# ---------------------------------------------------------------------------
# risk limits and hooks through every runner
# ---------------------------------------------------------------------------
import test_risk_limits as trl  # noqa: E402

RISK_CASES = [
    ({1: [("buy", 10.5)]}, dict(max_order_qty=10.0), trl._bars([100.0] * 4)),
    ({1: [("buy", 10.0)], 2: [("buy", 3.0)]}, dict(max_gross_exposure=1000.0), trl._bars([100.0] * 5)),
    ({1: [("enter", 0.5)], 2: [("halt", True)], 3: [("buy", 1.0)]}, {}, trl._bars([100.0] * 6)),
    (
        {1: [("buy", 100.0)], 3: [("buy", 1.0)], 5: [("buy", 1.0)]},
        dict(max_daily_loss=500.0),
        trl._days([100.0, 100.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0]),
    ),
]


@pytest.mark.parametrize("runner", sorted(trl.RUNNERS))
@pytest.mark.parametrize("case", range(len(RISK_CASES)))
def test_risk_events_and_rejections_are_identical_under_both_paths(runner, case, monkeypatch):
    script, limits, data = RISK_CASES[case]
    seen = []
    for python_path in (True, False):
        _with_python_path(monkeypatch, python_path)
        s = trl.Script(script)
        res = trl.RUNNERS[runner](s, data, trl._config(**limits))
        trades = [(t.entry_idx, t.exit_idx, t.size, t.pnl) for t in trl._result(res).trades()]
        seen.append((s.rejected, s.risk, s.canceled, s.filled, trades))
    assert seen[0] == seen[1]


def test_the_new_api_sessions_risk_and_error_suites_pass_on_the_python_path():
    """Their own assertions, with the Python command path selected."""
    import os
    import subprocess

    files = [
        "test_strategy_api.py", "test_sessions.py", "test_risk_limits.py", "test_errors.py",
    ]
    env = dict(os.environ, RAPTORBT_PY_COMMANDS="1")
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
         *[str(HERE / f) for f in files]],
        env=env, capture_output=True, text=True, timeout=900,
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-1000:]
