"""Shared fixtures."""

import gc

import pytest

import raptorbt


def pytest_configure(config):
    for name, doc in (
        ("portfolio_runner", "scenario covers run_portfolio_strategy"),
        ("tick_runner", "scenario covers run_tick_strategy"),
        ("tick_stream", "scenario covers TickStrategyStream"),
    ):
        config.addinivalue_line("markers", f"{name}: {doc}")


@pytest.fixture(autouse=True)
def _collect_cycles_on_this_thread():
    """Journals and records are unsendable (freed only by the thread that made them). A cycle left
    by an earlier test is otherwise collected by whichever thread allocates next, and when that is
    an executor thread the free raises and the test it lands in fails."""
    gc.collect()
    yield


@pytest.fixture(params=["legacy", "shared"])
def portfolio_runner(request):
    """``run_portfolio_strategy`` and the same call through ``run_strategies``.

    Both must behave identically, so a test written against one runs against
    both. The shared arm is skipped where ``run_strategies`` does not exist.
    """
    if request.param == "legacy":
        return raptorbt.run_portfolio_strategy
    if not hasattr(raptorbt, "run_strategies"):
        pytest.skip("run_strategies not available")
    return lambda strategy, data, **kw: raptorbt.run_strategies([strategy], data, **kw)[0]


@pytest.fixture(params=["legacy", "shared"])
def tick_runner(request):
    """``run_tick_strategy`` and the same call through ``run_strategies``."""
    if request.param == "legacy":
        return raptorbt.run_tick_strategy
    if not hasattr(raptorbt, "run_strategies"):
        pytest.skip("run_strategies not available")

    def shared(strategy, ticks, **kw):
        # Ticks carry event times; run_strategies asks for the declaration the
        # tick runner has no use for.
        kw.setdefault("bar_stamp", "close")
        return raptorbt.run_strategies([strategy], ticks=ticks, **kw)[0]

    return shared
