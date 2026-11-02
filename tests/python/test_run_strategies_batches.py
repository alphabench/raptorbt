"""``run_strategies(batch_size=...)``: batching changes nothing but the memory profile."""

import pytest

from raptorbt import run_strategies

from test_run_strategies_smoke import KW, SYMBOLS, Recorder, fingerprint, legacy, make_data, same

SUBSETS = [
    SYMBOLS,
    SYMBOLS[:2],
    SYMBOLS[1:4],
    SYMBOLS[::2],
    SYMBOLS[3:],
    SYMBOLS[:1],
    SYMBOLS[2:],
    SYMBOLS[:3],
]


def entries():
    return [(Recorder(fast=3 + i % 3), SUBSETS[i]) for i in range(len(SUBSETS))]


def test_batch_size_changes_nothing():
    data = make_data(3, n=120)
    k = len(SUBSETS)
    refs = []
    for i in range(k):
        ref = Recorder(fast=3 + i % 3)
        refs.append((ref, fingerprint(legacy(ref, data, SUBSETS[i]))))
    for size in (1, 3, 7, k, k + 5, None):
        batch = entries()
        got = run_strategies(batch, data, batch_size=size, **KW)
        assert len(got) == k
        for (strat, _), res, (ref, want) in zip(batch, got, refs):
            same(fingerprint(res), want)
            # The hooks saw exactly what they would in a run of their own.
            assert strat.rec == ref.rec


def test_the_store_is_built_once_and_shared_by_every_batch(monkeypatch):
    from raptorbt.strategy import multi_runner

    built, runtimes = [], []
    real_data, real_runtime = multi_runner.MarketData, multi_runner.Runtime

    def market(*a, **kw):
        built.append(1)
        return real_data(*a, **kw)

    def runtime(m, *a, **kw):
        runtimes.append(m)
        return real_runtime(m, *a, **kw)

    monkeypatch.setattr(multi_runner, "MarketData", market)
    monkeypatch.setattr(multi_runner, "Runtime", runtime)
    run_strategies(entries(), make_data(4, n=60), batch_size=3, **KW)
    assert len(built) == 1
    assert len(runtimes) == 3
    assert all(m is runtimes[0] for m in runtimes)


def test_memory_report_counts_the_store_once_and_the_peak_batch():
    data = make_data(5, n=60)
    _, whole = run_strategies(entries(), data, memory_report=True, **KW)
    _, tiled = run_strategies(entries(), data, batch_size=3, memory_report=True, **KW)
    assert whole["batches"] == 1 and tiled["batches"] == 3
    assert tiled["batch_size"] == 3 and tiled["strategies"] == len(SUBSETS)
    assert tiled["shared_store"] == whole["shared_store"]
    assert tiled["total"] - tiled["shared_store"] < whole["total"] - whole["shared_store"]
    assert 0 < tiled["engine_bytes_per_strategy"] < 1024


@pytest.mark.parametrize("bad", [0, -1, 2.5, "3", True])
def test_invalid_batch_size_raises(bad):
    with pytest.raises(ValueError, match="batch_size"):
        run_strategies([Recorder()], make_data(6, n=30), batch_size=bad, **KW)
