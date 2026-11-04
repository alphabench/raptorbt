"""Synthetic dataset generator: determinism, bar oracle, validity, engine smoke."""


import numpy as np
import pytest

import synthetic as syn  # noqa: E402

DAY = syn.DAY_NS


@pytest.fixture(scope="module")
def ds():
    return syn.generate(n_instruments=12, days=2, avg_ticks_per_sec=0.2, seed=7)


def test_same_seed_identical_different_seed_differs(ds):
    again = syn.generate(n_instruments=12, days=2, avg_ticks_per_sec=0.2, seed=7)
    assert again.manifest == ds.manifest
    other = syn.generate(n_instruments=12, days=2, avg_ticks_per_sec=0.2, seed=8)
    assert other.manifest["arrays"] != ds.manifest["arrays"]


def test_tier_s_params_exceed_200_instruments():
    assert syn.TIERS["S"]["n_instruments"] > 200
    # Instrument generation alone is cheap; ticks are not materialised.
    g = np.random.default_rng(1)
    insts, w, _ = syn._make_instruments(syn.TIERS["S"]["n_instruments"], g)
    assert len(insts) > 200 and len({i.symbol for i in insts}) == len(insts)
    assert abs(w.mean() - 1.0) < 1e-12


def test_instruments_heterogeneous(ds):
    big, _, _ = syn._make_instruments(250, np.random.default_rng(3))
    assert {i.tick_size for i in big} <= set(syn._TICK_SIZES)
    assert len({i.lot_size for i in big}) > 3
    assert any(i.multiplier == i.lot_size > 1 for i in big)
    assert all(20 * 0.9 <= i.base_price <= 50000 * 1.1 for i in big)


def test_ticks_valid(ds):
    opens = syn._session_opens(2)
    # weekday check on both session opens (Mon 2026-01-05, Tue 2026-01-06)
    assert [(int(o) // DAY + 3) % 7 for o in opens] == [0, 1]
    for inst in ds.instruments:
        t = ds.ticks[inst.symbol]
        n = len(t["timestamps"])
        assert n > 0
        assert all(len(a) == n for a in t.values())
        assert t["timestamps"].dtype == np.int64
        assert np.all(np.diff(t["timestamps"]) >= 0)
        inside = np.zeros(n, bool)
        for o in opens:
            inside |= (t["timestamps"] >= o) & (t["timestamps"] < o + syn.SESSION_LEN_NS)
        assert inside.all()
        for f in ("ltp", "bid", "ask"):
            k = t[f] / inst.tick_size
            assert np.all(np.abs(k - np.rint(k)) <= 1e-9 * np.maximum(1, np.abs(k)))
            assert np.all(t[f] > 0)
        assert np.all(t["bid"] < t["ask"])
        assert np.all((t["ltp"] >= t["bid"]) & (t["ltp"] <= t["ask"]))
        for f in ("ltq", "bid_qty", "ask_qty", "oi"):
            assert np.all(t[f] > 0)
            assert np.all(t[f] % inst.lot_size == 0)
        assert np.allclose(t["buy_qty_delta"] + t["sell_qty_delta"], t["ltq"])


def test_overnight_gap_present(ds):
    opens = syn._session_opens(2)
    gaps = 0
    for inst in ds.instruments:
        t = ds.ticks[inst.symbol]
        ts = t["timestamps"]
        d1 = ts >= opens[1]
        gaps += abs(t["ltp"][d1][0] - t["ltp"][~d1][-1]) > 0
    assert gaps > 0


def test_liquidity_heavy_tailed():
    d = syn.generate(n_instruments=60, days=1, avg_ticks_per_sec=0.2, seed=5, timeframes=("1d",))
    n = np.array([len(t["timestamps"]) for t in d.ticks.values()])
    assert n.max() > 4 * np.median(n)


@pytest.mark.parametrize("tf", syn.TIMEFRAMES)
def test_bar_reconciliation(ds, tf):
    step = syn._TF_NS[tf]
    opens = syn._session_opens(2)
    for inst in ds.instruments:
        t = ds.ticks[inst.symbol]
        b = ds.bars[tf][inst.symbol]
        ts = t["timestamps"]
        # independent: expected non-empty windows from the full window grid
        grid = []
        for o in opens:
            starts = np.arange(o, o + syn.SESSION_LEN_NS, step)
            ends = np.minimum(starts + step, o + syn.SESSION_LEN_NS)
            grid += list(zip(starts, ends))
        expected = [(s, e) for s, e in grid
                    if np.searchsorted(ts, e, "left") > np.searchsorted(ts, s, "left")]
        assert len(b["ts_open"]) == len(expected)
        assert list(b["ts_open"]) == [s for s, _ in expected]
        assert list(b["ts_close"]) == [e for _, e in expected]
        assert np.array_equal(b["timestamps"], b["ts_open"])
        for i, (s, e) in enumerate(expected):
            lo, hi = np.searchsorted(ts, s, "left"), np.searchsorted(ts, e, "left")
            p = t["ltp"][lo:hi]
            assert b["open"][i] == p[0] and b["close"][i] == p[-1]
            assert b["high"][i] == p.max() and b["low"][i] == p.min()
            assert b["volume"][i] == pytest.approx(t["ltq"][lo:hi].sum())


def test_last_hour_truncated(ds):
    b = ds.bars["1h"][ds.instruments[0].symbol]
    lens = (b["ts_close"] - b["ts_open"]) // (60 * syn.NS)
    assert set(lens) <= {60, 15}


def test_5m_equals_reaggregated_1m(ds):
    for inst in ds.instruments:
        m1, m5 = ds.bars["1m"][inst.symbol], ds.bars["5m"][inst.symbol]
        key = m1["ts_open"]
        bucket = []
        for o in m5["ts_open"]:
            sel = (key >= o) & (key < o + 300 * syn.NS)
            bucket.append((m1["open"][sel][0], m1["high"][sel].max(), m1["low"][sel].min(),
                           m1["close"][sel][-1], m1["volume"][sel].sum()))
        got = np.array(bucket)
        want = np.column_stack([m5[k] for k in ("open", "high", "low", "close", "volume")])
        assert np.allclose(got, want)


def test_aggregate_rejects_out_of_session():
    t = {"timestamps": np.array([0], dtype=np.int64), "ltp": np.array([1.0]),
         "ltq": np.array([1.0])}
    with pytest.raises(ValueError, match="outside session"):
        syn.aggregate_bars(t, "1m", syn.IST_OFFSET_NS, syn.SESSION_OPEN_LOCAL_NS)


def test_save_load_roundtrip_and_tamper(ds, tmp_path):
    syn.save(ds, tmp_path / "ds")
    back = syn.load(tmp_path / "ds")
    assert back.instruments == ds.instruments
    assert back.manifest == ds.manifest
    for sym in ds.ticks:
        for k, a in ds.ticks[sym].items():
            assert np.array_equal(a, back.ticks[sym][k]) and a.dtype == back.ticks[sym][k].dtype
    for tf in ds.bars:
        for sym in ds.bars[tf]:
            for k, a in ds.bars[tf][sym].items():
                assert np.array_equal(a, back.bars[tf][sym][k])

    flat = syn._flat(ds.ticks, ds.bars)
    flat["ticks/SYN000/ltp"] = flat["ticks/SYN000/ltp"].copy()
    flat["ticks/SYN000/ltp"][0] += 1.0
    np.savez(tmp_path / "ds" / "data.npz", **flat)
    with pytest.raises(ValueError, match=r"ticks/SYN000/ltp.*sha256"):
        syn.load(tmp_path / "ds")


def test_load_tier_caches(tmp_path, monkeypatch):
    monkeypatch.setitem(syn.TIERS, "S", dict(n_instruments=4, days=1, avg_ticks_per_sec=0.05))
    a = syn.load_tier("S", str(tmp_path))
    b = syn.load_tier("S", str(tmp_path))  # from cache, verified
    assert a.manifest == b.manifest


def test_engine_accepts_data(ds):
    import raptorbt

    class NoOp(raptorbt.Strategy):
        def on_bar(self, ctx):
            pass

    syms = [i.symbol for i in ds.instruments[:3]]
    data = {s: {k: ds.bars["1m"][s][k]
                for k in ("timestamps", "open", "high", "low", "close", "volume")} for s in syms}
    res = raptorbt.run_portfolio_strategy(NoOp(), data, bar_stamp="close")
    assert res is not None

    class NoOpTick(raptorbt.Strategy):
        pass

    t = {k: ds.ticks[syms[0]][k] for k in syn.TICK_FIELDS}
    assert raptorbt.run_tick_strategy(NoOpTick(), {syms[0]: t}) is not None
