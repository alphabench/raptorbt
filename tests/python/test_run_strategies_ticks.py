"""``run_strategies`` over trade ticks, quotes and order-book depth."""

import numpy as np
import pytest

from raptorbt._raptorbt import MarketData

NS = 1_000_000_000


def tick_arrays(n=6, **over):
    out = {
        "timestamps": (np.arange(n) * NS).astype(np.int64),
        "ltp": 100.0 + np.arange(n, dtype=np.float64),
        "bid": np.full(n, 99.0),
        "ask": np.full(n, 101.0),
    }
    out.update(over)
    return out


def depth_arrays(n=3, levels=4, **over):
    out = {
        "timestamps": (np.arange(n) * NS).astype(np.int64),
        "bid_prices": np.full((n, levels), 99.0),
        "bid_sizes": np.full((n, levels), 10.0),
        "ask_prices": np.full((n, levels), 101.0),
        "ask_sizes": np.full((n, levels), 10.0),
    }
    out.update(over)
    return out


# -- MarketData ---------------------------------------------------------------


def test_market_data_takes_ticks_and_depth_for_a_symbol_without_bars():
    MarketData(["AAA"], [None], [0], [tick_arrays()], [depth_arrays()])


@pytest.mark.parametrize(
    "ticks, depth, needle",
    [
        (tick_arrays(ltp=np.ones(3)), None, "ticks['AAA']: column `ltp`"),
        (tick_arrays(oi=np.ones(2)), None, "ticks['AAA']: column `oi`"),
        (tick_arrays(timestamps=np.array([5, 4, 6, 7, 8, 9], dtype=np.int64)), None,
         "ticks['AAA']: timestamps must not decrease"),
        ({"timestamps": np.arange(3, dtype=np.int64)}, None, "ticks['AAA']: missing 'ltp'"),
        (tick_arrays(), depth_arrays(bid_sizes=np.ones((3, 2))), "depth['AAA']: 'bid_prices' has 4 levels"),
        (tick_arrays(), depth_arrays(ask_prices=np.ones((2, 4))), "depth['AAA']: 'ask_prices' has 2 rows"),
        (tick_arrays(), depth_arrays(bid_prices=np.ones(3)), "depth['AAA']: 'bid_prices' must be a 2-D"),
        (tick_arrays(), {"timestamps": np.arange(2, dtype=np.int64)}, "depth['AAA']: missing 'bid_prices'"),
    ],
)
def test_market_data_errors_name_the_symbol_and_the_field(ticks, depth, needle):
    with pytest.raises(ValueError) as err:
        MarketData(["AAA"], [None], [0], [ticks], [depth])
    assert needle in str(err.value)


def test_market_data_refuses_a_symbol_with_no_data():
    with pytest.raises(ValueError, match="'AAA' has neither bars nor ticks"):
        MarketData(["AAA"], [None], [0])


# -- run_strategies input validation ------------------------------------------

from raptorbt import BacktestConfig, Strategy, run_strategies  # noqa: E402


class Idle(Strategy):
    pass


def call(**kw):
    kw.setdefault("ticks", {"AAA": tick_arrays()})
    return run_strategies([Idle()], **kw)


def test_a_tick_run_with_no_hooks_runs_and_trades_nothing():
    (res,) = call()
    assert res.result.trades() == []


@pytest.mark.parametrize(
    "kw, needle",
    [
        ({"ticks": {"AAA": {"timestamps": np.arange(3, dtype=np.int64)}}}, "ticks['AAA']: missing 'ltp'"),
        ({"ticks": {"AAA": {"ltp": np.ones(3)}}}, "ticks['AAA']: missing 'timestamps'"),
        ({"ticks": {"AAA": tick_arrays(bid=np.ones(2))}}, "ticks['AAA']: 'bid' has length 2, expected 6"),
        ({"ticks": {"AAA": tick_arrays(ltp=np.ones((6, 2)))}}, "ticks['AAA']: 'ltp' must be 1-D"),
        ({"ticks": {"AAA": tick_arrays(oi=["x"] * 6)}}, "ticks['AAA']: 'oi' is not a numeric array"),
        ({"ticks": {"AAA": [1, 2, 3]}}, "ticks['AAA']: expected a dict"),
        ({"ticks": {"AAA": tick_arrays(timestamps=np.array([3, 2, 1, 4, 5, 6]))}},
         "ticks['AAA']: timestamps must not decrease"),
        ({"ticks": [1]}, "ticks must be a dict"),
        ({"ticks": {}}, "ticks must contain at least one symbol"),
        ({"depth": {"BBB": depth_arrays()}}, "depth['BBB'] has no ticks['BBB']"),
        ({"depth": {"AAA": depth_arrays(bid_sizes=np.ones((3, 2)))}}, "depth['AAA']: 'bid_prices' has 4 levels"),
        ({"depth": {"AAA": {"timestamps": np.arange(2, dtype=np.int64)}}}, "depth['AAA']: missing 'bid_prices'"),
        ({"depth": {"AAA": depth_arrays(ask_sizes=np.ones(3))}}, "depth['AAA']: 'ask_sizes' must be 2-D"),
        ({"depth": {"AAA": depth_arrays(ask_prices=np.ones((2, 4)))}}, "depth['AAA']: 'ask_prices' has 2 rows"),
        ({"primary_bars": (5,)}, "primary_bars must be (step, unit)"),
        ({"primary_bars": ("5", "m")}, "primary_bars must be (step, unit)"),
        ({"primary_bars": (5, "parsecs")}, "primary_bars (5, 'parsecs')"),
        ({"ticks": None, "data": None}, "data must contain at least one symbol"),
    ],
)
def test_bad_tick_input_is_refused_naming_the_symbol_and_field(kw, needle):
    with pytest.raises(ValueError) as err:
        call(**kw)
    assert needle in str(err.value)


def test_primary_bars_need_ticks():
    bars = {
        "BBB": {
            "timestamps": np.arange(3, dtype=np.int64) * 60 * NS,
            **{k: np.ones(3) for k in ("open", "high", "low", "close", "volume")},
        }
    }
    with pytest.raises(ValueError, match="primary_bars aggregates trade prints"):
        run_strategies([Idle()], bars, primary_bars=(5, "m"), bar_stamp="close")


def test_an_atr_stop_on_a_tick_symbol_is_refused_up_front():
    cfg = BacktestConfig(initial_capital=1e6)
    cfg.set_atr_stop(2.0, 14)
    with pytest.raises(ValueError, match=r"ATR-based stop/target configured \(ATR period 14, symbol 'AAA'\)"):
        call(config=cfg)


def test_bars_and_ticks_can_share_a_call_and_a_symbol_may_have_both():
    bars = {
        s: {
            "timestamps": np.arange(4, dtype=np.int64) * 60 * NS,
            **{k: np.full(4, 100.0) for k in ("open", "high", "low", "close", "volume")},
        }
        for s in ("AAA", "BBB")
    }
    res = run_strategies(
        [(Idle(), ["AAA"]), (Idle(), ["BBB"]), Idle()],
        bars,
        ticks={"AAA": tick_arrays(), "CCC": tick_arrays()},
        bar_stamp="close",
    )
    assert [len(r.per_instrument) for r in res] == [1, 1, 3]
    assert [p.symbol for p in res[2].per_instrument] == ["AAA", "BBB", "CCC"]


# -- equivalence with run_tick_strategy ----------------------------------------
#
# The strategies are the scripted recorders of the frozen runner fixtures, so
# every hook a tick strategy can get is logged with the state visible at that
# instant (account, positions, working orders, indicators, cache, ctx.book,
# best bid/ask, last price) and compared along with the results.

import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import raptorbt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent / "golden"))
import runner_fixtures as rf  # noqa: E402

DATASETS = rf.load()["datasets"]
S0, S1, S2 = rf.S0, rf.S1, rf.S2
PRIMARY = (1, "m")


def feed():
    ticks = {s: rf._thaw_arrays(DATASETS["ticks"][s]) for s in (S0, S1, S2)}
    depth = {s: rf._thaw_arrays(DATASETS["depth"][s]) for s in (S0, S1, S2)}
    return ticks, depth


def config(margin=False):
    return BacktestConfig(initial_capital=1_000_000.0, fees=0.0005)


# (hook that drives the rhythm, every, enter_at, exit_at, frac, extras, symbols)
_VARIANTS = [
    ("on_trade_tick", 9, 2, 6, 0.30, [], None),
    ("on_trade_tick", 7, 1, 4, 0.20, [["on_quote", 5, 2]], None),
    ("on_trade_tick", 11, 3, 8, 0.25, [["on_order_book", 3, 1]], [S0, S2]),
    ("on_bar", 2, 0, 1, 0.20, [["on_quote", 6, 1]], None),
    ("on_trade_tick", 13, 2, 9, 0.35, [["on_bar", 2, 1]], [S1]),
    ("on_quote", 8, 2, 5, 0.15, [], [S0, S1]),
    ("on_trade_tick", 6, 1, 3, 0.10, [["on_order_book", 4, 0], ["on_quote", 7, 3]], [S1, S2]),
    ("on_order_book", 3, 1, 2, 0.20, [], [S0]),
    ("on_trade_tick", 10, 4, 7, 0.30, [], [S0, S1, S2]),
    ("on_trade_tick", 5, 2, 4, 0.12, [["on_bar", 3, 0]], [S2]),
]


def scripted(i):
    """Strategy ``i`` and the symbols it trades; fresh each call."""
    drive, every, enter_at, exit_at, frac, extra, symbols = _VARIANTS[i % len(_VARIANTS)]
    args = {
        "drive": drive, "every": every, "enter_at": enter_at, "exit_at": exit_at,
        "frac": frac, "extra": extra,
        "subs": [[2, "m"], [20, "tick"]] if i % 3 == 0 else [],
        "inds": [["sma", 2 + i % 3]] if i % 2 == 0 else [],
        "comp_inds": [[0, "sma", 2]] if i % 3 == 0 else [],
    }
    rec = rf.Recorder(script=rf.make_script({"name": "legacy", "args": args}))
    rec.tick_surface = True
    return rec, symbols


def observed(rec, result):
    return json.dumps(
        {"hooks": rec.hooks, "result": rf.digest_result(result)}, sort_keys=True, default=str
    )


def alone(i, **kw):
    """Strategy ``i`` in a ``run_tick_strategy`` call of its own."""
    ticks, depth = feed()
    rec, symbols = scripted(i)
    keep = [s for s in ticks if symbols is None or s in symbols]
    res = raptorbt.run_tick_strategy(
        rec,
        {s: ticks[s] for s in keep},
        config=config(),
        depth={s: depth[s] for s in keep},
        primary_bars=PRIMARY,
        **kw,
    )
    return observed(rec, res)


def together(order, **kw):
    """The strategies ``order`` in one ``run_strategies`` call; their records."""
    ticks, depth = feed()
    recs = [scripted(i) for i in order]
    results = run_strategies(
        [(rec, symbols) if symbols else rec for rec, symbols in recs],
        ticks=ticks,
        depth=depth,
        primary_bars=PRIMARY,
        config=config(),
        **kw,
    )
    return {i: observed(rec, res) for i, (rec, _), res in zip(order, recs, results)}


@pytest.fixture(scope="module")
def references():
    return {i: alone(i) for i in range(len(_VARIANTS))}


def test_the_scripted_strategies_actually_trade_and_hit_every_tick_hook(references):
    hooks = set()
    for rec in references.values():
        hooks |= {e["hook"] for e in json.loads(rec)["hooks"]}
    assert {"on_trade_tick", "on_quote", "on_order_book", "on_bar", "on_composite_bar",
            "on_order_filled", "on_position_opened", "on_position_closed"} <= hooks
    assert sum(len(json.loads(r)["result"]["trades"]) for r in references.values()) > 10


def test_many_strategies_each_equal_their_own_run_tick_strategy(references):
    order = list(range(len(_VARIANTS)))
    got = together(order)
    for i in order:
        assert got[i] == references[i], f"strategy {i} diverged from its own run_tick_strategy"


def test_a_strategy_is_unaffected_by_the_others_and_by_their_order(references):
    ids = list(range(len(_VARIANTS)))
    rng = random.Random(7)
    for _ in range(3):
        rng.shuffle(ids)
        got = together(ids)
        for i in ids:
            assert got[i] == references[i], f"order {ids}: strategy {i}"
    # Any subset, too: the others' absence changes nothing.
    assert together([4, 1])[4] == references[4]
    assert together([9])[9] == references[9]


def test_sharing_the_aggregators_and_indicators_changes_nothing(references):
    order = list(range(len(_VARIANTS)))
    private = together(order, share=False)
    shared = together(order, share=True)
    assert private == shared
    assert all(private[i] == references[i] for i in order)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 7, None])
def test_batches_change_nothing(batch_size, references):
    order = list(range(len(_VARIANTS)))
    got = together(order, batch_size=batch_size)
    assert all(got[i] == references[i] for i in order)


def test_account_kinds_and_directions_apply_to_every_strategy_as_in_a_single_call():
    kw = dict(account_type="margin", leverage=3.0, oms_type="hedging", directions={S2: -1})
    want = {i: alone(i, **kw) for i in (0, 4, 9)}
    ticks, depth = feed()
    recs = {i: scripted(i) for i in want}
    results = run_strategies(
        [(rec, symbols) if symbols else rec for rec, symbols in recs.values()],
        ticks=ticks, depth=depth, primary_bars=PRIMARY, config=config(), **kw,
    )
    for (i, (rec, _)), res in zip(recs.items(), results):
        assert observed(rec, res) == want[i]


# -- bars and ticks in one call ------------------------------------------------


def bar_feed():
    return {
        s: {k: v for k, v in rf._bars_for(
            {"data": {"symbols": [s], "n": 40}}, DATASETS)[s].items()}
        for s in (S0, S1)
    }


def test_bar_symbols_and_tick_symbols_in_one_call_each_match_their_own_runner():
    bars = bar_feed()
    ticks, depth = feed()
    bar_rec = rf.Recorder(script=rf.make_script({"name": "legacy", "args": {
        "drive": "on_bar", "every": 6, "enter_at": 1, "exit_at": 4, "frac": 0.2}}))
    tick_rec, _ = scripted(0)
    kw = dict(bar_stamp="open", bar_ns=60 * NS)
    got = run_strategies(
        [(bar_rec, [S0, S1]), (tick_rec, [S2])], bars, config=config(),
        ticks={S2: ticks[S2]}, depth={S2: depth[S2]}, primary_bars=PRIMARY, **kw,
    )
    ref_bar = rf.Recorder(script=bar_rec.script)
    want_bar = raptorbt.run_portfolio_strategy(ref_bar, bars, config=config(), **kw)
    ref_tick, _ = scripted(0)
    want_tick = raptorbt.run_tick_strategy(
        ref_tick, {S2: ticks[S2]}, config=config(), depth={S2: depth[S2]}, primary_bars=PRIMARY
    )
    assert observed(bar_rec, got[0]) == observed(ref_bar, want_bar)
    assert observed(tick_rec, got[1]) == observed(ref_tick, want_tick)


class Counter(Strategy):
    """Counts what a symbol with both bars and ticks delivers."""

    def __init__(self):
        super().__init__()
        self.seen = []
        self.sizes = []

    def on_bar(self, ctx):
        self.seen.append(("bar", ctx.symbol, ctx.bar_close_ts, ctx.idx))

    def on_trade(self, ctx, tick):
        self.seen.append(("trade", ctx.symbol, tick.timestamp, ctx.idx))
        self.sizes.append(len(ctx.series()["timestamps"]))

    def on_quote(self, ctx, quote):
        self.seen.append(("quote", ctx.symbol, quote.timestamp, ctx.idx))


def test_a_symbol_with_bars_and_ticks_sees_one_tape_in_time_order():
    ticks, _ = feed()
    t = ticks[S0]
    first = int(t["timestamps"][0])
    n_bars = 8
    bars = {
        S0: {
            "timestamps": first + np.arange(n_bars, dtype=np.int64) * 20 * NS,
            **{k: np.full(n_bars, v) for k, v in
               dict(open=100.0, high=101.0, low=99.0, close=100.5, volume=10.0).items()},
        }
    }
    strat = Counter()
    run_strategies([strat], bars, ticks={S0: t}, bar_stamp="close", config=config())
    kinds = [e[0] for e in strat.seen]
    assert kinds.count("bar") == n_bars
    assert kinds.count("trade") == int((t["ltp"] > 0).sum())
    assert kinds.count("quote") == int(((t["bid"] > 0) & (t["ask"] > 0)).sum())
    times = [e[2] for e in strat.seen]
    assert times == sorted(times)
    # One ordinal per instrument, counting every event.
    assert [e[3] for e in strat.seen] == list(range(len(strat.seen)))


def test_ctx_series_of_a_tick_symbol_stops_at_the_current_print():
    ticks, _ = feed()
    strat = Counter()
    run_strategies([(strat, [S1])], ticks={S1: ticks[S1]}, config=config())
    t = ticks[S1]
    prints = np.flatnonzero(t["ltp"] > 0)
    # Rows up to and including the row that printed.
    assert strat.sizes == [int(r) + 1 for r in prints]


class BookReader(Strategy):
    """Overrides only ``on_trade_tick``: the book, bids and asks it reads were
    seen by hooks it does not have, so they must be kept for it anyway."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def on_trade(self, ctx, tick):
        book = ctx.book
        self.seen.append((
            tuple(tick), ctx.tick is tick, ctx.quote, ctx.best_bid, ctx.best_ask, ctx.last_price,
            None if book is None else tuple(book),
        ))


def test_context_views_are_kept_for_a_strategy_without_the_matching_hooks():
    ticks, depth = feed()
    legacy, shared = BookReader(), BookReader()
    raptorbt.run_tick_strategy(legacy, ticks, depth=depth, config=config())
    run_strategies([shared], ticks=ticks, depth=depth, config=config())
    assert shared.seen == legacy.seen
    assert any(row[6] is not None for row in shared.seen), "the book must have been read"
    assert any(row[3] is not None for row in shared.seen)
