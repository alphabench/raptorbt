"""``ctx.indicators.expr``: an expression on the shared indicator bank.

An expression registered by a strategy is a cell of the same bank the built-in indicators use, so
it must show what they show: the value after the bar once the strategy reveals it, the value
before the bar until then, the numbers ``raptorbt.expr(...).eval`` gives over the same bars in a
batch run and on a node, and the same numbers after a restart. Two strategies with one canonical
expression on one stream share one cell. The single-instrument runners without the shared runtime
refuse it with a typed error rather than run it differently.
"""

import numpy as np
import pytest

import raptorbt as rb
from raptorbt import BacktestConfig, Indicator, LiveNode, Strategy, run_strategies, run_strategy_backtest
from raptorbt import journal as rj
from raptorbt.errors import ExpressionError, RaptorBTError
from raptorbt.strategy import orders as O

MIN = 60 * 10**9
T0 = 1_767_225_600 * 10**9
OHLCV = ("open", "high", "low", "close", "volume")


def bars(n=120, seed=7):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return {
        "timestamps": T0 + np.arange(1, n + 1, dtype=np.int64) * MIN,
        "open": open_,
        "high": np.maximum(open_, close) + 0.3,
        "low": np.minimum(open_, close) - 0.3,
        "close": close,
        "volume": np.round(rng.uniform(500.0, 1500.0, n)),
    }


def config(**kw):
    return BacktestConfig(initial_capital=1_000_000.0, fees=0.0, **kw)


def feed(node, data, lo, hi):
    syms = list(data)
    for i in range(lo, hi):
        node.push_group(
            int(data[syms[0]]["timestamps"][i]),
            {s: tuple(float(data[s][c][i]) for c in OHLCV) for s in syms},
        )


def run(strategy, data, mode="batch", **kw):
    """One strategy through the shared runtime (``batch``) or a node fed the same groups."""
    cfg = config(**kw)
    if mode == "batch":
        return run_strategies([strategy], data, cfg, bar_stamp="close")[0]
    node = LiveNode([strategy], list(data), cfg, bar_stamp="close")
    feed(node, data, 0, len(data[next(iter(data))]["timestamps"]))
    return node.finish()[0]


EXPRS = {
    "number": "ema(close, 3) - sma(close, 5) + sma(volume, 4) * 0.01",
    "flag": "zscore(close, 10) < -0.5 and adx(7) < 25",
    "volume_flag": "sma(close, 3) > ema(close, 5) and not (volume < 600)",
    "ratio": "(close - open) / (high - low) * sma(volume, 2)",
}
BOOLEAN = {"flag", "volume_flag"}


class Watch(Strategy):
    """Registers EXPRS on ASSET and keeps what every handle shows on every bar."""

    def __init__(self, exprs=None, **kw):
        super().__init__()
        self.exprs = EXPRS if exprs is None else exprs
        self.kw = kw
        self.rows = []

    def on_start(self, ctx):
        self.h = {k: ctx.indicators.expr(v, instrument="ASSET", **self.kw) for k, v in self.exprs.items()}

    def on_bar(self, ctx):
        self.rows.append((ctx.idx, {k: (h.ready, h.value) for k, h in self.h.items()}))


# -- the numbers -------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["batch", "node"])
def test_a_handle_equals_the_batch_evaluation_at_every_bar(mode):
    data = {"ASSET": bars()}
    s = Watch()
    run(s, data, mode)
    assert len(s.rows) == len(data["ASSET"]["close"])
    for key, text in EXPRS.items():
        program = rb.expr(text)
        want = program.eval(data["ASSET"])
        for i, row in s.rows:
            ready, value = row[key]
            assert ready == (i >= program.warmup), (key, i)
            if not ready or np.isnan(want[i]):
                assert value is None, (key, i, value)
            elif key in BOOLEAN:
                assert value is bool(want[i]), (key, i, value, want[i])
            else:
                assert isinstance(value, float) and value == want[i], (key, i, value, want[i])
    ready_bars = {key: sum(r[key][0] for _, r in s.rows) for key in EXPRS}
    assert all(n > 50 for n in ready_bars.values()), ready_bars


def test_live_after_warm_up_bars_gives_the_numbers_of_a_backtest():
    data = {"ASSET": bars()}
    cut = 70
    batch = Watch()
    run(batch, data, "batch")

    head = {"ASSET": {k: v[:cut] for k, v in data["ASSET"].items()}}
    live = Watch()
    node = LiveNode([live], ["ASSET"], config(), bar_stamp="close", warmup_bars=head)
    feed(node, data, cut, len(data["ASSET"]["close"]))
    node.finish()
    assert [r for _, r in live.rows] == [r for _, r in batch.rows]


def test_sma_by_expression_is_bit_equal_to_the_built_in_one():
    class Pair(Strategy):
        def __init__(self):
            super().__init__()
            self.rows = []

        def on_start(self, ctx):
            self.a = ctx.indicators.sma(instrument="ASSET", period=5)
            self.b = ctx.indicators.expr("sma(close, 5)", instrument="ASSET")
            self.c = ctx.indicators.sma(instrument="ASSET", period=3, timeframe="5m")
            self.d = ctx.indicators.expr("sma(close, 3)", instrument="ASSET", timeframe="5m")

        def on_bar(self, ctx, event):
            self.rows.append((self.a.ready, self.a.value, self.b.ready, self.b.value,
                              self.c.ready, self.c.value, self.d.ready, self.d.value))

    s = Pair()
    run(s, {"ASSET": bars()})
    assert any(r[0] for r in s.rows) and any(r[4] for r in s.rows)
    for ra, va, rb_, vb, rc, vc, rd, vd in s.rows:
        assert (ra, va) == (rb_, vb)
        assert (rc, vc) == (rd, vd)


def test_an_expression_on_a_timeframe_reads_the_composite_bars_with_their_volume():
    class Composite(Strategy):
        def __init__(self):
            super().__init__()
            self.volumes, self.seen = [], []

        def on_start(self, ctx):
            # The expression's stream is not delivered to on_bar unless the strategy subscribes.
            self.subscribe("ASSET", timeframe="5m")
            self.h = ctx.indicators.expr("sma(volume, 2)", instrument="ASSET", timeframe="5m")

        def on_bar(self, ctx, event):
            if event.timeframe is None:
                return
            self.volumes.append(event.volume)
            self.seen.append(self.h.value)

    s = Composite()
    run(s, {"ASSET": bars()})
    assert len(s.volumes) > 10
    assert s.seen[0] is None
    for k in range(1, len(s.volumes)):
        assert s.seen[k] == pytest.approx((s.volumes[k - 1] + s.volumes[k]) / 2.0, rel=1e-12)


# -- what a hook sees -----------------------------------------------------------------


@pytest.mark.parametrize("mode", ["batch", "node"])
def test_a_hook_at_the_open_sees_the_value_before_the_bar_as_a_built_in_does(mode):
    class Probe(Strategy):
        def __init__(self):
            super().__init__()
            self.seen, self.at_hook = {}, None

        def on_start(self, ctx):
            self.sma = ctx.indicators.sma(instrument="ASSET", period=5)
            self.ex = ctx.indicators.expr("sma(close, 5)", instrument="ASSET")
            self.flag = ctx.indicators.expr("close > sma(close, 5)", instrument="ASSET")

        def now(self):
            return (self.sma.value, self.ex.value, self.flag.value)

        def on_bar(self, ctx):
            self.seen[ctx.idx] = self.now()
            if ctx.idx == 10:
                self.submit_order(O.Market(side="buy", units=1.0), client_id="probe")

        def on_order_event(self, ctx, e):
            if e.kind == "order_filled" and e.client_order_id == "probe":
                self.at_hook = (max(self.seen), self.now())

    s = Probe()
    run(s, {"ASSET": bars()}, mode, fill_timing="next_bar_open")
    assert s.at_hook is not None, "the order never filled"
    last, shown = s.at_hook
    # The fill is heard at bar 11's open: bar 10 is the last the strategy has seen.
    assert last == 10
    assert shown == s.seen[10]
    assert shown[0] == shown[1]
    assert s.seen[11] != s.seen[10], "bar 11 changed nothing, so the check proves nothing"


def test_a_strategy_with_no_expression_is_unmoved_by_one_beside_it():
    class Cross(Strategy):
        def on_start(self, ctx):
            self.fast = ctx.indicators.sma(instrument="ASSET", period=3)
            self.slow = ctx.indicators.sma(instrument="ASSET", period=8)

        def on_bar(self, ctx):
            if not (self.fast.ready and self.slow.ready):
                return
            if ctx.position is None and self.fast.value > self.slow.value:
                self.enter(size_frac=0.2)
            elif ctx.position is not None and self.fast.value < self.slow.value:
                self.close_position()

    def trades(result):
        return [(t.entry_time, t.entry_price, t.exit_time, t.exit_price, t.size) for t in result.result.trades()]

    data = {"ASSET": bars(200)}
    alone = run_strategies([Cross()], data, config(), bar_stamp="close")[0]
    beside = run_strategies([Cross(), Watch()], data, config(), bar_stamp="close")[0]
    assert trades(alone) and trades(alone) == trades(beside)


# -- sharing and memory -------------------------------------------------------------


def report(strategies, data=None):
    data = data or {"ASSET": bars(10)}
    node = LiveNode(strategies, list(data), config(), bar_stamp="close")
    # A symbol's rings are allocated on its first push, at the rows the node keeps.
    feed(node, data, 0, 1)
    out = node.memory_report()
    node.finish()
    return out


def test_two_strategies_with_one_canonical_expression_share_one_cell():
    text = {"a": "sma(close, 400) > ema(close, 300)"}
    respelled = {"a": "sma(close,400)>ema(close,300)"}
    one = report([Watch(text)])
    two = report([Watch(text), Watch(respelled)])
    other = report([Watch(text), Watch({"a": "sma(close, 401) > ema(close, 300)"})])
    assert (one["indicator_cells"], one["indicator_registrations"]) == (1, 1)
    assert (two["indicator_cells"], two["indicator_registrations"]) == (1, 2)
    assert two["indicators"] == one["indicators"]
    assert two["engine_per_strategy"] > one["engine_per_strategy"]
    assert (other["indicator_cells"], other["indicator_registrations"]) == (2, 2)
    assert other["indicators"] > one["indicators"]


def test_the_memory_report_counts_what_an_expression_holds():
    small = report([Watch({"a": "sma(close, 10)"})])["indicators"]
    large = report([Watch({"a": "sma(close, 3000)"})])["indicators"]
    # 2990 more bars of window in the kernel's ring, 8 bytes each.
    assert large - small >= 2990 * 8


def test_a_node_keeps_the_history_an_expression_needs():
    short = report([Watch({"a": "sma(close, 5000)"})])["live_rings"]
    long = report([Watch({"a": "sma(close, 12000)"})])["live_rings"]
    assert long > short, "a 12000-bar window needs more than the default 10000 rows of history"


# -- restart ---------------------------------------------------------------------------


def test_a_restored_node_gives_the_expression_values_of_an_uninterrupted_one():
    data = {"ASSET": bars()}
    n = len(data["ASSET"]["close"])
    kw = dict(bar_stamp="close")
    for cut in (1, 30, 61, n - 1):
        journal = rj.Journal({"run": "expr"}, memory=True)
        before = Watch()
        node = LiveNode([before], ["ASSET"], config(), journal=journal, **kw)
        feed(node, data, 0, cut)
        records = [r.canonical() for r in journal.since(0)]

        after = Watch()
        restored = LiveNode.restore(
            None, [after], ["ASSET"], config(), journal=rj.Journal({"run": "expr"}, memory=True),
            records=records, **kw,
        )
        # The replay runs the new strategy over the journaled bars: its handles are rebuilt.
        assert after.rows == before.rows, cut
        assert restored.state_digest() == node.state_digest(), cut

        feed(node, data, cut, n)
        feed(restored, data, cut, n)
        assert after.rows == before.rows, cut
        assert any(r[key][0] for _, r in after.rows for key in EXPRS)


# -- refusals ----------------------------------------------------------------------------


def test_run_strategy_backtest_refuses_an_expression_with_a_typed_error():
    class S(Strategy):
        def on_start(self, ctx):
            ctx.indicators.expr("sma(close, 3)", instrument="ASSET")

        def on_bar(self, ctx):
            pass

    b = bars()
    with pytest.raises(ExpressionError) as e:
        run_strategy_backtest(
            S(), b["timestamps"], b["open"], b["high"], b["low"], b["close"], b["volume"],
            bar_stamp="close",
        )
    assert e.value.code == "expr_unsupported_here"
    assert "run_strategies" in e.value.valid


def test_a_multi_output_expression_needs_a_selection():
    class Bare(Strategy):
        def on_start(self, ctx):
            ctx.indicators.expr("macd(close)", instrument="ASSET")

    with pytest.raises(ExpressionError) as e:
        run(Bare(), {"ASSET": bars()})
    assert e.value.code == "expr_type"
    assert {"macd", "macdsignal", "macdhist"} <= set(e.value.valid)

    class Picked(Strategy):
        def __init__(self):
            super().__init__()
            self.rows = []

        def on_start(self, ctx):
            self.by_name = ctx.indicators.expr("macd(close)", instrument="ASSET", output="macdsignal")
            self.by_text = ctx.indicators.expr("macd(close).macdsignal", instrument="ASSET")

        def on_bar(self, ctx):
            self.rows.append((self.by_name.value, self.by_text.value))

    s = Picked()
    data = {"ASSET": bars()}
    run(s, data)
    want = rb.expr("macd(close).macdsignal").eval(data["ASSET"])
    for i, (a, b) in enumerate(s.rows):
        assert a == b
        assert (a is None) == bool(np.isnan(want[i]))
        assert a is None or a == want[i]
    with pytest.raises(AttributeError):
        s.by_name.values


def test_an_unknown_output_and_an_unknown_instrument_are_refused():
    class Unknown(Strategy):
        def on_start(self, ctx):
            ctx.indicators.expr("macd(close)", instrument="ASSET", output="nope")

    with pytest.raises(ExpressionError) as e:
        run(Unknown(), {"ASSET": bars()})
    assert e.value.code == "expr_type"

    class Elsewhere(Strategy):
        def on_start(self, ctx):
            ctx.indicators.expr("sma(close, 3)", instrument="NOT_IN_THE_RUN")

    with pytest.raises(RaptorBTError) as e:
        run(Elsewhere(), {"ASSET": bars()})
    assert e.value.code == "unknown_symbol"


# -- volume on every path ------------------------------------------------------------------


def test_a_hand_fed_expression_that_reads_volume_is_refused_a_bar_without_it():
    reads = Indicator.expr(rb.expr("sma(volume, 2)")._prog)
    with pytest.raises(RaptorBTError) as e:
        reads.update_bar(1.0, 1.0, 1.0, 1.0)
    assert e.value.code == "missing_field" and e.value.parameter == "volume"
    assert reads.update_count == 0
    assert reads.update_bar(1.0, 1.0, 1.0, 1.0, 10.0) is None
    assert reads.update_bar(1.0, 1.0, 1.0, 1.0, 30.0) == 20.0

    # An expression that does not read it takes the four prices as every indicator does.
    prices = Indicator.expr(rb.expr("sma(close, 2)")._prog)
    prices.update_bar(1.0, 1.0, 1.0, 2.0)
    assert prices.update_bar(1.0, 1.0, 1.0, 4.0) == 3.0


def test_the_single_instrument_runner_feeds_a_hand_registered_expression_its_volume():
    class Hand(Strategy):
        def __init__(self):
            super().__init__()
            self.vals = []

        def on_start(self, ctx):
            self.v = self.register_indicator(Indicator.expr(rb.expr("sma(volume, 3)")._prog))
            self.c = self.register_indicator(Indicator.expr(rb.expr("sma(close, 3)")._prog), stream_id=None)

        def on_bar(self, ctx):
            self.vals.append((self.v.value, self.c.value))

    s = Hand()
    b = bars(40)
    run_strategy_backtest(
        s, b["timestamps"], b["open"], b["high"], b["low"], b["close"], b["volume"],
        bar_stamp="close",
    )
    assert s.vals[0] == (None, None) and s.vals[1] == (None, None)
    for i in range(2, 40):
        v, c = s.vals[i]
        assert v == pytest.approx(b["volume"][i - 2 : i + 1].mean(), rel=1e-12)
        assert c == pytest.approx(b["close"][i - 2 : i + 1].mean(), rel=1e-12)


def test_capabilities_lists_the_expression_indicator():
    streaming = {i["name"] for i in rb.capabilities()["indicators"]["streaming"]}
    assert "expr" in streaming
