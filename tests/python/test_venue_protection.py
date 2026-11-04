"""What a venue holds as protection is what a batch run would hold.

* After an add, the protective orders cover the whole position, not the size they were placed for.
* An add to a sold option leg that its hedge still covers is filled, not refused as unfillable.
"""
from test_venue_parity import PATHS, T0, bars, run

from raptorbt import BacktestConfig
from raptorbt.strategy import orders as O


def _sizes(out):
    return {p: sorted((round(t.size, 6), round(t.exit_price, 6)) for t in r.result.trades()) for p, r in out.items()}


def test_a_stop_placed_for_ten_closes_twenty_after_an_add():
    # long 10 with a stop at 95, 10 more added with no level of its own; bar 5 trades down to 90
    rows = [(100, 101, 99, 100)] * 5 + [(100, 100, 90, 92)] + [(92, 93, 91, 92)] * 2
    script = {
        1: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0, stop_price=95.0)),
        3: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0)),
    }
    out = run(script, {"X": bars(rows, T0)}, lambda: BacktestConfig(initial_capital=1e6, fees=0.0), paths=PATHS)
    got = _sizes(out)
    assert got["batch"] == [(20.0, 95.0)], got
    assert got["node"] == got["batch"], got
    assert got["paper"] == got["batch"], got


def test_an_add_to_a_sold_option_leg_that_its_hedge_still_covers_is_filled():
    # hedge: 100 of the 100 call bought; sold: 50 of the 105 call, then 50 more. The hedge covers
    # 100 sold, so under defined_risk the add needs no margin of its own and is not unfillable
    from _chain_fixture import EXP1, N_BARS, U, symbol, universe

    from raptorbt import LiveNode, MarginModel, Strategy, run_strategies
    from raptorbt.execution import PaperExecution

    sold, hedge = symbol(105.0, "call", EXP1), symbol(100.0, "call", EXP1)

    class Spread(Strategy):
        def on_bar(self, ctx):
            if ctx.symbol != U:
                return
            if ctx.idx == 3:
                self.submit_order(O.Market(side="buy", units=100.0, instrument=hedge), client_id="hedge")
            if ctx.idx == 4:
                self.submit_order(O.Market(side="sell", units=50.0, instrument=sold), client_id="sold")
            if ctx.idx == 6:
                self.submit_order(O.Market(side="sell", units=50.0, instrument=sold), client_id="add")

    data, inst = universe()
    kw = dict(instruments=inst, account_type="margin", leverage=1.0, bar_stamp="close")
    for mode in ("batch", "node", "paper"):
        cfg = BacktestConfig(initial_capital=1e6, fees=0.0, margin_model=MarginModel.defined_risk())
        if mode == "batch":
            result = run_strategies([Spread()], data, cfg, **kw)[0]
        else:
            ex = PaperExecution() if mode == "paper" else None
            node = LiveNode([Spread()], list(data), cfg, execution=ex, **kw)
            for i in range(N_BARS):
                node.push_group(
                    int(data[U]["timestamps"][i]),
                    {x: tuple(float(data[x][c][i]) for c in ("open", "high", "low", "close", "volume")) for x in data},
                )
            result = node.finish()[0]
        got = [(o.client_id, o.status, o.filled_qty) for o in result.result.orders()]
        assert got == [("hedge", "filled", 100.0), ("sold", "filled", 50.0), ("add", "filled", 50.0)], (mode, got)
