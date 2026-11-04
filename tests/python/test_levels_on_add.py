"""An explicit level stays where a broker's would when an add moves the
average entry, and an add that names a level moves it on every path. Every number is computed by hand from the bars written here. Bars are
1 minute, stamped at their close; fees are zero."""

from raptorbt import BacktestConfig
from raptorbt.strategy import orders as O
from test_venue_parity import PATHS, T0, bars, run


def nbo():
    return BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing="next_bar_open")


def closed(out):
    return {p: sorted((round(t.size, 6), round(t.exit_price, 6)) for t in r.result.trades()) for p, r in out.items()}


FLAT = (100, 101, 99, 100)


def test_an_add_with_no_level_leaves_an_explicit_stop_where_it_rests():
    """Long 10 at 100 (open of bar 2) with a stop at 95. Add 10 at 110 (open of bar 3): average 105.
    A broker's stop stays at 95: bar 4's low of 97 does not touch it, bar 5's low of 90 does, at 95
    (bar 5 opens at 98). Trade: 20 x (95 - 105) = -200.
    Before: batch and the in-process node shifted the stop by the average's move to 100 and closed
    at 100 on bar 4 (-100); paper held 95."""
    rows = [FLAT, FLAT, (100, 111, 99, 110), (110, 111, 109, 110), (110, 110, 97, 98), (98, 99, 90, 91), (91, 92, 90, 91)]
    script = {
        1: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0, stop_price=95.0)),
        2: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0)),
    }
    got = closed(run(script, {"X": bars(rows, T0)}, nbo, paths=PATHS))
    assert got == {p: [(20.0, 95.0)] for p in PATHS}, got


def test_an_add_that_names_a_stop_moves_the_stop_on_every_path():
    """As above with the add naming a stop of 102. Bar 4 (low 101, open 110) stops at 102:
    20 x (102 - 105) = -60. Before: paper and a venue kept the first order's 95 and exited at 95 on bar 5 (-200)."""
    rows = [FLAT, FLAT, (100, 111, 99, 110), (110, 111, 109, 110), (110, 110, 101, 103), (103, 104, 94, 95), (95, 96, 94, 95)]
    script = {
        1: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0, stop_price=95.0)),
        2: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0, stop_price=102.0)),
    }
    got = closed(run(script, {"X": bars(rows, T0)}, nbo, paths=PATHS))
    assert got == {p: [(20.0, 102.0)] for p in PATHS}, got


def test_a_first_entry_with_no_level_and_an_add_that_names_one_is_protected_on_every_path():
    """Long 10 at 100 with no stop, then 10 more at 110 naming a stop of 102 (average 105). Bar 4
    (low 101) stops all 20 at 102: 20 x (102 - 105) = -60."""
    rows = [FLAT, FLAT, (100, 111, 99, 110), (110, 111, 109, 110), (110, 110, 101, 103), (103, 104, 94, 95), (95, 96, 94, 95)]
    script = {
        1: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0)),
        2: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0, stop_price=102.0)),
    }
    got = closed(run(script, {"X": bars(rows, T0)}, nbo, paths=PATHS))
    assert got == {p: [(20.0, 102.0)] for p in PATHS}, got
