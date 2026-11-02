"""Account groups: several strategies on one account.

A group of one is the account an isolated strategy has; strategies on one
account draw on one pool of cash and margin, in the order they are listed;
account-level halts and limits cover every strategy on the account.
"""

import numpy as np
import pytest

import raptorbt
from raptorbt import (
    AccountResult,
    AccountSpec,
    BacktestConfig,
    LiveNode,
    Strategy,
    run_strategies,
)
from raptorbt.errors import InvalidParameter
from raptorbt.strategy import accounts as accounts_module
from raptorbt.strategy.accounts import plan_batches

from test_run_strategies_smoke import KW, SYMBOLS, Recorder, fingerprint, make_data, same

MIN = 60_000_000_000
T0 = 1_700_000_000_000_000_000
CAP = 100_000.0


class Scripted(Strategy):
    """Enters on the given bar indices and exits on others; records cash."""

    def __init__(self, enter=(), exit=(), frac=1.0, **kw):
        super().__init__(**kw)
        self.enter_at, self.exit_at, self.frac = set(enter), set(exit), frac
        self.cash = []

    def on_bar(self, ctx):
        self.cash.append(ctx.cash)
        if ctx.idx in self.enter_at and ctx.is_flat:
            self.enter_long(size_frac=self.frac)
        elif ctx.idx in self.exit_at and not ctx.is_flat:
            self.close_position()


def drifting(drift, n=120, symbols=SYMBOLS[:2], seed=0):
    """Bars that fall (or rise) by ``drift`` a bar, plus a little noise."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, s in enumerate(symbols):
        ts = (np.arange(n, dtype=np.int64) + 1) * MIN + T0
        close = 100.0 * np.cumprod(1.0 + drift + rng.normal(0, 0.002, n))
        open_ = np.concatenate([[100.0], close[:-1]])
        out[s] = {
            "timestamps": ts,
            "open": open_,
            "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    return out


def pnl(res):
    return sum(t.pnl for t in res.result.trades())


def refusals(res):
    """Orders the engine refused, whatever the reason."""
    return [o.reject_reason for o in res.result.orders() if o.status == "rejected"]


def isolated(strategy, symbols, data, capital=CAP, **kw):
    cfg = kw.pop("config", None) or BacktestConfig(initial_capital=capital)
    return run_strategies([(strategy, symbols)], data, cfg, **KW, **kw)[0]


# -- a group of one is an isolated account ------------------------------------


@pytest.mark.parametrize(
    "mode",
    [
        dict(account_type="cash"),
        dict(account_type="margin", leverage=5.0),
    ],
)
def test_a_group_of_one_is_bit_identical_to_an_isolated_strategy(mode):
    data = make_data(3, n=300)
    limits = dict(max_drawdown_pct=15.0, max_positions=2, max_daily_loss_pct=4.0)
    cfg = BacktestConfig(initial_capital=CAP, fees=0.001, flatten_on_daily_loss=True, **limits)
    syms = SYMBOLS[:4]
    alone = Recorder()
    want = run_strategies([(alone, syms)], data, cfg, **KW, **mode)[0]
    grouped = Recorder()
    # The call's own capital and limits are replaced by the account's.
    other = BacktestConfig(
        initial_capital=7.0, fees=0.001, flatten_on_daily_loss=True, max_positions=1
    )
    got, accounts = run_strategies(
        [(grouped, syms, {"account": "a"})],
        data,
        other,
        accounts={"a": AccountSpec(initial_capital=CAP, **mode, **limits)},
        **KW,
    )
    same(fingerprint(got[0]), fingerprint(want))
    assert got[0].halted == want.halted and got[0].halted_at == want.halted_at
    assert grouped.rec == alone.rec  # the hooks saw the same equity and cash
    (account,) = accounts.values()
    assert account.halted == want.halted and account.halted_at == want.halted_at
    assert account.equity_curve().tobytes() == want.result.equity_curve().tobytes()
    assert len(want.result.trades()) > 3


# -- one pool of cash ---------------------------------------------------------


def test_the_second_strategy_is_refused_cash_the_first_used_in_the_same_event():
    data = drifting(0.0005, symbols=["AAA"])
    a, b = Scripted(enter=[5]), Scripted(enter=[5])
    (ra, rb), accounts = run_strategies(
        [(a, ["AAA"], {"account": "x"}), (b, ["AAA"], {"account": "x"})],
        data,
        accounts={"x": AccountSpec(initial_capital=CAP)},
        **KW,
    )
    assert len(ra.result.trades()) == 1 and len(rb.result.trades()) == 0
    assert len(refusals(rb)) >= 1
    # Alone, the second would have traded: the pool is why it did not.
    assert len(isolated(Scripted(enter=[5]), ["AAA"], data).result.trades()) == 1
    assert accounts["x"].balance == pytest.approx(CAP + pnl(ra))


def test_the_second_strategy_is_refused_cash_the_first_still_holds_at_a_later_event():
    data = drifting(0.0005, symbols=["AAA", "BBB"])
    a, b = Scripted(enter=[5]), Scripted(enter=[40])
    (ra, rb), _ = run_strategies(
        [(a, ["AAA"], {"account": "x"}), (b, ["BBB"], {"account": "x"})],
        data,
        accounts={"x": AccountSpec(initial_capital=CAP)},
        **KW,
    )
    assert len(ra.result.trades()) == 1
    assert len(rb.result.trades()) == 0 and len(refusals(rb)) >= 1


def test_cash_returns_to_the_pool_when_a_position_closes():
    data = drifting(0.0005, symbols=["AAA", "BBB"])
    a, b = Scripted(enter=[5], exit=[20]), Scripted(enter=[40])
    (ra, rb), _ = run_strategies(
        [(a, ["AAA"], {"account": "x"}), (b, ["BBB"], {"account": "x"})],
        data,
        accounts={"x": AccountSpec(initial_capital=CAP)},
        **KW,
    )
    assert len(ra.result.trades()) == 1 and len(rb.result.trades()) == 1


# -- order, and permutation ---------------------------------------------------


def test_within_an_account_the_listed_order_decides_who_gets_the_cash():
    data = drifting(0.0005, symbols=["AAA"])
    spec = {"x": AccountSpec(initial_capital=CAP)}
    for first, second in ((0, 1), (1, 0)):
        pair = [Scripted(enter=[5]), Scripted(enter=[5])]
        listed = [pair[first], pair[second]]
        run_strategies(
            [(s, ["AAA"], {"account": "x"}) for s in listed], data, accounts=spec, **KW
        )
        # The strategy listed first traded; the one listed second did not.
        assert pair[first].cash[5] == pytest.approx(CAP)
        assert pair[second].cash[5] < CAP / 100


def test_results_do_not_depend_on_the_order_across_accounts_or_of_isolated_strategies():
    data = make_data(5, n=240)
    specs = {
        "g1": AccountSpec(initial_capital=CAP, max_positions=2),
        "g2": AccountSpec(initial_capital=60_000.0, account_type="margin", leverage=3.0),
    }

    def build():
        s = {
            "a": (Recorder(fast=3), SYMBOLS[:3], "g1"),
            "b": (Recorder(fast=4), SYMBOLS[1:4], "g1"),
            "c": (Recorder(fast=5), SYMBOLS[2:], "g2"),
            "d": (Recorder(fast=6), SYMBOLS[:2], "g2"),
            "e": (Recorder(fast=7), SYMBOLS[:], None),
        }
        return s

    def run(order):
        s = build()
        entries = [(s[k][0], s[k][1], {"account": s[k][2]}) for k in order]
        results, accounts = run_strategies(entries, data, accounts=specs, **KW)
        return {k: fingerprint(r) for k, r in zip(order, results)}, accounts

    base, base_accounts = run("abcde")
    # Within a group the order is kept (a before b, c before d); across groups
    # and for the isolated strategy it is free.
    for order in ("cdabe", "ecadb", "acebd", "eacbd"):
        got, accounts = run(order)
        for k in "abcde":
            same(got[k], base[k])
        for name in specs:
            assert accounts[name].balance == base_accounts[name].balance
            assert accounts[name].equity_curve().tobytes() == (
                base_accounts[name].equity_curve().tobytes()
            )
    # Reversing a group changes its results, as the rule says it should.
    got, _ = run("bacde")
    assert repr(got["a"]) != repr(base["a"]) or repr(got["b"]) != repr(base["b"])


def test_isolated_strategies_beside_groups_are_unaffected():
    data = make_data(2, n=200)
    alone = {i: fingerprint(isolated(Recorder(fast=3 + i), SYMBOLS[i:], data)) for i in range(2)}
    entries = [
        (Recorder(fast=3), SYMBOLS[0:], None),
        (Recorder(fast=9), SYMBOLS[1:], "x"),
        (Recorder(fast=4), SYMBOLS[1:], None),
        (Recorder(fast=8), SYMBOLS[:3], "x"),
    ]
    results, _ = run_strategies(
        [(s, y, {"account": n}) for s, y, n in entries],
        data,
        BacktestConfig(initial_capital=CAP),
        accounts={"x": AccountSpec(initial_capital=250_000.0)},
        **KW,
    )
    same(fingerprint(results[0]), alone[0])
    # fast=4 over SYMBOLS[1:] is not one of the references: run it alone.
    same(fingerprint(results[2]), fingerprint(isolated(Recorder(fast=4), SYMBOLS[1:], data)))


# -- halts fan out ------------------------------------------------------------


def test_a_margin_call_halts_every_strategy_on_the_account():
    data = drifting(-0.015, symbols=["AAA", "BBB"])
    busy, idle = Scripted(enter=[2]), Scripted()
    (rbusy, ridle), accounts = run_strategies(
        [(busy, ["AAA"], {"account": "m"}), (idle, ["BBB"], {"account": "m"})],
        data,
        accounts={"m": AccountSpec(initial_capital=50_000.0, account_type="margin", leverage=20.0)},
        **KW,
    )
    assert rbusy.halted and ridle.halted
    assert rbusy.halted_at == ridle.halted_at == accounts["m"].halted_at
    assert accounts["m"].halted
    # The same strategy alone is margin-called too, at the same ordinal.
    alone = isolated(Scripted(enter=[2]), ["AAA"], data, 50_000.0, account_type="margin", leverage=20.0)
    assert alone.halted


def test_the_drawdown_limit_halts_every_strategy_on_the_account():
    data = drifting(-0.008, symbols=["AAA", "BBB"])
    busy, idle = Scripted(enter=[2], frac=0.8), Scripted(enter=[100])
    (rbusy, ridle), accounts = run_strategies(
        [(busy, ["AAA"], {"account": "d"}), (idle, ["BBB"], {"account": "d"})],
        data,
        accounts={"d": AccountSpec(initial_capital=CAP, max_drawdown_pct=5.0)},
        **KW,
    )
    assert rbusy.halted and ridle.halted
    assert rbusy.halted_at == ridle.halted_at == accounts["d"].halted_at
    # The idle one's later entry was refused by the account's halt.
    assert len(ridle.result.trades()) == 0 and len(refusals(ridle)) >= 1


def test_the_daily_loss_limit_is_the_accounts_and_flattens_every_strategy():
    data = drifting(-0.004, n=100, symbols=["AAA", "BBB"])
    cfg = BacktestConfig(initial_capital=1.0, flatten_on_daily_loss=True)
    held = [Scripted(enter=[2], frac=0.4), Scripted(enter=[2], frac=0.4), Scripted(enter=[80])]
    results, accounts = run_strategies(
        [
            (held[0], ["AAA"], {"account": "q"}),
            (held[1], ["BBB"], {"account": "q"}),
            (held[2], ["BBB"], {"account": "q"}),
        ],
        data,
        cfg,
        accounts={"q": AccountSpec(initial_capital=CAP, max_daily_loss_pct=1.0)},
        **KW,
    )
    # Each loss alone is under 1% of the account; together they are not.
    flattened = [r.result.trades()[0] for r in results[:2]]
    assert all(t.exit_idx < 99 for t in flattened)
    assert len(results[2].result.trades()) == 0
    # Neither strategy would have reached its own limit: the account did.
    assert max(abs(t.pnl) for t in flattened) < 0.01 * CAP


def test_max_positions_counts_the_whole_account():
    data = drifting(0.0002, symbols=SYMBOLS[:3])
    strategies = [Scripted(enter=[5], frac=0.2) for _ in range(3)]
    results, _ = run_strategies(
        [(s, [SYMBOLS[i]], {"account": "p"}) for i, s in enumerate(strategies)],
        data,
        accounts={"p": AccountSpec(initial_capital=CAP, max_positions=2)},
        **KW,
    )
    assert [len(r.result.trades()) for r in results] == [1, 1, 0]
    assert len(refusals(results[2])) == 1


def test_max_gross_exposure_counts_the_whole_account():
    data = drifting(0.0002, symbols=SYMBOLS[:2])
    first, second = Scripted(enter=[5], frac=0.3), Scripted(enter=[5], frac=0.3)
    results, _ = run_strategies(
        [(first, [SYMBOLS[0]], {"account": "e"}), (second, [SYMBOLS[1]], {"account": "e"})],
        data,
        accounts={"e": AccountSpec(initial_capital=CAP, max_gross_exposure=40_000.0)},
        **KW,
    )
    assert len(results[0].result.trades()) == 1 and len(results[1].result.trades()) == 0
    # Each fits under the cap alone.
    capped = isolated(
        Scripted(enter=[5], frac=0.3),
        [SYMBOLS[1]],
        data,
        config=BacktestConfig(initial_capital=CAP, max_gross_exposure=40_000.0),
    )
    assert len(capped.result.trades()) == 1


def test_a_strategy_that_halts_itself_halts_only_itself():
    class Quitter(Scripted):
        def on_bar(self, ctx):
            if ctx.idx == 10:
                ctx.halt()
            super().on_bar(ctx)

    data = drifting(0.0005, symbols=["AAA", "BBB"])
    quitter, other = Quitter(enter=[20]), Scripted(enter=[20], frac=0.5)
    (rq, ro), accounts = run_strategies(
        [(quitter, ["AAA"], {"account": "h"}), (other, ["BBB"], {"account": "h"})],
        data,
        accounts={"h": AccountSpec(initial_capital=CAP)},
        **KW,
    )
    assert rq.halted and not ro.halted and not accounts["h"].halted
    assert len(ro.result.trades()) == 1


# -- results ------------------------------------------------------------------


def test_the_account_result_and_the_attributed_curves():
    data = drifting(0.001, symbols=["AAA", "BBB"])
    strategies = [Scripted(enter=[5], exit=[60], frac=0.4), Scripted(enter=[10], exit=[70], frac=0.4)]
    results, accounts = run_strategies(
        [(s, [SYMBOLS[i]], {"account": "r"}) for i, s in enumerate(strategies)],
        data,
        accounts={"r": AccountSpec(initial_capital=CAP)},
        **KW,
    )
    assert list(accounts) == ["r"]
    account = accounts["r"]
    assert isinstance(account, AccountResult)
    assert account.name == "r" and account.strategies == [0, 1]
    assert account.initial_capital == CAP and not account.halted and account.halted_at is None
    assert account.balance == pytest.approx(CAP + sum(pnl(r) for r in results))
    assert account.equity_curve()[-1] == account.balance
    n = len(account.equity_curve())
    assert len(account.timestamps()) == len(account.drawdown_curve()) == len(account.locked_margin()) == n
    assert (account.locked_margin() == 0).all()  # cash account
    # A strategy's curve starts from its share of the account and moves only
    # by its own trades, so the shares and the P&L add up to the balance.
    finals = [r.result.equity_curve()[-1] for r in results]
    assert results[0].result.equity_curve()[0] == pytest.approx(CAP / 2, rel=0.01)
    assert sum(finals) == pytest.approx(account.balance)
    assert finals[0] == pytest.approx(CAP / 2 + pnl(results[0]))


def test_a_margin_account_reports_locked_margin():
    data = drifting(0.001, symbols=["AAA"])
    _, accounts = run_strategies(
        [(Scripted(enter=[5], exit=[60], frac=0.5), ["AAA"], {"account": "m"})],
        data,
        accounts={"m": AccountSpec(initial_capital=CAP, account_type="margin", leverage=4.0)},
        **KW,
    )
    assert accounts["m"].locked_margin().max() > 0


def test_the_return_shape_depends_only_on_accounts_and_the_memory_report():
    data = drifting(0.001, symbols=["AAA"])
    plain = run_strategies([Scripted()], data, **KW)
    assert isinstance(plain, list)
    spec = {"x": AccountSpec(initial_capital=CAP)}
    results, accounts = run_strategies([Scripted()], data, accounts=spec, **KW)
    assert isinstance(results, list) and accounts == {}
    results, report, accounts = run_strategies(
        [(Scripted(), None, {"account": "x"})], data, accounts=spec, memory_report=True, **KW
    )
    assert "total" in report and list(accounts) == ["x"]


def test_entry_forms():
    data = drifting(0.001, symbols=["AAA", "BBB"])
    spec = {"x": AccountSpec(initial_capital=CAP)}
    s1, s2, s3 = Scripted(), Scripted(), Scripted()
    _, accounts = run_strategies(
        [
            (s1, ["AAA"], {"account": "x"}),
            {"strategy": s2, "symbols": ["AAA", "BBB"], "account": "x"},
            {"strategy": s3},
        ],
        data,
        accounts=spec,
        **KW,
    )
    assert accounts["x"].strategies == [0, 1]
    assert s2.cash and s3.cash  # all ran


# -- batching -----------------------------------------------------------------


def test_the_planner_keeps_an_account_in_one_batch():
    names = ["a", None, "b", "a", None, "b", "a", None]
    for size in (1, 2, 3, 4, 100):
        batches = plan_batches(names, size)
        assert sorted(i for b in batches for i in b) == list(range(len(names)))
        for name in ("a", "b"):
            homes = {k for k, b in enumerate(batches) if any(names[i] == name for i in b)}
            assert len(homes) == 1
        for b in batches:
            members = [i for i in b if names[i] == "a"]
            assert members == sorted(members)  # input order within an account
    assert plan_batches(names, 100) == [[0, 3, 6, 1, 2, 5, 4, 7]]
    assert plan_batches([None] * 7, 3) == [[0, 1, 2], [3, 4, 5], [6]]
    # A batch grows to hold an account larger than the batch size.
    assert max(len(b) for b in plan_batches(names, 2)) == 3


def test_batch_size_changes_no_result_with_accounts():
    data = make_data(4, n=160)
    spec = {
        "x": AccountSpec(initial_capital=CAP, max_positions=2),
        "y": AccountSpec(initial_capital=80_000.0, account_type="margin", leverage=2.0),
    }
    names = ["x", None, "y", "x", None, "y", "x", "y"]

    def run(size):
        entries = [
            (Recorder(fast=3 + i), SYMBOLS[i % 3 :], {"account": n}) for i, n in enumerate(names)
        ]
        results, accounts = run_strategies(entries, data, accounts=spec, batch_size=size, **KW)
        return [fingerprint(r) for r in results], accounts

    base, base_accounts = run(None)
    for size in (1, 2, 3):
        got, accounts = run(size)
        for a, b in zip(got, base):
            same(a, b)
        for name in spec:
            assert accounts[name].equity_curve().tobytes() == base_accounts[name].equity_curve().tobytes()
            assert accounts[name].strategies == base_accounts[name].strategies


def test_each_batch_runs_whole_accounts(monkeypatch):
    from raptorbt.strategy import multi_runner

    seen = []
    real = multi_runner._run_group

    def spy(entries, *a, names=None, **kw):
        seen.append(list(names))
        return real(entries, *a, names=names, **kw)

    monkeypatch.setattr(multi_runner, "_run_group", spy)
    data = drifting(0.001, symbols=["AAA"])
    names = ["x", None, "x", None, "x"]
    run_strategies(
        [(Scripted(), ["AAA"], {"account": n}) for n in names],
        data,
        accounts={"x": AccountSpec(initial_capital=CAP)},
        batch_size=2,
        **KW,
    )
    assert sum(b.count("x") for b in seen) == 3
    assert [b.count("x") for b in seen if "x" in b] == [3]


def test_an_account_over_the_cap_is_refused(monkeypatch):
    monkeypatch.setattr(accounts_module, "MAX_ACCOUNT_STRATEGIES", 2)
    data = drifting(0.001, symbols=["AAA"])
    with pytest.raises(InvalidParameter) as info:
        run_strategies(
            [(Scripted(), ["AAA"], {"account": "x"}) for _ in range(3)],
            data,
            accounts={"x": AccountSpec(initial_capital=CAP)},
            **KW,
        )
    assert info.value.code == "account_too_large"


# -- errors -------------------------------------------------------------------


def test_unknown_account_names_raise():
    data = drifting(0.001, symbols=["AAA"])
    for accounts in (None, {"y": AccountSpec(initial_capital=CAP)}):
        with pytest.raises(InvalidParameter) as info:
            run_strategies([(Scripted(), ["AAA"], {"account": "x"})], data, accounts=accounts, **KW)
        assert info.value.code == "unknown_account" and info.value.value == "x"


@pytest.mark.parametrize(
    "field,value",
    [
        ("initial_capital", 0),
        ("initial_capital", float("nan")),
        ("initial_capital", True),
        ("account_type", "futures"),
        ("leverage", -1),
        ("max_drawdown_pct", 0),
        ("max_drawdown_pct", 120),
        ("max_daily_loss", -5),
        ("max_daily_loss_pct", 101),
        ("max_gross_exposure", 0),
        ("max_positions", 0),
        ("max_positions", 1.5),
    ],
)
def test_account_spec_validates(field, value):
    with pytest.raises(InvalidParameter) as info:
        AccountSpec(**{"initial_capital": CAP, field: value})
    assert info.value.code == "invalid_account_spec" and info.value.parameter == field


@pytest.mark.parametrize("bad", [[1], {"x": 5}, {1: AccountSpec(initial_capital=CAP)}])
def test_accounts_must_be_a_dict_of_specs(bad):
    with pytest.raises(InvalidParameter) as info:
        run_strategies([Scripted()], drifting(0.0, symbols=["AAA"]), accounts=bad, **KW)
    assert info.value.code == "invalid_accounts"


@pytest.mark.parametrize(
    "entry",
    [
        {"symbols": ["AAA"]},
        {"strategy": Scripted(), "colour": 1},
        (Scripted(), ["AAA"], {"acount": "x"}),
        (Scripted(), ["AAA"], "x"),
        (Scripted(), ["AAA"], {"account": 5}),
    ],
)
def test_bad_entries_raise(entry):
    with pytest.raises(InvalidParameter) as info:
        run_strategies([entry], drifting(0.0, symbols=["AAA"]), **KW)
    assert info.value.code == "invalid_strategy_entry"


# -- a property ---------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_balance_is_capital_plus_the_members_pnl_and_cash_stays_non_negative(seed):
    rng = np.random.default_rng(seed)
    data = make_data(seed, n=150)
    cash_mode = bool(rng.integers(2))
    n_groups = int(rng.integers(1, 4))
    specs, entries, members = {}, [], {}
    for g in range(n_groups):
        name = f"g{g}"
        specs[name] = AccountSpec(
            initial_capital=float(rng.integers(20, 200)) * 1_000,
            account_type="cash" if cash_mode else "margin",
            leverage=float(rng.integers(1, 6)),
            max_positions=int(rng.integers(1, 4)) if rng.random() < 0.4 else None,
            max_drawdown_pct=float(rng.integers(5, 40)) if rng.random() < 0.3 else None,
        )
        for _ in range(int(rng.integers(1, 5))):
            enter = rng.choice(150, size=int(rng.integers(1, 6)), replace=False)
            exit_ = rng.choice(150, size=int(rng.integers(1, 6)), replace=False)
            s = Scripted(enter=enter.tolist(), exit=exit_.tolist(), frac=float(rng.uniform(0.1, 1.0)))
            syms = [SYMBOLS[int(i)] for i in sorted(rng.choice(5, size=int(rng.integers(1, 4)), replace=False))]
            entries.append((s, syms, {"account": name}))
            members.setdefault(name, []).append(len(entries) - 1)
    # An isolated strategy or two alongside.
    for _ in range(int(rng.integers(0, 3))):
        entries.append((Scripted(enter=[int(rng.integers(150))]), SYMBOLS[:2], {"account": None}))
    cfg = BacktestConfig(initial_capital=CAP, fees=0.001)
    results, accounts = run_strategies(entries, data, cfg, accounts=specs, **KW)
    for name, idx in members.items():
        account = accounts[name]
        assert account.strategies == idx
        total = sum(pnl(results[i]) for i in idx)
        assert account.balance == pytest.approx(specs[name].initial_capital + total, abs=1e-6 * CAP)
        if cash_mode:
            assert min(min(entries[i][0].cash) for i in idx if entries[i][0].cash) >= -1e-6


# -- LiveNode -----------------------------------------------------------------


def feed(node, data, bars):
    for i in range(bars):
        for s, series in data.items():
            node.push_bar(
                s,
                int(series["timestamps"][i]),
                *(float(series[k][i]) for k in ("open", "high", "low", "close", "volume")),
            )


def test_live_node_groups_match_the_batch_runtime():
    data = drifting(0.001, n=80, symbols=["AAA", "BBB"])
    spec = {"x": AccountSpec(initial_capital=CAP, max_positions=1)}

    def strategies():
        return [
            Scripted(enter=[5], exit=[50], frac=0.5),
            Scripted(enter=[10], exit=[60], frac=0.5),
            Scripted(enter=[10], exit=[60], frac=0.5),
        ]

    def entries(ss):
        return [
            (ss[0], ["AAA"], {"account": "x"}),
            (ss[1], ["BBB"], {"account": "x"}),
            (ss[2], ["AAA"], {"account": None}),
        ]

    want, want_accounts = run_strategies(entries(strategies()), data, accounts=spec, **KW)
    node = LiveNode(entries(strategies()), ["AAA", "BBB"], accounts=spec, **KW)
    feed(node, data, 80)
    got, got_accounts = node.finish()
    for a, b in zip(got, want):
        same(fingerprint(a), fingerprint(b))
    assert got_accounts["x"].balance == want_accounts["x"].balance
    assert got_accounts["x"].strategies == [0, 1]
    assert got_accounts["x"].equity_curve().tobytes() == want_accounts["x"].equity_curve().tobytes()
    assert len(got[1].result.trades()) == 0  # the account is full: one position at a time


def test_live_node_without_accounts_returns_a_list():
    node = LiveNode([Scripted()], ["AAA"], **KW)
    feed(node, drifting(0.0, n=5, symbols=["AAA"]), 5)
    assert isinstance(node.finish(), list)


def test_halt_account_halts_every_strategy_on_it_and_no_other():
    data = drifting(0.0005, n=60, symbols=["AAA", "BBB"])
    ss = [Scripted(enter=[3], frac=0.3), Scripted(enter=[3], frac=0.3), Scripted(enter=[40], frac=0.3)]
    node = LiveNode(
        [
            (ss[0], ["AAA"], {"account": "x"}),
            (ss[1], ["BBB"], {"account": "x"}),
            (ss[2], ["AAA"], {"account": None}),
        ],
        ["AAA", "BBB"],
        accounts={"x": AccountSpec(initial_capital=CAP)},
        **KW,
    )
    feed(node, {s: {k: v[:20] for k, v in d.items()} for s, d in data.items()}, 20)
    node.halt_account("x", flatten=True)
    assert node.is_halted(0) and node.is_halted(1) and not node.is_halted(2)
    feed(node, {s: {k: v[20:] for k, v in d.items()} for s, d in data.items()}, 40)
    results, accounts = node.finish()
    assert results[0].halted and results[1].halted and not results[2].halted
    assert results[0].halted_at == results[1].halted_at == accounts["x"].halted_at
    assert accounts["x"].halted
    # Flattened: the positions opened at 3 were closed soon after the halt.
    assert all(t.exit_idx <= 21 for r in results[:2] for t in r.result.trades())
    assert len(results[2].result.trades()) == 1


def test_halt_account_names_must_exist():
    node = LiveNode([Scripted()], ["AAA"], **KW)
    with pytest.raises(InvalidParameter) as info:
        node.halt_account("nope")
    assert info.value.code == "unknown_account"
