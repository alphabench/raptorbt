"""``skills.for_task``: which skills a task needs, checked on tasks an agent would actually ask.

The golden tables pin the first skill of each task (or, where two skills are honest answers, that
the right one is in the first two). Beyond them: the skills a match depends on follow it,
transitively and once; rare words outweigh common ones; function and module names find their
skill (``garch`` -> statistics, ``doji`` -> expressions, ``sizing`` -> risk); the stemmer joins
``analysis`` with ``analyses``.
"""

import pytest

import raptorbt.skills as skills

#: task -> the skill that must come first
FIRST = {
    "explore a series and chain results before any strategy exists": "raptorbt-analysis",
    "analyse the returns of a price series": "raptorbt-analysis",
    "export the skills for another agent": "raptorbt-analysis",
    "set a stop-loss and target": "raptorbt-orders",
    "compute ATR and RSI": "raptorbt-indicators",
    "my call raised an error": "raptorbt-troubleshooting",
    "combine 5m and 15m bars": "raptorbt-timeframes",
    "find doji candles": "raptorbt-expressions",
    "garch volatility forecast": "raptorbt-statistics",
    "check if two stocks are cointegrated": "raptorbt-statistics",
    "estimate the hurst exponent": "raptorbt-statistics",
    "detect volatility regimes with a markov model": "raptorbt-statistics",
    "data quality audit of bars before research": "raptorbt-statistics",
    "size a position from risk and stop distance": "raptorbt-risk",
    "run a backtest with fees and slippage": "raptorbt-backtesting",
    "sweep parameters with walk-forward": "raptorbt-optimization",
    "monte carlo bootstrap robustness": "raptorbt-optimization",
    "deploy a strategy to paper trading": "raptorbt-live-trading",
    "recover a live node after a crash": "raptorbt-live-trading",
    "connect kite broker adapter": "raptorbt-live-trading",
    "trade nifty options iron condor": "raptorbt-instruments",
    "option greeks and implied volatility": "raptorbt-instruments",
    "price a call option with black scholes": "raptorbt-instruments",
    "market hours holidays trading session": "raptorbt-instruments",
    "build bars from ticks": "raptorbt-market-data",
    "tick level quotes and order book": "raptorbt-market-data",
    "corporate actions splits dividends in a backtest": "raptorbt-backtesting",
    "shared account with drawdown limits": "raptorbt-portfolio",
    "score a momentum factor with information coefficient": "raptorbt-analysis",
    "read sharpe ratio and drawdown from results": "raptorbt-results",
    "write a class based strategy with on_bar": "raptorbt-strategy",
    "compute bollinger bands and macd": "raptorbt-indicators",
    "write a signal as a formula close above sma 200": "raptorbt-expressions",
    "multi timeframe daily trend filter on minute bars": "raptorbt-timeframes",
    "margin model for sold options": "raptorbt-risk",
    "register an indicator in on_start": "raptorbt-indicators",
    "why does the order get rejected": "raptorbt-orders",
    "what does the handle_expired error mean": "raptorbt-troubleshooting",
    "subscribe to hourly bars": "raptorbt-subscriptions",
    "hedge a position with a trailing stop": "raptorbt-orders",
    "which function computes the average true range": "raptorbt-expressions",
    "look up the doji candlestick pattern function": "raptorbt-expressions",
}

#: task -> a skill that must be among the first two (two skills are honest answers)
TOP_TWO = {
    "rank stocks by cross sectional momentum": "raptorbt-portfolio",
    "what does this indicator look like on a flat name": "raptorbt-analysis",
    "test whether a price series is stationary": "raptorbt-statistics",
    "probability of backtest overfitting": "raptorbt-statistics",
    "evaluate an expression over a csv of bars from the shell": "raptorbt-analysis",
    "explore an indicator before building a strategy": "raptorbt-analysis",
}


def test_the_golden_table_has_about_forty_tasks():
    assert len(FIRST) + len(TOP_TWO) >= 40


@pytest.mark.parametrize("task", sorted(FIRST))
def test_the_first_skill_of_a_task(task):
    assert skills.for_task(task)[0] == FIRST[task]


@pytest.mark.parametrize("task", sorted(TOP_TWO))
def test_the_right_skill_is_among_the_first_two(task):
    assert TOP_TWO[task] in skills.for_task(task)[:2]


def test_no_match_is_an_empty_list():
    assert skills.for_task("xyzzy plugh") == []
    assert skills.for_task("") == []
    assert skills.for_task("the of and") == []


# -------------------------------------------------------------------------- dependencies


def deps(name):
    return {s.name: s for s in skills.index()}[name].depends_on


def test_dependencies_follow_the_matches_transitively_and_once():
    out = skills.for_task("find doji candles")
    assert out[0] == "raptorbt-expressions"
    assert len(out) == len(set(out))
    for name in out:
        for dep in deps(name):
            assert dep in out, (name, dep)
    # indicators is only a dependency of expressions here, and it depends on strategy, which
    # depends on core: all are listed, each dependency before what needs it
    assert {"raptorbt-core", "raptorbt-indicators", "raptorbt-strategy"} <= set(out)
    order = {n: i for i, n in enumerate(out)}
    assert order["raptorbt-core"] < order["raptorbt-strategy"] < order["raptorbt-indicators"]


def test_matches_come_before_dependencies_in_score_order():
    out = skills.for_task("set a stop-loss and target")
    assert out[0] == "raptorbt-orders" and {"raptorbt-core", "raptorbt-instruments"} <= set(out)
    matched = [n for n in out if n in skills.for_task("set a stop-loss and target")]
    assert matched == out


def test_a_skill_whose_dependencies_are_all_listed_adds_nothing():
    out = skills.for_task("my call raised an error")
    assert out == ["raptorbt-troubleshooting", "raptorbt-core"]


# ------------------------------------------------------------- stemming and rare words


def test_the_stemmer_joins_word_forms():
    stem = skills._stem
    for a, b in (
        ("analysis", "analyses"), ("strategy", "strategies"), ("order", "orders"),
        ("cointegration", "cointegrated"), ("compute", "computing"), ("explore", "exploring"),
        ("size", "sizing"), ("estimate", "estimation"), ("bar", "bars"), ("class", "classes"),
    ):
        assert stem(a) == stem(b), (a, b, stem(a), stem(b))
    assert stem("analysis") != stem("analytic")
    assert skills.for_task("analyses")[0] == "raptorbt-analysis" == skills.for_task("analysis")[0]
    assert stem("is") == "is" and stem("bus") == "bus"


def test_a_word_only_one_skill_uses_outweighs_one_every_skill_uses():
    idf = skills._index()[2]
    assert idf[skills._stem("condor")] > 2 * idf[skills._stem("strategy")]
    assert skills.for_task("condor")[0] == "raptorbt-instruments"
    assert skills.for_task("strategy")[0] == "raptorbt-strategy"


def test_function_and_module_names_find_their_skill():
    assert skills.for_task("garch")[0] == "raptorbt-statistics"
    assert skills.for_task("doji")[0] == "raptorbt-expressions"
    assert skills.for_task("cdlhammer")[0] == "raptorbt-expressions"
    assert skills.for_task("stochrsi")[0] == "raptorbt-expressions"
    assert skills.for_task("pbo_cscv")[0] == "raptorbt-statistics"
    assert skills.for_task("sizing")[0] == "raptorbt-risk"
    assert skills.for_task("pricing")[0] == "raptorbt-instruments"


def test_every_registry_function_name_finds_a_skill():
    from raptorbt import expressions

    missing = [f["name"] for f in expressions.functions() if not skills.for_task(f["name"].replace("_", " "))]
    assert missing == []


def test_the_task_is_not_changed_by_case_or_punctuation():
    a = skills.for_task("Set a STOP-LOSS, and target!")
    assert a == skills.for_task("set a stop loss and target") and a[0] == "raptorbt-orders"
