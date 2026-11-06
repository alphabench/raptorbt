"""Factor research on a panel: rank, chain, and score against the returns that follow.

Forty names, each with a daily drift that persists. ``rank(roc(close, 20))`` ranks them on every
date and the summary names the instruments at both ends of the last cross-section. Its handle
chains into ``zscore_xs``, and ``Session.factor`` scores that against forward returns: the rank
IC, its t-statistic with the correction for overlapping returns (``t_hac``, beside the naive
``t_naive`` that is too large at long horizons), the hit rate, the gross long-short spread and the
turnover. Nothing is charged for costs (``no_costs``), so the spread is an upper bound. The
summary pins to ``factor_report`` on the same arrays, and the per-date arrays stay behind the
handle.

Skills: raptorbt-analysis, raptorbt-expressions, raptorbt-statistics.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import _data  # noqa: E402
from raptorbt import expressions  # noqa: E402
from raptorbt.analysis import Session  # noqa: E402

NAMES = 40
DAYS = 250


def panel(seed: int = 5) -> dict[str, np.ndarray]:
    """Daily closes: a drift per name that persists, plus noise."""
    rng = np.random.default_rng(seed)
    drift = rng.normal(0.0, 0.001, NAMES)
    close = 100.0 * np.exp(np.cumsum(drift + rng.normal(0.0, 0.01, (DAYS, NAMES)), axis=0))
    ts = _data.ist_ns(_data.FIRST_DAY, 15, 30) + np.arange(DAYS, dtype=np.int64) * _data.DAY_NS
    return {"timestamps": ts, "close": close}


def main():
    data = panel()
    names = [f"STK{j:02d}" for j in range(NAMES)]
    session = Session()
    session.load_panel(data, bar_stamp="close", instruments=names)

    # 1. a cross-sectional expression: the summary names instruments
    rank = session.eval("rank(roc(close, 20))")
    top, bottom = rank["last"]["top"][0], rank["last"]["bottom"][0]
    print(f"rank(roc(close, 20)): warm-up {rank['warmup']}, {rank['coverage']['median']:.0f} names a date; "
          f"strongest {top[0]} ({top[1]}), weakest {bottom[0]} ({bottom[1]})")

    # 2. chain the handle; the repeat is the same handle
    z = session.eval(f"zscore_xs({rank['handle']})")
    assert z["warmup"] == rank["warmup"]
    assert session.eval(f"zscore_xs({rank['handle']})")["handle"] == z["handle"]

    # 3. score it against forward returns
    report = session.factor(z["handle"], horizons=(1, 5, 10), quantiles=5)
    for h, row in report["horizons"].items():
        print(f"horizon {h:>2}: IC {row['ic_mean']:+.3f}  t_hac {row['t_hac']:.2f} (naive {row['t_naive']:.2f})  "
              f"hit {row['hit_rate']:.0%}  long-short {row['long_short_mean']:+.4f}/day (gross)")
    print(f"turnover top {report['turnover']['top']:.2f}  bottom {report['turnover']['bottom']:.2f}  no_costs {report['no_costs']}")

    # 4. the summary is factor_report on the same arrays; the per-date IC is behind the handle
    direct = expressions.factor_report(session.series(z["handle"]), data["close"], horizons=(1, 5, 10), quantiles=5)
    for h in (1, 5, 10):
        assert np.array_equal(session.series(f"{report['handle']}_ic_{h}"), direct["horizons"][h]["ic"], equal_nan=True)
    assert report["no_costs"] is True
    assert report["horizons"]["10"]["ic_mean"] > 0 and report["horizons"]["10"]["t_hac"] < report["horizons"]["10"]["t_naive"]

    # 5. look at a few names' standardised ranks, full precision
    last = session.series(z["handle"], tail=1, instruments=["STK00", "STK01"])[0]
    print("last z-scores:", {n: round(float(v), 4) for n, v in zip(("STK00", "STK01"), last)})


if __name__ == "__main__":
    main()
