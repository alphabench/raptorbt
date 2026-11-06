"""A statistics workflow that ends in an expression: stationarity, then half-life, then a signal.

Is the price series mean-reverting at all (ADF and KPSS through ``Session.stats``)? How fast does
it revert (the rolling ``half_life`` function, read from the summary of its handle)? Then the
signal window is chosen from that number instead of guessed: ``zscore(close, 2 * half-life) <
-1``. Every step is one call on a handle; the numbers come back as summaries and nothing larger
than a summary is printed. A test over a window describes the window, so the result is a
screen, not a trading rule: the next step is a backtest (see the third example).

Skills: raptorbt-analysis, raptorbt-statistics, raptorbt-expressions.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import _data  # noqa: E402
from raptorbt.analysis import Session  # noqa: E402


def mean_reverting(n: int, seed: int, phi: float = 0.9, level: float = 100.0) -> np.ndarray:
    """An AR(1) around ``level``: each move undoes ``1 - phi`` of the distance to it."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0.0, 1.0)
    return level + x


def main():
    ts = _data.session_closes(days=3)
    session = Session()
    session.load({"timestamps": ts, "close": mean_reverting(len(ts), seed=11)}, bar_stamp="close")

    # 1. stationarity of the close
    close = session.eval("close")["handle"]
    st = session.stats("stationarity", close)
    print(f"stationarity: {st['reading']}  ADF p-value {st['adf']['p_value']:.2g}  n {st['n']}")
    assert st["reading"] == "stationary"

    # 2. how fast it reverts, in bars
    hl = session.eval("half_life(close, 200)")
    half_life = hl["dist"]["p50"]
    print(f"half-life: median {half_life} bars  (p05 {hl['dist']['p05']}, p95 {hl['dist']['p95']})  warm-up {hl['warmup']}")

    # 3. the window comes from the half-life; the signal is one expression
    window = int(round(2 * half_life))
    signal = session.eval(f"zscore(close, {window}) < -1")
    print(f"signal zscore(close, {window}) < -1: true {signal['true']} of {signal['known']} known bars "
          f"({signal['true_rate']:.1%})  handle {signal['handle']}")
    assert signal["kind"] == "bool" and 0 < signal["true"] < signal["known"]


if __name__ == "__main__":
    main()
