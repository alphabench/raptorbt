"""Analysis for an agent that explores before it builds: one call, a small true answer.

A :class:`Session` loads bars once and evaluates expressions over them; each result is kept
under a short handle (``h_`` and ten hex digits) that the next expression reads as a series,
and comes back as a summary of at most 2 KB: its shape, warm-up, NaN count, distribution, last
values, flags, whether it is causal, what its numbers were checked against, and a hash.

>>> import numpy as np
>>> from raptorbt.analysis import Session
>>> s = Session()
>>> t = np.arange(1, 101, dtype=np.int64) * 60 * 10**9
>>> _ = s.load({"timestamps": t, "close": 100 + np.sin(np.arange(100) / 5)}, bar_stamp="close")
>>> a = s.eval("rsi(close, 14)")
>>> b = s.eval(f"zscore({a['handle']}, 20)")
>>> (a["warmup"], b["warmup"], b["causal"])
(14, 33, True)

The same session is the command line: ``python -m raptorbt eval "rsi(close, 14)" --data bars.csv
--bar-stamp close`` (see ``raptorbt-analysis``).

This package reads and writes files (the command line's session cache, ``export``), so it is
not imported by ``import raptorbt`` and is not available inside a sandboxed strategy environment that forbids file access. The pure parts
are ``raptorbt.expressions`` and ``raptorbt.stats``.
"""

from raptorbt.analysis.session import Session

__all__ = ["Session"]
