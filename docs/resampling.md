# Resampling

Ticks into bars, and bars into coarser bars, on a grid you declare. There is one time-window
bar builder in the engine (the one behind `BarAggregator` and `aggregate_bars`); resampling is
that builder with the choices a feed that runs for days needs. There is no second aggregator.

```python
import raptorbt as r

# whole arrays (the GIL is released)
bars = r.resample_trades(ts, price, qty, 5 * 60 * 10**9, closed="right")
bars = r.resample_bars(ts, o, h, l, c, v, 30 * 60 * 10**9, source_ns=60 * 10**9,
                       anchor_ns=open_ns, tz_offset_ns=r.IST_OFFSET_NS, closed="right")

# one record at a time
s = r.Resampler(60 * 10**9, closed="right", close_on="clock", gaps="carry")
for bar in s.push_trade(ts, price, qty):   # the bars this trade closed
    ...
```

`bars` is a dict of arrays: `timestamp` (bucket ends), `open`, `high`, `low`, `close`,
`volume`, `n_source`, `complete`, `carried`. `Resampler` calls return `ResampledBar` objects
with the same fields plus `closed_by`. `raptorbt.sampling.TimeBars` and `Resample` name the two
common right-closed declarations.

## What you choose

| Option | Meaning |
|---|---|
| `width_ns` | Bucket width. |
| `source_ns` | Width of the bars fed in (0 for trades). The width must be a multiple of it and the anchor must sit on its grid, or the declaration is refused. |
| `anchor_ns`, `tz_offset_ns` | Boundaries at `anchor_ns + k * width_ns` on the local clock `tz_offset_ns` ahead of UTC. A 30-minute grid from a 09:15 open is `anchor_ns=09:15`, `tz_offset_ns=IST_OFFSET_NS`. |
| `session_anchored` | Restart the grid at every `on_session(True, ts)`; records before the first open are not bucketed; `on_session(False, ts)` stamps the session's last bucket at the close. |
| `closed` | Which edge a record exactly on a boundary belongs to. `"left"` is `[start, end)`: the boundary print opens the next bar, and bars are stamped at their open. `"right"` is `(start, end]`: it closes the bar it ends, and bars are stamped at their close. |
| `close_on` | `"next_print"`: a bar closes when a later record arrives. `"clock"`: as soon as `on_clock(ts)` reaches its end. A bucket's last source bar closes it at once either way. |
| `gaps` | `"skip"`, or `"carry"`: a flat zero-volume bar at the last close for every empty interval. |
| `min_source` | Drop a bar built from fewer source records. |

A bar is stamped with its bucket's end. `n_source` is how many records built it (0 for a
carried bar), `complete` is false for a bar of resampled bars that saw fewer source bars than
its width spans, and `closed_by` says what closed it: `clock`, `print`, `carry`, `session` or
`flush`.

## Refusals

Nothing is repaired. A record earlier than the open bucket (or inside one the clock already
closed), a non-finite price, a negative volume, a source bar off the source grid or of another
width, and trades fed to a bar resampler (or the reverse) raise, and the resampler is left as it
was. `resample_*` names the row.

## Agreement with the reference sampler

`tests/python/differential/test_resample_differential.py` replays streams that
a reference sampler was fed (anchors from a session open,
clock and next-print closing, carried gaps, skipped source bars, `min_source`) and requires the
same bars call by call: stamp, prices and volume bit for bit, source count, `complete`,
`closed_by`. The default options are the plain aggregator's: `test_resample.py` checks
`resample_trades` against `aggregate_bars`.

## Intended differences from the reference sampler

- **Every empty interval is carried.** The reference filled only the intervals one clock step passed over,
  so a clock that closed a bar at each boundary left the quiet minutes before the next print as
  holes. Now each empty interval after the last bar, however it was closed, is a carried bar.
- **A late bar is refused.** The reference `Resample` merged a bar from an earlier bucket into the open
  one, putting its prices in the wrong bar. A record from before the open bucket now raises
  (the reference `TimeBars` already did).
- **`push_bar`'s `period` is optional.** The reference read a missing period as 0 and refused it; 0 now
  means "not stated" and a stated one must match `source_ns`.
- **Closing by a source bar is `print`, by a trade is `clock`** under `close_on="clock"`, as a
  trade past the boundary is the clock's signal and a bar is just the next record.
- **Not ported:** the count, volume, turnover, range, Renko, Heikin-Ashi and hybrid samplers
  (`BarAggregator` has count, volume, value, Renko and signed-flow units), `TimeBars` fed bars
  (read as prints at their close), `first_partial`, `state()` / `load()` snapshots, and
  `adjust()` / `restart()` / `drop_ended()` for futures rolls.
