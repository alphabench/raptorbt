# Option pricing, Greeks and implied volatility

`raptorbt.pricing` prices European options and solves for implied volatility.
It is plain Black-Scholes-Merton and Black-76; there is no hidden calibration.

```python
import numpy as np
from raptorbt import pricing as pr

pr.price("call", 24_000.0, 24_100.0, tau=7 / 365, vol=0.14, rate=0.065, dividend_yield=0.012)
g = pr.greeks("put", 24_000.0, 23_800.0, 7 / 365, 0.16)       # Greeks(price, delta, ...)
iv = pr.implied_vol(112.5, "call", 24_000.0, 24_100.0, 7 / 365, rate=0.065)

# Arrays broadcast like NumPy and return arrays; invalid elements are NaN.
strikes = np.arange(23_000.0, 25_001.0, 100.0)
chain = pr.greeks("call", 24_000.0, strikes, 7 / 365, 0.14)   # GreeksArray
```

## Models

| `model=`                  | `underlying` is | forward                         | `dividend_yield` |
|---------------------------|-----------------|---------------------------------|------------------|
| `"black_scholes"` (default) | spot           | `S * exp((r - q) * tau)`        | allowed          |
| `"black76"`               | forward/futures | `underlying` itself             | must be 0        |

Price: with `F` the forward, `df = exp(-r * tau)`, `sd = vol * sqrt(tau)`,
`d1 = (ln(F/K) + sd^2/2) / sd`, `d2 = d1 - sd`:

* call `df * (F * N(d1) - K * N(d2))`, put `df * (K * N(-d2) - F * N(-d1))`.

Both models are the same formula on a different forward, so they agree when
fed the same forward. `N` is `0.5 * erfc(-x / sqrt 2)` using a fixed
rational approximation (error under one ulp), so results are identical on
every platform.

## Units

* `tau` is in years; use `pr.year_fraction(now_ns, expiry_ns)` for a 365-day
  year between nanosecond timestamps.
* `vol`, `rate`, `dividend_yield` are annual decimals, continuously compounded.
* `delta` is per unit of `underlying` (per unit of the forward for Black-76).
* `gamma` is the second derivative in `underlying`.
* `vega` is per 1.00 of volatility; divide by 100 for a volatility point.
* `theta` is per year and negative when the option decays; divide by 365 for
  a calendar day. For Black-76 the forward is held fixed.
* `rho` is per 1.00 of rate; for Black-76 it is `-tau * price` (the rate only
  discounts).

## Edge cases are defined

| input                              | result |
|------------------------------------|--------|
| `tau == 0`                         | price is intrinsic; delta is +-1 in the money, else 0; other Greeks 0 |
| `vol == 0`, `tau > 0`              | price is the discounted intrinsic value on the forward; Greeks are the exact derivatives of that payoff (delta jumps at the forward; gamma and vega are 0) |
| spot/strike <= 0, tau < 0, vol < 0, any NaN or infinity | scalar call: `ValueError`; array call: `NaN` for that element |
| Black-76 with `dividend_yield != 0` | `ValueError` / `NaN` |

`NaN` always means "absent". A missing or invalid element never turns into
zero.

## Implied volatility

`implied_vol` solves price(vol) = target on `[lo, hi]` (default 0.0001 to 5,
that is 0.01% to 500%) with a safeguarded Newton iteration: analytic vega
steps while they stay inside a shrinking bracket and make progress, bisection
otherwise. It stops when the bracket or the step is narrower than `tol`
(default 1e-12). Accepted answers reprice the target to about 1e-12 relative.

When there is no answer it says why instead of returning a number:

| `IvStatus`          | meaning |
|---------------------|---------|
| `OK`                | solved |
| `EXPIRED`           | `tau == 0` |
| `INVALID_INPUT`     | an input is invalid or not finite |
| `BELOW_INTRINSIC`   | the price is negative or below discounted intrinsic value (an arbitrage) |
| `ABOVE_UPPER_BOUND` | the price is at or above the discounted forward (call) or strike (put) |
| `BELOW_BRACKET`     | the price is at or below the price at `lo`: zero time value, or a price of exactly 0 |
| `ABOVE_BRACKET`     | the price needs a volatility at or above `hi` |
| `NO_CONVERGENCE`    | `max_iter` ran out inside the bracket |

`implied_vol(...)` raises `ImpliedVolError` (a `ValueError`, with `.status`) for
a scalar and returns `NaN` for an array element. `implied_vol_status(...)`
returns `(iv, status)` and never raises for a missing answer.

Numerical limits to know: deep out-of-the-money prices carry almost no
information about volatility (vega near zero), so a one-tick price change can
move the implied volatility a lot; a price that has underflowed to zero is
reported as `BELOW_BRACKET`, not as a volatility.

## Performance and threads

Array entry points take C-contiguous or strided `float64` arrays (strided
input is copied once), borrow them without further copies, release the GIL,
and use all cores for 4096 elements or more. Results do not depend on the
thread count: each element is computed independently with no reduction.

## Option chain

`raptorbt.OptionChain` selects contracts from `InstrumentSpec` options (each
needs an `underlying`, `expiration_ns` and a strike). **Every query takes the
event time `now_ns`** and sees only contracts listed then: a contract is
visible from its `activation_ns` (always, if it has none) up to but not
including its `expiration_ns`. An expiry is listed once any of its contracts
is. There is no call that returns the unfiltered chain.

```python
chain = raptorbt.OptionChain(specs)
now = ctx.timestamp                       # the event time, nanoseconds
exp = chain.pick_expiry("NIFTY", now, "monthly", n=0)   # or "nearest", "nth", "weekly", "date"
k = chain.atm_strike("NIFTY", exp, now, spot)           # ties go to the lower strike
call = chain.contract("NIFTY", exp, k, "call", now)
otm = chain.strike_at_offset("NIFTY", exp, now, spot, +2)
wing = chain.by_delta("NIFTY", exp, now, "put", -0.15, spot, vol=0.14, rate=0.065)
cheap = chain.closest_by("NIFTY", exp, now, "call", 5.0, lambda c: quotes.get(c.symbol))
```

* `"nearest"` is the first expiry at or after `now + min_days`; `"nth"` counts
  listed expiries from 0; `"date"` matches the calendar day of `date_ns`.
* `"monthly"` is the last expiry of its calendar month and `"weekly"` any other.
  This is a calendar fact, computed over **every expiry the chain was built
  with**, so it does not change as contracts get listed. Build the chain from
  the full expiry calendar. Months and days follow `utc_offset_ns` (default UTC).
* `by_delta` prices each listed contract of that right at a flat `vol` and
  takes the nearest delta (calls positive, puts negative).
* `closest_by(measure)` takes any number per contract (a quoted premium, a
  spread); contracts for which it returns `None` or `NaN` are skipped.
* Ties on distance always go to the lower strike. Lists are ascending.

## Provenance and differences from the reference

The formulas, conventions and edge-case rules were checked against a reference
implementation on a 4,704-row grid (`tests/python/golden/pricing_reference.json`,
numbers only): prices agree to
1e-16 relative, Greeks too wherever `vol > 0`. Intended differences are listed
in `CHANGELOG.md` and in the header of `tests/python/test_pricing.py`. Not
ported: American-exercise, Bachelier, SABR, SVI, Heston and binomial models.
