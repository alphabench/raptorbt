# Strategy requirements (the manifest)

A deployment system has to plan data fetching, warm-up, capital and placement before it runs
a strategy. `Strategy.requirements` is the strategy's declaration of what it needs: pure data
on the class, readable once the class is imported, with no instance, no I/O and no run.

```python
from dataclasses import dataclass
from raptorbt.strategy import (
    Bars, Chain, Future, Holding, Legs, Requirements, Strategy, StrategyConfig,
    StrikeWindow, Underlying, param,
)
from raptorbt.structures import Expiry

@dataclass(frozen=True)
class Config(StrategyConfig):
    width: int = param(2, min=1, max=10, description="strikes between the short legs")

class Condor(Strategy):
    requirements = Requirements(
        roles={
            "index": Underlying(asset="index"),
            "hedge": Future(of="index", contract="front"),
            "options": Chain(of="index", expiry=Expiry.weekly(), strikes=StrikeWindow.around(6)),
        },
        timeframes=["1m", "5m"],
        holding=Holding.intraday(flatten_before="15:15"),
        data=["bar"],
        warmup=Bars(50),
        max_structure=Legs(4, lots=2),
        params=Config,
    )
```

## Roles

A role is a name for an instrument the strategy does not know in advance. A host binds every
`Underlying` role to a symbol; the others are derived or fixed.

| Kind | Meaning |
|---|---|
| `Underlying(asset=)` | Bound by the deployer. `asset` is equity, etf, index, currency or commodity. |
| `Instrument(symbol, asset=)` | A fixed symbol. |
| `Future(of, contract=)` | A futures contract of the `of` Underlying: `front`, `next`, or `continuous` (front, rolled to the next at expiry). |
| `Chain(of, expiry, strikes, rights=)` | Options on the `of` Underlying or Future. `expiry` is one `raptorbt.structures.Expiry` (weekly, monthly, nth, nearest, date, exact) or a tuple. `strikes` is `StrikeWindow(low, high)` (listed-strike offsets from the money strike; `StrikeWindow.around(n)`) or `ByDelta(low, high)` (absolute delta band). `rights` defaults to both. |

## Other fields

- `timeframes`: every bar series the strategy reads (`TIMEFRAMES` strings or `Timeframe`).
  `Requirements.feed_interval` is the finest time-based one, which a host must supply.
- `holding`: `Holding.intraday(flatten_before="HH:MM")` (exchange clock) or `Holding.overnight()`.
- `data`: any of `bar`, `trade`, `quote`, `book` (the `data_type` of `Strategy.subscribe`).
- `warmup`: `Bars(n, timeframe=None)` or `Duration.minutes/hours/days(n)`.
- `max_structure`: `Legs(n, lots=1)`, the largest set of legs held at once.
- `params`: a `StrategyConfig` dataclass, described by `params_schema`.

## Parameters

`params_schema(ConfigClass)` reads the dataclass fields. Types are `int`, `float`, `str`,
`bool`, `Literal[...]`, each optionally `| None`. `param(default, min=, max=, choices=,
description=)` attaches bounds. `ParamSchema.validate(values)` returns `Issue`s (`unknown_param`,
`missing_param`, `invalid_param`, `out_of_bounds`, `not_a_choice`). A field it cannot describe
raises `RequirementsError` (`unsupported_type`, `unsupported_default`, `invalid_bounds`,
`invalid_default`).

## Checks

`check_requirements(StrategyClass, dry_run=True)` returns a list of `Issue(code, path, message)`.

- Static: `no_requirements`, `no_roles`, `missing_holding`, `no_timeframes`, `unknown_role`,
  `invalid_base`, `undeclared_timeframe` (warm-up), `timeframes_without_bars`.
- Dry run: the strategy runs on a short synthetic tape built from the declaration (each
  `Underlying` role is bound to its upper-cased role name; futures and a synthetic option
  universe are listed; default config). A subscription outside the declaration is
  `undeclared_instrument`, `undeclared_timeframe` or `undeclared_data`; an exception is
  `dry_run_failed`; an unresolvable declaration is `dry_run_unresolved`.

The declaration is read from the imported class, not from source: importing a strategy module
runs its top level. A host that must not import untrusted code reads the JSON a trusted build
step exported (`to_dict`) instead.

## Hash

`requirements_hash(StrategyClass | Requirements)` is the SHA-256 hex of the canonical JSON
(sorted keys, `,` and `:` separators) of `{"raptorbt": <version>, "requirements": <to_dict()>}`.
Roles, timeframes and data are normalised, so writing them in another order does not change it.
Store it with a saved strategy; a different engine version or declaration gives a different hash.

## Resolution

`resolve(requirements, *, now_ns, bindings, specs, prices=None, vols=None, chain=None)` returns
a `SymbolPlan`: for each role the concrete symbols at `now_ns`, the expiries and the money
strike, and `symbols` (all of them, deduplicated). It never raises for data it lacks; each
gap is an `Issue` (`unbound_role`, `no_contract`, `no_options`, `no_expiry`, `needs_price`,
`needs_vol`) and the role resolves to nothing.

- Futures: the unexpired contracts of the base symbol ordered by expiry; `front` is the
  first, `next` the second, `continuous` the first with the second in `detail["roll_to"]`.
- Chains: expiries by `Expiry.resolve`; the money strike is the listed strike nearest
  `prices[underlying]` (the lower on a tie); a window indexes the strikes listed for that
  expiry and clips at the ends; a delta band takes the strikes between the contracts nearest
  the band's edges at `vols[underlying]` (Black-Scholes, calls positive, puts negative).
  A chain on a Future is on that future's resolved symbol.

## JSON

`Requirements.to_dict()` gives schema 1:

```json
{"schema": 1,
 "roles": {"<name>": {"kind": "underlying", "asset": "index"},
           "<name>": {"kind": "instrument", "symbol": "S", "asset": "equity"},
           "<name>": {"kind": "future", "of": "<role>", "contract": "front"},
           "<name>": {"kind": "chain", "of": "<role>", "expiry": [{"which": "weekly", "n": 0, "min_days": 0,
                      "date_ns": null, "utc_offset_ns": 0, "exact_ns": null}],
                      "strikes": {"by": "window", "low": -6, "high": 6}, "rights": ["call", "put"]}},
 "timeframes": ["1m", "5m"],
 "holding": {"mode": "intraday", "flatten_before": "15:15"},
 "data": ["bar"],
 "warmup": {"bars": 50, "timeframe": null},
 "max_structure": {"legs": 4, "lots": 2},
 "params": [{"name": "width", "type": "int", "default": 2, "required": false, "min": 1, "max": 10,
             "choices": null, "nullable": false, "description": null}]}
```

`warmup` is `{"seconds": n}` for a `Duration`; `strikes` is `{"by": "delta", "low", "high"}` for
`ByDelta`; a renko timeframe is written `"1renko:2.5"`. `Requirements.from_dict` reads it back
(`params` returns as a `ParamSchema`), and `requirements_hash` is equal for the round trip.
