---
name: raptorbt-deployment
description: >
    What a host must provide and decide to run a validated strategy for real:
    admitting a strategy from its requirements manifest, resolving its roles
    into symbols, building the node with bindings, a venue adapter, a journal
    and an epoch, supervising it, recovering it, and what to verify before real
    money.
version: 1.0
api_version: 2.x
depends_on:
    - raptorbt-core
    - raptorbt-strategy
    - raptorbt-live-trading
    - raptorbt-execution
    - raptorbt-risk
status: stable
---

# RaptorBT Deployment

## 1. When to use this skill

Load it when you move a strategy that passed its backtest toward paper or real
money, when you are the platform that runs strategies other people wrote, or
when you must decide what a worker has to hold, check and log. The mechanics of
the node, journal and recovery are [live-trading](../live-trading/SKILL.md); the
venue contract is [execution](../execution/SKILL.md). This skill is the
division of labour between the engine, your host and the operator, and the
order in which to verify it.

## 2. Domain overview

The engine runs a strategy deterministically on the events it is given. It does
not choose symbols, fetch data, hold credentials, own a calendar, restart a
crashed process, or decide who is allowed to trade. A host does.

**What the host provides.**

| Need | Where it comes from |
| --- | --- |
| The strategy's declaration | `Strategy.requirements` (pure data on the class), or the JSON a trusted build step exported with `Requirements.to_dict()` |
| Concrete symbols | `resolve(requirements, now_ns=, bindings=, specs=, prices=, vols=)`: you bind every `Underlying` role and supply the `InstrumentSpec`s and last prices |
| Instruments and roles at run time | `LiveNode(instruments={symbol: spec}, bindings={role: symbol})`, read by the strategy through `ctx.instrument`, `ctx.option_chain`, `ctx.role` |
| Market data with real arrival times | `push_tick`, `push_bar`, `push_depth` with `arrival_ns=`; the clock your supervisor trusts |
| A venue | an `ExecutionAdapter` (a broker mapping from `raptorbt.adapters`, built with a fresh `namespace` and an authenticated push channel), or `PaperExecution()` |
| Durable memory | a `Journal` with a file or a sink, an `fsync` policy, and a copy off the machine |
| Fencing | an `epoch` number per worker generation, enforced by the venue adapter's `fence(epoch)` |
| History to warm up | bars covering the declaration's `warmup`, pushed as `warmup_bars=` |
| Supervision | something that calls `node.heartbeat()` on a timer and reads `node.health()` |

**What the host decides** (the engine will not): parameter values within the
declared bounds; capital, account type, leverage and the risk limits of
`BacktestConfig`; `order_check` rules; `warmup` mode; `protective` mode; what an
operator must approve after a recovery; who may start, halt or flatten; what
runs at a session's start and end.

**The manifest.** `requirements` names roles (`Underlying`, `Instrument`,
`Universe`, `Future`, `Chain`), timeframes, data kinds (`bar`, `trade`, `quote`,
`book`), `Holding` (intraday with a flatten time, or overnight), `warmup`, the
legs of the structure held at once (`max_structure=Legs(OptionLeg(...), ...)`:
side, right, strike offset or delta with a window, expiry), the largest position
per role (`max_position={role: MaxPosition(lots=, notional=, fraction=)}`) and a
parameter schema. It is what lets a host plan data, warm-up, capital and
placement before it runs anything: a host estimates capital from `max_position`
without asking the deployer, lists the contracts to subscribe with
`resolve_legs(...)` and margins the structure with
`legs.structure(vol=).resolve(...).margin(model, prices)`. A `Universe` is a
basket the deployer fills (`min` to `max` symbols of one asset class); bind it
with `bindings={"basket": ["A", "B"]}` to `run_strategies` or `LiveNode`, and
the strategy reads the list with `ctx.role("basket")`. A binding with the wrong
shape is `invalid_binding`; one with too few or too many symbols is
`universe_size`.
`check_requirements(cls)` verifies the class does not subscribe to more than it
declared (it runs the strategy on a synthetic tape with every `Underlying`
and `Universe` role bound, so `ctx.role(...)` works there);
`requirements_hash(cls)` is a SHA-256 of the canonical declaration and
the engine version: store it with a saved strategy, and a different declaration
or a different engine gives a different hash, so a saved strategy is re-admitted
rather than silently run on a new engine. Importing a strategy module runs its
top level; a host that must not import untrusted code reads the exported JSON
and never imports the class.

**Resolution never guesses.** `resolve` returns a `SymbolPlan` (`roles`,
`symbols`, `issues`). Each gap is an `Issue` and the role resolves to nothing:
`unbound_role`, `no_contract`, `no_options`, `no_expiry`, `needs_price`,
`needs_vol`, and for a universe `invalid_binding` and `universe_size`. A plan with
issues is not deployable. Resolve again as time passes
(a continuous future rolls, a weekly expiry passes); a role's symbols are a
function of the time you ask.

**Epochs and workers.** Exactly one worker at a time may send for an account.
Give each generation of a worker an epoch greater than the last (a counter you
store, not a clock). The venue adapter's `fence(epoch)` raises `Fenced` when a
newer epoch owns the account; the older node writes a `fence` record and stops
sending. This protects against a stale process; it cannot recall a request
already in flight and does not stop a process that never asks. Where the venue
cannot enforce an epoch, your orchestrator must guarantee a single writer.

**Versions.** The installed engine version is `importlib.metadata.version("raptorbt")`; a skill's
`api_version` is its major (`2.x`). On `import raptorbt` a daemon thread may ask
pypi.org, at most once a day, whether a newer release exists and log one line.
It never blocks and never changes behaviour; set `RAPTORBT_NO_VERSION_CHECK=1`
to open no socket (CI is skipped automatically). Pin the engine version in the
image: a journal is replayed by the engine, so restore with the engine version
that wrote it (not tested across versions: a replay that makes a different
record raises `replay_diverged`); finish a session before changing the engine.

## 3. API reference

| Name | What it is |
| --- | --- |
| `raptorbt.strategy.Requirements`, `Underlying`, `Instrument`, `Universe`, `Future`, `Chain`, `StrikeWindow`, `ByDelta`, `Holding`, `Bars`, `Duration`, `Legs`, `OptionLeg`, `MaxPosition` | The declaration. `to_dict()` / `from_dict()` (schema 2) round trip, and `requirements_hash` is equal for the round trip. `Legs(4, lots=2)`, the count-only form, is refused (`legs_need_shape`). |
| `raptorbt.strategy.check_requirements(cls, dry_run=True)` | `list[Issue(code, path, message)]`. Static: `no_roles`, `missing_holding`, `no_timeframes`, `unknown_role`, `invalid_base`, `undeclared_timeframe`, `timeframes_without_bars`, `role_data_not_declared`, and for the legs `leg_chain_ambiguous`, `leg_outside_chain`. Dry run: `undeclared_instrument`, `undeclared_timeframe`, `undeclared_data`, `dry_run_failed` (the strategy raised), `dry_run_unresolved`. |
| `raptorbt.strategy.requirements_hash(cls_or_requirements)` | The hash above. |
| `raptorbt.strategy.resolve(requirements, *, now_ns, bindings, specs, prices=None, vols=None, chain=None)` | A `SymbolPlan` of concrete symbols per role at `now_ns`. `bindings` maps an `Underlying` to a symbol and a `Universe` to a list. |
| `raptorbt.strategy.resolve_legs(requirements, *, now_ns, bindings, specs, prices, chain=None)` | A `LegPlan`: for each declared leg (`LegContracts`) the contracts it can mean (one for an `offset` leg, the window of candidates for a `delta` leg), the units and the sign margin models read. `plan.lines(prices)` is the argument of `MarginModel.required_legs`; `issues` lists what could not be resolved (`needs_price`, `no_expiry`, `leg_strike_unlisted`). |
| `raptorbt.strategy.params_schema(ConfigClass)`, `ParamSchema.validate(values)`, `param(default, min=, max=, choices=, description=)` | The parameter schema and its issues: `unknown_param`, `missing_param`, `invalid_param`, `out_of_bounds`, `not_a_choice`. |
| `raptorbt.capabilities()` | A JSON-serialisable dict of what this engine supports (order kinds, time in force, fill timings, hooks, indicators, adapters, error codes): read it to decide what to offer instead of hard-coding. |
| `LiveNode(..., journal=, epoch=, instruments=, bindings=, order_check=, warmup=, protective=)`, `heartbeat()`, `health()`, `checkpoint()`, `LiveNode.restore(...)`, `halt(i, flatten=)`, `halt_account(name, flatten=)` | What you wire ([live-trading](../live-trading/SKILL.md)). |
| `raptorbt.version_check` | The once-a-day release check (`check_for_update`, `is_outdated`, `parse_version`). |
| `raptorbt.crash_report`, `raptorbt.enable_crash_reporting` | Opt-in, off-by-default engine-panic reports, sent to the maintainers' own server (named in the README) unless you name another collector (`RAPTORBT_CRASH_REPORTING=1`, `RAPTORBT_CRASH_ENDPOINT`, `RAPTORBT_CRASH_KEY`); fields listed in the README. |

## 4. Standard workflow

1. **Admit.** Import (or read the JSON), `check_requirements`, validate the
   parameters against `params_schema`, compute `requirements_hash`, store it.
2. **Plan.** Bind every `Underlying` role and every `Universe` role (a list),
   `resolve`, refuse a plan with
   issues, and check that the symbols and timeframes are ones you can feed.
3. **Rehearse** on recorded data: the backtest ([backtesting](../backtesting/SKILL.md);
   `run_strategies(trade_from=)` runs the history that primes indicators without
   trading it: orders decided before it are refused `warmup`), then the same strategy through `PaperExecution()` with a journal and real
   arrival times.
4. **Build the node** with the plan's symbols, `instruments=`, `bindings=`, your
   adapter, `epoch=`, a journal whose header carries the run id, the strategy
   and the requirements hash, `warmup_bars=`, `order_check=`, and risk limits.
5. **Run preflight** (section 9) against the exact configuration.
6. **Paper**, then **small real size** under tight limits, watching
   `health()` and the journal; scale only after a full session including a
   deliberate restart.
7. **Operate**: heartbeat on a timer, journal off the machine, a runbook for
   `recovering`, and a halt path that works while the feed is down.

## 5. Examples

Admit, plan and build. The declaration is checked, its parameters validated and
hashed, the roles resolved to concrete symbols at a moment in time, and the node
is built from the plan with the hash in the journal's header. A plan with a gap
names it:

```python
from dataclasses import dataclass

import raptorbt
from raptorbt import BacktestConfig, LiveNode, Strategy, StrategyConfig, journal as rj
from raptorbt.execution import PaperExecution
from raptorbt.strategy import (Bars, Chain, Future, Holding, Requirements, StrikeWindow, Underlying,
                               check_requirements, param, params_schema, requirements_hash, resolve)
from raptorbt.structures import Expiry

DAY = 86_400 * 10**9
NOW = 1_700_000_000 * 10**9
EXPIRY = NOW + 10 * DAY


def universe():
    specs = {"IDX": raptorbt.InstrumentSpec.equity("IDX"),
             "FUT1": raptorbt.InstrumentSpec.futures_contract("FUT1", EXPIRY, 50.0, underlying="IDX")}
    for strike in (95.0, 100.0, 105.0):
        for right in ("call", "put"):
            symbol = f"IDX-{strike:g}-{right[0]}"
            specs[symbol] = raptorbt.InstrumentSpec.option(symbol, strike, right, EXPIRY, 50.0, underlying="IDX")
    return specs


@dataclass(frozen=True)
class Config(StrategyConfig):
    width: int = param(1, min=1, max=3, description="strikes either side of the money")


class Condor(Strategy):
    requirements = Requirements(
        roles={"index": Underlying(asset="index"),
               "hedge": Future(of="index", contract="front"),
               "options": Chain(of="index", expiry=Expiry.nearest(), strikes=StrikeWindow.around(1))},
        timeframes=["1m"], holding=Holding.overnight(), data=["bar"], warmup=Bars(20), params=Config)


specs = universe()

# 1. admit
assert check_requirements(Condor) == []
issues = params_schema(Config).validate({"width": 9})
assert [i.code for i in issues] == ["out_of_bounds"]                       # refuse before it runs
assert params_schema(Config).validate({"width": 2}) == []
declared = requirements_hash(Condor)
assert requirements_hash(Requirements.from_dict(Condor.requirements.to_dict())) == declared

# 2. plan: a gap is named, never guessed
unbound = resolve(Condor.requirements, now_ns=NOW, bindings={}, specs=list(specs.values()), prices={})
assert unbound.symbols == () and [i.code for i in unbound.issues] == ["unbound_role"]
unpriced = resolve(Condor.requirements, now_ns=NOW, bindings={"index": "IDX"}, specs=list(specs.values()))
assert [i.code for i in unpriced.issues] == ["needs_price"]                # no price, no money strike
plan = resolve(Condor.requirements, now_ns=NOW, bindings={"index": "IDX"}, specs=list(specs.values()),
               prices={"IDX": 100.0})
assert plan.issues == () and plan.roles["hedge"].symbols == ("FUT1",)
assert plan.symbols == ("IDX", "FUT1", "IDX-95-c", "IDX-95-p", "IDX-100-c", "IDX-100-p", "IDX-105-c", "IDX-105-p")

# 3. build, with the hash in the journal header
journal = rj.Journal({"run": "r-17", "strategy": "condor", "requirements_hash": declared}, memory=True)
node = LiveNode([Condor()], list(plan.symbols), BacktestConfig(fees=0.0, initial_capital=1_000_000.0),
                bar_stamp="close", execution=PaperExecution(), instruments=specs, bindings={"index": "IDX"},
                journal=journal, epoch=1)
health = node.heartbeat()
assert health["state"] == "live" and health["epoch"] == 1 and not health["fenced"] and health["unconfirmed"] == []
assert journal.head()[2]["requirements_hash"] == declared
```

A dry run that finds the strategy broken before it is deployed:

```python

from raptorbt import Strategy
from raptorbt.strategy import Holding, Requirements, Underlying, check_requirements


class Broken(Strategy):
    requirements = Requirements(roles={"asset": Underlying(asset="equity")}, timeframes=["1m"],
                                holding=Holding.overnight())

    def on_init(self, ctx):
        raise RuntimeError("boom")


class Undeclared(Strategy):
    pass


(issue,) = check_requirements(Broken)
assert issue.code == "dry_run_failed" and "boom" in issue.message
assert [i.code for i in check_requirements(Undeclared)] == ["no_requirements"]
```

The recovery runbook as code. The worker died while the venue filled part of an
order. The restored node does not guess: it names every difference and stays in
`recovering` until the facts the host has are pushed and the rest is resolved or
accepted by a person. The operator's decision is written into the journal:

```python
import os
import tempfile

from raptorbt import BacktestConfig, LiveNode, Strategy, journal as rj
from raptorbt.execution import ExecutionReport
from raptorbt.strategy import orders

MIN = 60_000_000_000
CONFIG = BacktestConfig(initial_capital=1_000_000.0, fees=0.0)


class Exchange:
    namespace = "N1"
    keeps_own_records = True  # remembers the node's orders itself, so a restored node need not teach it
    def __init__(self):
        self.orders, self.held, self.cash = {}, {}, 1_000_000.0

    def submit(self, request):
        self.orders.setdefault(request.client_order_id, {"client_order_id": request.client_order_id, "open": True,
                                                         "symbol": request.symbol, "quantity": request.quantity,
                                                         "filled_qty": 0.0})

    def cancel(self, client_order_id):
        pass

    def modify(self, client_order_id, *, quantity=None, limit_price=None, trigger_price=None):
        pass

    def open_orders(self):
        return [dict(o) for o in self.orders.values() if o["open"]]

    def positions(self):
        return dict(self.held)

    def balances(self):
        return {"cash": self.cash}


class Trader(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.submit_order(orders.Limit(side="buy", price=90.0, quantity=5))


path = os.path.join(tempfile.mkdtemp(), "run.journal")
venue = Exchange()
node = LiveNode([Trader()], ["AAA"], CONFIG, bar_stamp="close", execution=venue,
                journal=rj.Journal({"run": "r-17"}, file=path))
for i in range(3):
    node.push_bar("AAA", (i + 1) * MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)
del node                                                      # the worker dies

venue.orders["s0-0"]["filled_qty"] = 3.0                      # meanwhile the venue filled 3 at 90
venue.held, venue.cash = {"AAA": 3.0}, 1_000_000.0 - 3 * 90.0

journal, records = rj.open_for_recovery(path, memory=True)
found = []
node = LiveNode.restore(None, [Trader()], ["AAA"], CONFIG, journal=journal, records=records, execution=venue,
                        bar_stamp="close", on_discrepancy=found.append)
assert node.state == "recovering"
assert sorted(d.kind for d in found) == ["cash_mismatch", "order_mismatch", "position_mismatch"]

# The host pushes what the venue reported; each fact it can prove shrinks the difference.
node.push_report("s0-0", ExecutionReport.accepted("V1", 2 * MIN), arrival_ns=10**15)
node.push_report("s0-0", ExecutionReport.fill("X1", 3.0, 90.0, 3 * MIN), arrival_ns=10**15)
assert node.reconcile() == [] and node.state == "recovering"      # nothing differs; resume is still explicit
node.annotate(3 * MIN, "operator", {"decision": "venue fill of 3 applied from the broker's trade book"})
node.resume()
assert node.state == "live"
assert [r.kind for r in node.journal.since(0)][-1] == "resume"
```

A preflight a host runs against the exact configuration before it deploys: the
same strategy on a paper node and in-process must trade alike, and a node
rebuilt from its journal at awkward cut points must end in the same account
state. It returns the failures instead of asserting, so a pipeline can print
them all:

```python
import numpy as np
from raptorbt import BacktestConfig, LiveNode, Strategy, journal as rj
from raptorbt.execution import PaperExecution
from raptorbt.strategy import orders

MIN = 60_000_000_000
close = 100.0 + np.cumsum(np.random.default_rng(7).normal(0.0, 0.4, 90))
BARS = [((i + 1) * MIN, close[i] - 0.1, close[i] + 0.4, close[i] - 0.4, close[i], 1000.0) for i in range(90)]
CONFIG = BacktestConfig(initial_capital=200_000.0, fees=0.0005, slippage=0.0003)


class Rhythm(Strategy):
    def on_bar(self, ctx):
        if ctx.idx % 15 == 3:
            self.submit_order(orders.Market(side="buy", quantity=20))
        if ctx.idx % 15 == 10:
            self.submit_order(orders.Market(side="sell", quantity=20, reduce_only=True))


class Quiet:                                        # a venue that takes requests and answers nothing
    namespace = "N1"
    keeps_own_records = True  # remembers the node's orders itself, so a restored node need not teach it
    def submit(self, request):
        pass

    def cancel(self, client_order_id):
        pass

    def modify(self, client_order_id, *, quantity=None, limit_price=None, trigger_price=None):
        pass

    def open_orders(self):
        return []


def trades(execution):
    node = LiveNode([Rhythm()], ["AAA"], CONFIG, bar_stamp="close", execution=execution)
    for bar in BARS:
        node.push_bar("AAA", *bar)
    return [(t.entry_idx, t.entry_price, t.exit_idx, t.exit_price, t.size, t.fees)
            for t in node.finish()[0].result.trades()]


def preflight():
    failures = []
    plain, paper = trades(None), trades(PaperExecution())
    if not plain or plain != paper:
        failures.append("paper and in-process trades differ")
    for cut in (7, 31, 62):
        journal = rj.Journal({"run": "preflight"}, memory=True)
        node = LiveNode([Rhythm()], ["AAA"], CONFIG, bar_stamp="close", execution=Quiet(), journal=journal)
        for bar in BARS[:cut]:
            node.push_bar("AAA", *bar)
        rebuilt = LiveNode.restore(None, [Rhythm()], ["AAA"], CONFIG, journal=rj.Journal({"run": "preflight"}, memory=True),
                                   records=journal.since(0), execution=Quiet(), bar_stamp="close", reconcile="skip")
        if rebuilt.state_digest() != node.state_digest():
            failures.append(f"restore at bar {cut} changed the account state")
    return failures


assert preflight() == []
```

## 6. Composition

- Parameters, hooks and the manifest: [strategy](../strategy/SKILL.md).
  Limits that refuse orders and halt trading: [risk](../risk/SKILL.md).
- The node, journal, recovery and broker adapters: [live-trading](../live-trading/SKILL.md);
  the venue contract and `order_check`: [execution](../execution/SKILL.md).
- Calendars and sessions (`SessionSchedule`, `squareoff_time`) and contract
  specs: [instruments](../instruments/SKILL.md). Feeds and timeframes:
  [market-data](../market-data/SKILL.md), [timeframes](../timeframes/SKILL.md).
- What goes wrong and what it is called: [troubleshooting](../troubleshooting/SKILL.md).
- Runnable examples: `examples/live-trading/paper_node.py`.

## 7. Constraints and correctness

- **Event time.** Decisions are a function of the events pushed and their
  arrival times, not of the wall clock. A replay reproduces them only if the
  host recorded the arrival times it pushed with.
- **One writer per account.** Epochs and adapter fencing are a safeguard, not a
  lock. If the venue cannot enforce an epoch, the orchestrator must.
- **Strategies must be deterministic** to be recoverable: no wall clock, random
  source or environment reads in hooks. Seed randomness from the journal header.
- **Secrets.** Tokens and keys live in `Secret`s inside adapters and never in a
  journal record, a request, an event or a log line. Keep them out of the
  journal header too: it is stored in clear.
- **Untrusted authors.** Importing a strategy runs its top level. Admit from
  exported JSON where you cannot import, and run unreviewed strategies only
  where a single misbehaving process is contained (separate process, limits,
  no credentials).
- **Money is fixed-point.** Cash, fees and P&L are booked in minor units at
  `money_decimals` (default 2); reconcile with the broker in the same units.
  A non-finite or out-of-range amount stops the run with `money_overflow`
  instead of wrapping.
- **Limits are engine-side only for what the node sees.** A venue can fill what
  the book would refuse; that arrives as `fill_breach` ([execution](../execution/SKILL.md)).
  Add venue-side limits at the broker as well.
- **Version changes.** Treat an engine upgrade as a new deployment: re-run
  admission (the hash changes), the backtest, the preflight, and do not restore
  a journal across versions.
- **Gaps in the engine today** (so you can plan around them): a push that
  raised because an adapter's `submit` raised an unknown-outcome exception
  leaves a journal that cannot be restored (`venue_unavailable` during replay);
  `node.health()["unconfirmed"]` is empty on a node with no journal; a push to
  an unknown symbol raises a plain `KeyError`; seed errors for
  `initial_positions` on an invalid account raise plain `ValueError`. See
  [troubleshooting](../troubleshooting/SKILL.md).

## 8. Common mistakes

Running a plan that has issues, and a role that nobody bound:

<!-- skill: invalid -->
```python
import raptorbt
from raptorbt.strategy import Holding, Requirements, Underlying, resolve

requirements = Requirements(roles={"asset": Underlying(asset="equity")}, timeframes=["1m"], holding=Holding.overnight())
plan = resolve(requirements, now_ns=1_700_000_000 * 10**9, bindings={}, specs=[raptorbt.InstrumentSpec.equity("AAA")])
assert plan.symbols, "unbound_role: nothing resolved, and the plan says why in plan.issues"
```

- Deploying without ever restoring. A journal you have not replayed is a hope.
  Kill the worker in rehearsal.
- Giving two live workers the same epoch, or a clock-derived one that can go
  backwards. Use a counter you store.
- Reusing a namespace, or building the broker adapter without
  `trusted=True` or a verifier, then passing `authenticated=True` to silence it.
- Warming up with fewer bars than the longest indicator needs, or with bars
  stamped differently from live ones (`bar_stamp`).
- Letting `warmup="live"` replay history into a real venue. The default for
  your own adapter is muted, and a live warm-up that makes an order is refused.
- Resolving the plan once at start and never again: a front contract rolls and
  a weekly expiry passes. Re-resolve on a timer and compare.
- Trusting `requirements_hash` as a signature. It detects a change; it does not
  authenticate an author.
- Auto-accepting every discrepancy after a restart. `resume(accept="all")` is a
  decision a person made, and belongs in the journal.
- Scaling size on the strength of a paper run that never saw a restart, a
  fenced node, a rejected order or a stale feed.

## 9. Validation

What to verify before real money. Each is a test you can run; write down the
result.

1. **Static.** `python -m raptorbt.skills check --strict` is clean for the
   skills you rely on; `check_requirements` returns no issue; the parameters
   validate; the stored `requirements_hash` equals the current one.
2. **Plan.** `resolve` returns no issue at the start of the session and again
   at its end; every symbol is one the feed delivers; the declared `warmup` is
   met.
3. **Parity.** The strategy on `PaperExecution()` and in-process trades alike on
   a recorded day (the preflight above), and every difference is one of those
   listed in [execution](../execution/SKILL.md).
4. **Adapter.** `run_conformance` passes for any adapter you wrote; for a
   shipped mapping, the payload shapes in its docstring are confirmed against a
   paper or very small live account (fields the broker's public pages do not
   state are marked); the namespace is fresh; every push channel is
   authenticated; the static IP or session the broker requires is in place.
5. **Money.** A hand-computed order: quantity, lot size, tick size, fees from
   the schedule you will actually be charged, margin. The first real order is
   the smallest the venue accepts, under a tight `order_check` and
   `max_order_qty` / `max_position_qty` / daily-loss limits.
6. **Recovery.** Kill the worker at several points, including between a
   request and its answer; restore; `state_digest()` equals; no order was sent
   twice; every discrepancy was resolved by a fact or a recorded decision.
7. **Fencing.** Start a second worker at a higher epoch: the first reports
   `fenced` and sends nothing.
8. **Failure.** Feed gap, broker 429/5xx, expired token, rejected order,
   cancel losing a race with a fill: each ends in a state a supervisor can see
   (`health()`, adapter events), none in a silent one.
9. **Halt.** `node.halt(i, flatten=True)` and `halt_account(...)` work with the
   feed down, and you have tried them.
10. **Audit.** The journal is shipped off the machine; `rj.read(path)` verifies
    the chain; an operator can find, for any order, its decision (`command`), its
    request (`adapter_request`) and the venue's answers (`report`).

## 10. Related skills

- [Live trading](../live-trading/SKILL.md)
- [Execution](../execution/SKILL.md)
- [Strategy](../strategy/SKILL.md)
- [Risk](../risk/SKILL.md)
- [Troubleshooting](../troubleshooting/SKILL.md)
