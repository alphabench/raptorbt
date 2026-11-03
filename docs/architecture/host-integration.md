# Embedding raptorbt: what a host controls

A platform that runs raptorbt inside its own system needs four things that are
the same in a backtest and live. This page is their contract.

## One order check (`order_check=`)

```python
def check(order: OrderProposal, ctx) -> None | str | OrderRefusal: ...

run_strategies(strategies, data, order_check=check)
LiveNode(strategies, symbols, order_check=check, execution=venue)
```

* The check sees every typed order a strategy submits (`submit_order`, and what is
  built on it: brackets, structures, exits), once, on the event that produced it,
  before the engine's own checks and before any venue.
* `None` accepts. A code (`str` or `OrderRefusal(code)`) refuses: the order is recorded
  as rejected with that code, never works, never reaches the venue. The strategy hears
  it in `on_order_rejected` / `on_order_event`, `event.reject_reason == code`. It counts
  as a rejection in the run's risk counters.
* The check is pure: it must not submit or cancel orders, and an exception it raises
  stops the run (never read as acceptance). A code is 1-64 bytes; at most 1024 distinct
  codes exist per process, so codes are a stable set, not ids.
* The `enter()` / `close_position()` intents carry no order. With a check installed they
  raise `order_check_intent`; use `submit_order`.
* Time-sliced (`Twap`) schedules are not shown to the check; their parent order is.
* Parity is tested: the same strategy and check refuse the same client ids through
  `run_strategies`, a `LiveNode` in-process and a `LiveNode` on `PaperExecution`
  (`tests/python/test_live_node_host.py`).

## Order tags

`orders.Market(..., tags={"basket": "b-1"})`. Up to 16 string pairs, keys 1-64 bytes,
values up to 256, `client_id` is reserved (the engine writes the strategy's own order id
there). On a node with a venue they arrive in `OrderRequest.tags`; in-process they are
validated and ignored. The check sees them too.

## Warm-up

`LiveNode(warmup=...)`:

| value | what warm-up bars do |
|---|---|
| `"muted"` | hooks run, indicators / aggregators / strategy state update; queued orders and intents are discarded; no position is held (seeds from `initial_positions` stay); nothing reaches a venue |
| `"live"` | bars replay as ordinary bars: orders match, equity is marked (a real `ExecutionAdapter` refuses this) |
| `None` | `"muted"` with your own `ExecutionAdapter`, else `"live"` |

`ctx.is_warmup` is true while a warm-up bar is handled. `node.warm_up({symbol: bars})`
replays history pushed after construction, always muted. Warm-up bars are bars for the
node's clock: they cannot be older than what a symbol has already seen.

## Adopted positions

`initial_positions={strategy: {symbol: seed}}`, `seed` =
`{"quantity", "avg_price", "timestamp_ns"?, "margin_minor"?, "cost_minor"?}`.

* `quantity` is signed; negative is a short and needs a margin account.
* Margin account: `margin_minor` (int, minor units of `money_decimals`) is the margin the
  account holds for the position. Locked as stated; the balance is not debited. Without
  it only fully funded accounts are adopted (the notional locks); anything else is
  refused, because the margin a broker holds is not derivable from quantity and price.
  Option legs margined by the built-in SPAN-style deposit are re-priced with their group
  after the first applied event, like any leg.
* Cash account: `cost_minor` is the cash the position cost (default `avg_price *
  quantity`, rounded to the money scale). Longs only.
* No order, fill, fee, trade record or entry event. The entry price is `avg_price`: a
  short gains as the price falls from it. Adoption happens before any event is applied.

## Instruments, chains and roles from the context

`run_strategies(..., instruments={symbol: InstrumentSpec}, bindings={role: symbol})` and
`LiveNode(..., instruments=..., bindings=...)` build one view of the run's instruments that
every strategy reads through its context:

* `ctx.instrument(key=None)`: the `InstrumentSpec` of a symbol, of a role bound to a symbol,
  or of the symbol being handled. A symbol with no spec raises `no_instrument_spec`.
* `ctx.option_chain(underlying=None)`: an `OptionChain` of the option specs that are listed
  (`activation_ns <= now`) and unexpired at `ctx.event_ns`, so a strategy cannot pick a
  contract the market had not listed. Before the first event there is no time and it holds
  every option spec. Pass `ctx.event_ns` as `now_ns` to its queries.
* `ctx.role(name)`: resolves the strategy's `Requirements` role at the context's time with
  `requirements.resolve`, from the run's specs and the last price of each symbol the strategy
  sees: a symbol for `Underlying`, `Instrument` and `Future`, a tuple for `Chain`. An
  unresolvable role raises `unresolved_role` and says why.

A `bindings` key no strategy declares raises `unknown_role_binding`; a bound symbol outside
the run raises `unknown_symbol`.

## Percent exits

See the module docs of `raptorbt.strategy.exits`. `ExitRule` has `stop_loss_pct` /
`take_profit_pct` (percent of the covered positions' entry value), `stop_loss` / `take_profit`
(currency P&L before costs) and `premium_at_or_above` / `premium_at_or_below`. `set_exit`
covers one symbol (`symbol=`), a named group of symbols (`symbols=`, `name=`) or a structure
group (`group=`). A rule fires once and sends reduce-only market orders with ids `{rule}:x{n}`
and tags `{"exit_rule", "close"}`; they pass `order_check`. A venue adapter that defines
`place_exit_rule(request)` is offered the rule first, only for positions that exist (a rule set
before its entry filled has no legs and is watched by the engine); `Unsupported` / `OrderRefused`
leaves it with the engine. The orders a held rule will fire (`request.exit_client_ids`) are
registered at the node, so their fills are booked, and a reconciliation does not report them
missing while the rule waits. The venue holds the legs it was given: when the position's size
changes the engine calls `replace_exit_rule(old, new)` (or `cancel_exit_rule(old)` and watches the
rule itself), when the position is gone it calls `cancel_exit_rule`, and `cancel_exit(handle)`
calls it too; all three are `Unsupported` in `BrokerAdapter` unless a mapping overrides them. A
rule the venue cannot take back is never reported cancelled (`exit_rule_not_cancellable`). An
answer that is neither yes nor no (`exit_rule_unconfirmed`) registers the exit orders anyway,
stops the engine watching the rule too (no double close), and shows in the next reconciliation as
an `exit_rule_unconfirmed` discrepancy. A refused close is retried with a backoff (1 s, doubling,
at most 30 s) up to `MAX_CLOSE_ATTEMPTS`, then `on_exit_failed(ctx, handle)` is called (the book
raises `exit_close_failed` if the strategy has none).

## Continuous futures

`Future(of="index", contract="continuous", roll_days=3)` in the strategy's requirements: `ctx.role(name)`
is the front contract of the underlying, and becomes the next one three days before the front's
expiry (`roll_days=0` rolls at expiry; the time is read from the contracts' `expiration_ns`).
`class S(RollEvents, Strategy)` adds `on_roll(ctx, RollEvent(role, from_symbol, to_symbol, ts_ns))`,
called after the strategy's own `on_bar` / `on_trade` of the first event at or after the rule's
time (a role's first event only records its contract). With `auto_roll = True` the node moves the
strategy's positions in the old contract with a reduce-only market close and a market open in the
new one on the same side, sized to keep the **notional exposure** (`roll_sizing = "notional"`, the
default: `floor(N_old * L_old * P_old / (L_new * P_new))` lots; `"lots"` keeps the lot count), with the
residual reported on `on_roll_sized` and in the open's tags (ids `{role}:roll{n}:c{k}` / `:o{k}`, tags `roll`, `from`, `to`),
all through `order_check`. `continuous_series(contracts, specs, roll_days=, adjust=)` builds one
history for indicators: `"raw"` keeps traded prices, `"back"` shifts earlier bars by each roll's
gap, `"ratio"` scales them; there is no default.

In a batch run both the old and the new contract must be among the run's symbols. A node adds the next
contract itself when the strategy's `auto_roll` rolls to one the node does not trade but holds the spec of
(next section), or a host adds it ahead of the roll with `plan_instruments(node, roll_lead_days=...)`.

## Instruments that change on a running node

A chain window follows the money and a future rolls: the instrument set of a node is not fixed at construction.

```python
node.add_instrument("IDX-A-110-c", spec, strategies=[0], history=64)   # between pushes
node.retire_instrument("IDX-A-90-c")                                   # refused while in use
plan = plan_instruments(node, margin_strikes=1, roll_lead_days=1)      # pure: reads the node
node.apply_plan(plan)                                                  # retire, then add
```

* **Ids.** The new instrument takes the next id and becomes the last local instrument of each chosen
  strategy (default: every strategy that was given no symbols of its own). Each strategy's instruments stay
  in ascending id order, so no existing local index, order id, position or event ordinal moves.
* **What follows it.** The composite subscriptions and the symbol-less indicators a strategy declared in
  `on_start`; a copy of the clock schedule set there; a `PaperExecution` shadow book; the order engine and
  the account (groups included); `ctx.symbols`, `ctx.instrument`, `ctx.option_chain`, `ctx.role` and the
  `Requirements` resolution. An indicator registered with `symbol=` belongs to that symbol only.
* **Memory.** The rings of an added instrument hold `history` rows (at least what the node's indicators
  read) and are allocated on its first push. Retiring frees them, with the aggregators. It is refused
  (`instrument_in_use`) while a strategy holds a position, a working order, a protective level the engine
  watches or intents held for the next step. The id stays: a retired symbol's slot (a few hundred bytes of
  accounting plus the books' kernel, which keeps its trades and orders for the results) lasts the life of the
  node, but the per-event loops of the books skip retired kernels, so an event costs what it cost before.
* **Reviving.** `add_instrument` of a retired symbol brings it back with the same id: fresh rings and
  aggregators, its kernels flat as they were retired, for the strategies that held it. The record says
  `revive`. A price that wanders back over a window edge reuses the id.
* **Venue.** An execution adapter that has `add_instrument(symbol, spec)` / `retire_instrument(symbol)`
  is told before any order for the symbol can reach it, and again when a restored node replays its
  journal, so a fresh adapter learns the contracts. These are local calls that send nothing to the venue
  (the journaled, fenced, replay-muted path is for requests); a fenced node refuses the add before it
  reaches the adapter. It may refuse an add (`Unsupported`): the node stays as it was. `BrokerAdapter` learns the
  contract from `instruments` or its `resolve_instrument(symbol, spec)` hook.
* **Hooks.** `Strategy.on_instrument_added(ctx, symbol)` and `on_instrument_retired(ctx, symbol)`.
* **Journal.** `instrument_add` (symbol, spec as JSON, strategies, history, bar offset) and
  `instrument_retire`, written before the engine acts, at the node's clock; `restore()` replays them from the
  journal, so a node killed after an add recovers with the instrument. A request a strategy makes through
  `RollEvents` is added at the node's next input and journaled the same way.
* **Cost.** A node that never adds an instrument runs the same loop as before. The per-event work of the
  books and the aggregator index grows with the instruments ever added, retired ones included, so a host
  that cycles through thousands of contracts in one session starts a new node (journal `seq_base`) at its
  session boundary.
* **Not carried:** corporate actions for an added instrument (declare them for every symbol at construction),
  ATR-based stops on it (refused), and a `symbol=`-less indicator shared across strategies by the engine's
  bank: a node runs those privately so they can follow new instruments.
