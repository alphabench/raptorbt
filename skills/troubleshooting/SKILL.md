---
name: raptorbt-troubleshooting
description: >
    Structured errors by symptom: how to read a RaptorBT error, the codes a
    run, an order, a live node, a recovery and an adapter raise or report, how
    to read a journal, the mistakes that recur across skills, and the engine
    gaps known today.
version: 1.0
api_version: 1.x
depends_on:
    - raptorbt-core
status: stable
---

# RaptorBT Troubleshooting

## 1. When to use this skill

Load it when a call raised a RaptorBT error, an order was rejected, a run was
refused, results look wrong, a live node stopped or came back `recovering`, or
a broker adapter reported an event you do not recognise. Start from the symptom
(section 3), then follow the `skill` named in the error to the domain that owns
it.

## 2. Domain overview

**Every error the engine raises on purpose is a `RaptorBTError`.** It carries:

| Field | Meaning |
| --- | --- |
| `code` | A stable snake_case identifier. Branch on this, never on the message. |
| `operation` | The call that failed (`run_strategy_backtest`, `LiveNode.restore`). |
| `parameter`, `value` | What was invalid and what it was. |
| `valid` | The accepted alternatives, or `None`. |
| `skill` | The documentation domain that owns the code: load that skill. |
| `to_dict()` | All of the above, JSON-serialisable; `raptorbt.errors.from_dict` rebuilds it. |

The first line of `str(error)` is the short message; the lines after list what
was requested, what is supported and the related skill. Each class also
subclasses the builtin it replaced (`InvalidParameter` and `InvalidData` are
`ValueError`, `StateError` is `RuntimeError`, `UnknownSymbol` is `KeyError`, `InvalidType` is `TypeError`), so
older `except ValueError` code keeps working. `raptorbt.errors.CODES` maps every
code to its class, owning skill and meaning; `raptorbt.errors.markdown()`
renders the catalogue (`docs/errors.md` is generated from it).

**Two kinds of "no".** An *error* stops a call before it changes anything: the
state is as it was. A *rejection* is an outcome: an order the book or the venue
would not take is recorded as `order_rejected` with a reason, the run goes on,
and the strategy hears it in `on_order_rejected`. Do not catch an error to
continue as if nothing happened, and do not treat a rejection as an exception.

**An exception from a venue is not a refusal.** An adapter that raises
`OrderRefused` says the order will never trade. Any other exception means *it may
or may not have arrived*: it leaves the order `Submitted`, propagates out of the
push that caused it, and is settled by the venue's report or a reconciliation.

## 3. API reference

Where to look, by what you saw.

**A run refused before it started (data and configuration).**

| Code | Class | Cause and fix |
| --- | --- | --- |
| `non_finite_price` | `InvalidData` | A bar's open, high, low or close is NaN or infinite; the message names the column and first index. Clean the data; a pushed tick (`ltp`, `bid` or `ask`, the first in that order named) or bar with a NaN or infinite price is refused the same way, before it is journaled, and changes nothing. |
| `decreasing_timestamps` | `InvalidData` | A batch tape goes backwards. A pushed event older than its symbol's latest raises `invalid_parameter` ("timestamps must not decrease"). A tick older only than another symbol's latest, or than the last `advance_time`, is not refused: it is taken at its own time. |
| `length_mismatch`, `missing_field`, `empty_data`, `invalid_array`, `invalid_input_type`, `no_symbols`, `duplicate_symbols` | `InvalidData` / `InvalidParameter` | The arrays are not the same length, a field is absent, there is nothing to run on, or the argument is not a dict of numeric arrays. See [market-data](../market-data/SKILL.md). |
| `missing_bar_stamp`, `invalid_bar_stamp`, `missing_bar_ns` | `InvalidParameter` | `bar_stamp` is required and is `"open"` or `"close"`; `"open"` needs `bar_ns`. Any entry that takes bars refuses a missing `bar_stamp`; pass it. |
| `invalid_fill_timing`, `invalid_oms_type`, `invalid_account_type`, `invalid_leverage`, `invalid_money_decimals`, `invalid_squareoff_time`, `invalid_config`, `invalid_parameter` | `InvalidParameter` | A `BacktestConfig` or run argument is out of range. `valid` lists the alternatives. See [core](../core/SKILL.md). |
| `fee_schedule_gap`, `invalid_fee_schedule` | `InvalidParameter` | A fee schedule has no row for an instrument's venue and segment, or the instrument has no `venue`, or a fill falls before the schedule's first row (the error names the earliest date; presets start 2023-04-01 for NSE cash and NFO, 2024-10-01 elsewhere) (the run is refused rather than charged nothing; `valid` lists the venues the schedule prices), or a card is malformed. Give the `InstrumentSpec` a `venue`, add the row, or set `fee_segment`. See [backtesting](../backtesting/SKILL.md). |
| `squareoff_needs_timezone` | `InvalidParameter` | `squareoff_time` is a local time of day and the instrument has no `SessionSchedule` and the config no `session_tz_offset_ns`. Name the zone (`IST_OFFSET_NS`, or `0` for UTC) or attach a schedule. |
| `mixed_holding` | `InvalidParameter` | Strategies in one run declare different `requirements.holding` (intraday with different flatten times, or intraday beside overnight or undeclared), so there is no single square-off time. Set `squareoff_time`, or run them apart. |
| `order_validity_unsupported` | `InvalidOrder` | A `gtc` or `gtd` order on an instrument whose exchange segment holds only day orders (NSE equity derivatives, currency, BSE, MCX), or a `gtd` dated past the contract's maturity on NSE commodity. `valid` lists `day`, `ioc`, `fok`; keeping an order alive across days is the host's resubmission, not the exchange's. |
| `session_not_covered` | `InvalidData` | A bar falls on a date outside those the instrument's session schedule covers (the `mcx()` preset covers 2025-11-03 to 2026-11-01; a calendar document covers its `valid_from`..`valid_to`), so whether the market traded is unknown. The message names the date and the covered ranges; load a calendar document for the period. |
| `invalid_calendar` | `InvalidParameter` | A document given to `SessionSchedule.from_calendars` is not valid JSON, has no `utc_offset`, names no entry for the venue, has a bad time or date, or uses another zone than the first document. The message names the document. |
| `money_overflow` | `MoneyOverflow` (an `ArithmeticError`) | A booked amount is not finite or is past the integer range of minor units at `money_decimals`; the run stops instead of wrapping. Lower the amount, or lower `money_decimals`. |
| `unknown_symbol`, `no_instrument_spec`, `unknown_account` | `UnknownSymbol` / `InvalidParameter` | A name that is not in this run. `valid` lists the names. |

**A call was refused before it ran, by what it was given.** Every row is a
`RaptorBTError` whose `valid` lists the alternatives or states the rule; the
state is as it was before the call.

*Strategies, accounts and runs.*

| Code | Class | Cause and fix |
| --- | --- | --- |
| `invalid_strategy`, `duplicate_strategy`, `empty_strategies`, `invalid_strategy_entry` | `InvalidParameter` | An entry is not a `Strategy` instance, subclass, `(strategy, symbols)` pair or entry dict; the same instance is listed twice (pass the class, or a new instance each); or the list is empty. See [strategy](../strategy/SKILL.md). |
| `invalid_accounts`, `invalid_account_spec`, `account_too_large` | `InvalidParameter` | `accounts` is not `dict[str, AccountSpec]`, a spec field is out of range (capital, leverage and limits are positive; percentages at most 100), or more than 500 strategies share one account. See [portfolio](../portfolio/SKILL.md). |
| `invalid_batch_size`, `invalid_curve_samples`, `invalid_history` | `InvalidParameter` | A count that must be a positive integer (or `None`). |
| `invalid_bar_ns`, `invalid_bar_spec`, `invalid_bar_step`, `invalid_primary_bars`, `primary_bars_without_ticks` | `InvalidParameter` / `InvalidSubscription` | A bar width is not a positive whole number of nanoseconds, a `(step, unit)` pair is wrong (step at least 1; unit one of the bar units in `valid`), or `primary_bars` was given with no `ticks` to aggregate. See [timeframes](../timeframes/SKILL.md). |
| `invalid_timeframe`, `invalid_data_type`, `not_subscribed_timeframe`, `not_subscribed_data`, `invalid_indicator_input`, `declaration_outside_run` | `InvalidSubscription` / `InvalidParameter` / `StateError` | `subscribe()` or a `ctx.data` view named a timeframe or data type that is not listed in `valid`, asked for what the strategy never subscribed to, or was called after `on_start`. Subscribe in `on_init` / `on_start`. Indicators read `close` only. See [subscriptions](../subscriptions/SKILL.md). |
| `portfolio_total_unavailable` | `UnsupportedOperation` | A `ctx.portfolio` total needs the shared runtime: use `run_strategies`, not `run_strategy_backtest`. |
| `invalid_input` | `InvalidParameter` | An array or number was refused by the computation (an indicator period, a covariance shape); the message names it. |
| `invalid_parameter`, `invalid_config` | `InvalidParameter` | A value is out of range in the engine's own check; the message names it (a stop or target percent is a fraction in (0, 1], so 5% is `0.05`). |
| `invalid_argument_type` | `InvalidType` (also a `TypeError`) | A non-typed order reached `submit_order` (build it with `orders.Market`, `orders.Limit`, ...), a custom indicator lacks `update(bar)`, `ready` or `value`, or a strategy defines a hook under both its name and its deprecated alias (keep the new name). |
| `unknown_skill` | `UnknownSymbol` (also a `KeyError`) | `raptorbt.skills.load` was given a name that is not a skill; `valid` lists the names. |
| `expr_syntax`, `expr_unknown_function`, `expr_type`, `expr_unbound_parameter`, `expr_unsupported_here`, `sweep_too_large` | `ExpressionError` (also a `ValueError`), `InvalidParameter` for `sweep_too_large` | An expression was refused. `position` is the character offset and `snippet` the source line with a caret; `valid` lists the closest names for a misspelt function or column. Python `and`, `or`, `not` and chained comparisons on an `Expr` raise `expr_type`: use `&`, `|`, `~`. `expr_unbound_parameter`: a `$name` has no value; the text is accepted (it can be swept) and every other use refuses. `sweep_too_large`: a sweep has more combinations or would need more memory than `max_combinations` / `max_bytes`; the message states the size. See [expressions](../expressions/SKILL.md). |
| `insufficient_observations`, `degenerate_series`, `did_not_converge` | `InvalidData` (also a `ValueError`) | A `raptorbt.stats` function refused: too few observations (the message gives the count it got and the count it needs), data that cannot support the statistic (a constant series, series that are exact combinations of one another), or a fit that did not settle (GARCH, regimes). No number is returned. Other refusals of these functions are `invalid_input`. See [statistics](../statistics/SKILL.md). |
| `unknown_handle`, `handle_expired`, `session_memory_exceeded` | `InvalidParameter`, `StateError`, `InvalidParameter` | An analysis session refused a handle or a result. `unknown_handle`: the name is not a result of this session (`valid` lists the nearest). `handle_expired`: the session dropped or evicted it to stay inside `max_bytes` or `max_handles`; `valid` and the message hold the `Session.eval(...)` call that recomputes it. `session_memory_exceeded`: one result (with the loaded bars) is larger than `max_bytes`; use fewer bars or a larger budget. From the command line the same fields print as JSON and the exit status is 2. See [analysis](../analysis/SKILL.md). |
| `invalid_direction`, `invalid_combine_mode`, `invalid_allocation`, `invalid_option_type`, `invalid_strike_selection`, `invalid_size_type` | `InvalidParameter` | An enumerated argument is not one of the values in `valid`. See [backtesting](../backtesting/SKILL.md). |
| `unsupported_short_ticks` | `UnsupportedOperation` | `run_tick_backtest` is long-only: `quantity` is zero or more. Use `run_strategies` with ticks for shorts. |
| `unsupported_corporate_actions`, `invalid_corporate_action`, `invalid_adjustment_policy` | `UnsupportedOperation` / `InvalidParameter` | Corporate actions went to a runner that cannot apply them (the runners that can are in `valid`), an action has an unknown kind or a bad ratio, amount or pay date, or actions came without `adjustment_policy` (`raw` or `adjusted`). See [execution](../execution/SKILL.md). |
| `unsupported_on_bars` | `UnsupportedOperation` | `order_latency_ns` and `partial_fills` are tick-path settings; on bars they would change nothing. Run on ticks or leave them off. |
| `invalid_strategies`, `invalid_instrument_spec`, `instrument_in_use` | `InvalidParameter` / `StateError` | `add_instrument` named strategies the node does not have, or a spec for another symbol; `retire_instrument` met an open position or working order (close and cancel first). |
| `invalid_exit_rule` | `InvalidParameter` | An `ExitRule` has no condition, a non-positive threshold, or `stop_loss_pct` above 100. |
| `invalid_expiry_pick`, `invalid_option_right` | `InvalidParameter` | `OptionChain.pick_expiry` or an option's right is not one of the values in `valid`. |

*Market data.*

| Code | Class | Cause and fix |
| --- | --- | --- |
| `misaligned_legs` | `InvalidData` | The legs of a basket or pair do not share their timestamps: trade only on bars every leg has. |
| `depth_without_ticks` | `InvalidData` | `depth` names a symbol with no `ticks`. |
| `invalid_data` | `InvalidData` | `adjust_history` or a resample was given a field it cannot use; the message and `parameter` name it. |
| `invalid_trade_from` | `InvalidParameter` | `trade_from` must be an event time in integer nanoseconds since the epoch (or `None`). |
| `window_state` | `StateError` | A windowed runtime was pushed to outside `run_strategies_windowed`, so its next window cannot continue from its history, or `ctx.keep_history` after the first window asked for rows already released (declare it in `on_start`). |
| `invalid_window`, `window_overlap`, `window_schema` | `InvalidParameter` / `InvalidData` | `run_strategies_windowed` takes `Window` objects in ascending time. A window must start after the last event of the one before it (it may not cut a timestamp or overlap), end at or after its own rows, and carry bars, ticks and depth, for symbols of `symbols=` only, as the first window declared them. |
| `invalid_adjust`, `no_continuous_bars` | `InvalidParameter` / `InvalidData` | `continuous_series` needs `adjust` of `raw`, `back` or `ratio`, and bars for each contract inside its own window. |

*Orders.*

| Code | Class | Cause and fix |
| --- | --- | --- |
| `invalid_order_side`, `invalid_time_in_force`, `invalid_order_kind`, `invalid_offset_kind` | `InvalidOrder` | An order term is not one of the values in `valid`. See [orders](../orders/SKILL.md). |
| `invalid_order_value`, `missing_order_field`, `incompatible_order_options` | `InvalidOrder` | A price, size, offset, protective level (`stop_price`, `target_price`) or count is out of range, NaN or infinite; a field the kind needs is missing (`limit_price`, `trigger_price`, `offset`, `expire_ns` for `gtd`); or two options do not go together (`at_open` with a limit, `post_only` with a market). |
| `invalid_order_link`, `invalid_order_intent`, `invalid_order_tags` | `InvalidOrder` | An OCO or parent link names too few or unknown orders; a bar queued two `enter()` or two `close_position()`; or `tags` is not at most 16 `str` pairs (no `client_id` key). |
| `invalid_order_reference`, `unknown_order` | `InvalidOrder` | A position id or client id the run does not hold. A submitted order that has finished is a no-op, not this error. |
| `off_lot_quantity`, `off_tick_price` | `InvalidOrder` | A quantity is not whole lots, or a price not whole ticks; `valid` lists the nearest. Round before submitting. |
| `bracket_under_hedging` | `InvalidOrder` | A reduce-only child order cannot close a position under `hedging`: attach `stop_price` and `target_price` to the entry, or use `netting`. |

*Instruments, sessions, margin and sizing.*

| Code | Class | Cause and fix |
| --- | --- | --- |
| `untradable_instrument` | `UnsupportedOperation` | An index was given to a single-instrument session. In `run_strategies` and `LiveNode` an index is accepted as data and an order on it is refused `not_tradable`. |
| `invalid_session_schedule`, `invalid_session_date`, `invalid_instrument_metadata`, `mcx_session_unknown`, `invalid_settlement` | `InvalidParameter` | A schedule window, weekday or date is wrong; a `venue` or `currency` is empty; an MCX contract of no known class has no `session=`; a settlement style or price is not in `valid`. See [instruments](../instruments/SKILL.md). |
| `invalid_roll_sizing`, `roll_price_missing` | `InvalidParameter` / `StateError` | `roll_sizing` is `notional` or `lots`; a notional roll needs the new contract to have a bar before the roll (feed one, or size by lots). |
| `invalid_margin_model` | `InvalidParameter` | A margin model name, rule or leg does not fit, or a `margin_model` was given to a cash account. See [risk](../risk/SKILL.md). |
| `invalid_structure`, `structure_contract_missing`, `structure_missing_mark` | `InvalidParameter` / `StateError` | A leg, selector or submission argument is invalid; the chain lists no contract the leg selects at the event time; or a price or mark a structure needs was not supplied (`prices` or `marks`). |
| `structure_limit_breached`, `structure_margin_exceeded` | `InvalidParameter` | The legs' prices make the net premium worse than `net_limit`; the margin model needs more than `available`. Nothing was queued. |
| `below_one_lot`, `invalid_sizing_input` | `InvalidParameter` | A sizing rule yields under one lot (nothing is rounded up), or an input is not a positive finite number. |
| `unknown_role_binding`, `unknown_role`, `no_requirements`, `unresolved_role`, `role_before_event`, `no_instrument_spec` | `InvalidParameter` / `UnknownSymbol` | A `bindings` key or `ctx.role` name no strategy declares, a strategy with no `requirements`, a role the data cannot resolve at this time, a `Future` role read before the first event (read it in `on_bar`, not `on_start`), or a symbol with no `InstrumentSpec`. |
| `invalid_binding`, `universe_size`, `universe_instrument` | `InvalidParameter` | A role bound to the wrong shape (a list for an `Underlying`; for a `Universe` anything but a list of distinct symbols); a `Universe` bound to fewer than its `min` or more than its `max` symbols; `ctx.instrument(role)` of a `Universe`, which names many symbols (`ctx.role(role)` lists them). |

*Live node, journal and venue, not listed above.*

| Code | Class | What happened and what to do |
| --- | --- | --- |
| `adoption_refused`, `invalid_initial_positions` | `InvalidParameter` | A seeded position does not fit the account (a short or a posted margin on a cash account, a cost basis on a margin one) or names a strategy or symbol the node lacks. |
| `history_exhausted` | `StateError` | The history ring is full of rows a strategy asked to keep, or a depth row was overwritten. Raise `history`, or keep fewer rows. |
| `group_open`, `group_timestamp_mismatch`, `duplicate_bar`, `conflicting_bar` | `StateError` | A timestamp group is open (`flush()` it first), a bar was pushed into a group at another time, or a bar was pushed again (identical: a retry, refused; different: a delivered bar is not revised). |
| `stale_epoch`, `epoch_required`, `fence_unsupported` | `StateError` / `InvalidParameter` | `restore()` needs an `epoch` newer than the journal's; an `epoch` was given with an adapter whose `can_fence` is not true. |
| `adapter_cannot_adopt`, `adapter_namespace_missing`, `adapter_namespace_mismatch`, `hedging_unsupported_by_venue` | `StateError` / `InvalidParameter` | The adapter cannot account for the orders the node holds (`restore_orders` or `keeps_own_records`), has no `namespace`, is not the namespace the journal was written for, or nets positions under a hedging node. |
| `exit_rule_unconfirmed`, `exit_rule_not_cancellable`, `exit_close_failed` | `StateError` | The venue did not say what it holds for an exit rule (ask it: reconcile), cannot take a held rule back, or refused the closes of a rule the allowed number of times. |
| `roll_close_failed`, `roll_open_failed` | `StateError` | An auto-roll's close or open ended without filling: the position is still in the old contract (close failed) or in neither (open failed). Override `on_roll_failed` to handle it. A close that ends `canceled` or `rejected` for `expired` at the old contract's expiry is not a failure: the contract settled the position and the open follows (`on_roll_settled`). |

**An order was rejected** (`order_rejected`, `event.reject_reason`).

| Reason | Meaning |
| --- | --- |
| `invalid_qty`, `zero_size` | No size on an opening order, or a capital fraction that rounds to nothing at the lot step. |
| `insufficient_capital`, `insufficient_margin`, `margin_unpriced` | Not enough cash or margin, or the margin model has no price for the instrument. |
| `reduce_only`, `post_only` | The order would not reduce, or would have crossed the book. |
| `not_tradable` | The order (or an `enter()`) was for an index, a data-only reference: trade the instrument it tracks. |
| `warmup` | The order was decided before the run's `trade_from`: the bars before it only warm the strategy up. Return early while `ctx.is_warmup` is true. |
| `expired`, `inactive`, `market_closed`, `squareoff` | Time in force, an inactive instrument, outside the session, or after the cutoff. |
| `order_size`, `max_position`, `max_exposure`, `max_positions`, `daily_loss`, `killed`, `drawdown_halt`, `margin_call` | A risk limit ([risk](../risk/SKILL.md)). |
| `no_mark` | A market order bound for a real venue under `max_order_notional` or `max_gross_exposure` had no price to be checked at: its instrument has had no print. Send it after the instrument's first bar or tick, or give the order a limit price. |
| your own code | `order_check` refused it. |
| the venue's text | `OrderRefused("...")` from the adapter, or `unsupported: ...` from `Unsupported`; a broker mapping gives the broker's code and message. |

A venue's fill can still breach a limit: it is booked and announced as
`fill_breach` (in `on_risk_event`) with the limit named; a fill the book cannot
book yet is `fill_deferred`, journaled as `unbooked_fill`, and booked when it
can. See [execution](../execution/SKILL.md).

**A live node refused or stopped.**

| Code | Class | What happened and what to do |
| --- | --- | --- |
| `runtime_busy` | `StateError` | A push was made from inside a hook. Queue it and push after the call returns. |
| `node_finished`, `session_finished`, `run_finished` | `StateError` | The node or session is finished; create a new one. |
| `recovering` | `StateError` | The node was rebuilt and has not resumed. Market data and `finish` are refused. Reconcile, push the reports you have, `resend` what the venue does not hold, then `resume(accept=)`. |
| `fenced` | `StateError` | A newer epoch owns the account. This node sends nothing more and every push raises. Do not retry: stop the process; a supervisor decides who the writer is. |
| `journal_failed` | `StateError` | The journal could not take a record (disk full, sink raised); the node stopped before acting on it. Fix the sink and recover from what is durable. |
| `journal_evicted` | `StateError` | A bounded in-memory journal dropped records you asked for; read them from the durable sink. |
| `replay_diverged` | `StateError` | The replay made a record the journal does not hold: a different feed, symbols, warm-up, configuration or strategy, or a strategy that is not deterministic (wall clock, randomness, environment). Restore with exactly what wrote the journal. |
| `invalid_checkpoint` | `InvalidData` | The checkpoint is truncated, damaged, from another version, or was taken without its log and no `records=` were given. |
| `no_reconcile` | `UnsupportedOperation` | The adapter has none of `open_orders()`, `positions()`, `balances()`, so a restore cannot ask the venue. Add one, or pass `reconcile="skip"` knowingly (it is journaled). |
| `venue_unavailable` | `StateError` | A request may not have reached the venue; the order stays as it was until the venue reports. Ask the venue; do not resend blindly. |
| `venue_fill_not_booked` | `StateError` | The venue filled an order but the book could not take the fill: reconcile the position. |
| `correction_not_reversible` | `StateError` | A correction or bust of an exit, or of an entry whose position was since reduced, cannot be taken back exactly. Nothing changed; reconcile the position. |
| `node_poisoned` | `StateError` | A call stopped part way through a booking and the books cannot be trusted. Stop trading it; rebuild the node from the venue's positions (for example by seeding `initial_positions`). |
| `unknown_client_order_id` | `InvalidParameter` | A report names an order never sent. Key reports by the request's `client_order_id` (`s0-3`), not the strategy's id. It also refuses the fills of an exit the venue fired itself. |
| `invalid_execution_report` | `InvalidData` | A report contradicts the order: an over-fill, a repeated `exec_id` with other terms, a reject after a fill. The order is unchanged. |
| `duplicate_client_order_id` | `InvalidOrder` | An id was reused for a different order. Same id and terms is the same order and is not sent twice. |
| `order_check_intent` | `InvalidOrder` | `enter()` or `close_position()` with an `order_check` installed. Use `submit_order`, or `order_check_scope="typed"` when the check judges typed orders only. |
| `invalid_order_refusal`, `invalid_order_check`, `invalid_order_check_scope` | `InvalidParameter` | The check returned something other than `None`, a code (1 to 64 bytes) or an object with a `str` `code`; or more than 1024 distinct codes; or it is not callable; or the scope is not `"all"` or `"typed"`. An exception the check raises propagates and stops the run. |
| `unsupported_in_venue_run`, `venue_requires_units` | `UnsupportedOperation` / `InvalidOrder` | The request has no equivalent on a venue (for example an order made on warm-up history against a real venue), or its size could not be resolved to units. |
| `invalid_initial_positions`, `invalid_warmup`, `invalid_execution`, `invalid_adjustment_policy`, `invalid_corporate_action`, `unknown_role_binding`, `roll_target_not_subscribed` | `InvalidParameter` / `InvalidSubscription` | A node argument does not fit: see [live-trading](../live-trading/SKILL.md). |

**A requirements check or resolution found a problem** (these are `Issue`
values, not exceptions): `no_requirements`, `no_roles`, `missing_holding`,
`no_timeframes`, `unknown_role`, `invalid_base`, `undeclared_timeframe`,
`undeclared_instrument`, `undeclared_data`, `timeframes_without_bars`,
`dry_run_failed` (the strategy raised during the dry run: the message names the
exception), `dry_run_unresolved`; and from `resolve`: `unbound_role`,
`no_contract`, `no_options`, `no_expiry`, `needs_price`, `needs_vol`. A plan with
issues resolves nothing for the affected role.

**A broker adapter reported an event** (`adapter.drain_events()`).

| Event | Meaning |
| --- | --- |
| `unconfirmed` | A write may have been processed and the lookup by tag found nothing. The order stays `Submitted` for reconciliation. |
| `session_expired` | The token is no longer valid and no refresh is set; the outbox is paused until `set_token()`. |
| `payload_rejected`, `malformed_payload` | A push message failed verification, or lacked a quantity or price, and was dropped. |
| `fenced_drop` | Unsent jobs of an older epoch were dropped. |
| `venue_id_mismatch` | The broker sent a different venue id for an order with the same terms. |
| `connection_lost`, `connection_restored` | The feed gap; after a restore the adapter reads the order book. |

An adapter with no `namespace` raises `ValueError` at construction;
`handle_payload` on an unauthenticated channel raises `AdapterError`;
`fetch_*` / `orders()` / `positions()` raise `MalformedRow` rather than default a
missing number to zero.

## 4. Standard workflow

1. Read the error, not the traceback: `e.code`, `e.operation`, `e.parameter`,
   `e.value`, `e.valid`, `e.skill`.
2. Load the skill the error names, and find the code in the tables above.
3. Reproduce on the smallest tape or feed that shows it (five to ten bars).
4. For a live node, read `node.health()` and the journal before changing
   anything (section 5); never restart a node you have not looked at.
5. Fix the cause, then re-run the narrowest check that failed; for a skill
   example, `python -m raptorbt.skills check`.

## 5. Examples

Anatomy of an error. A NaN in a bar is refused before the run, names the column
and index, and the same object answers to `ValueError`:

```python
import numpy as np

from raptorbt import BacktestConfig, Strategy, run_strategy_backtest
from raptorbt.errors import InvalidData, RaptorBTError


class Idle(Strategy):
    pass


close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
tape = dict(timestamps=np.arange(5, dtype=np.int64) * 60_000_000_000, open=close, high=close + 1.0,
            low=close - 1.0, close=close.copy(), volume=np.full(5, 1000.0))
tape["close"][2] = float("nan")

try:
    run_strategy_backtest(Idle(), **tape, bar_stamp="close", config=BacktestConfig(fees=0.0))
    raise AssertionError("a NaN price must be refused")
except ValueError as e:                                  # the builtin it replaced still catches it
    assert isinstance(e, InvalidData) and isinstance(e, RaptorBTError)
    assert e.code == "non_finite_price" and e.parameter == "close" and e.value == 2
    assert e.skill == "market-data" and "index 2" in str(e).splitlines()[0]
    assert e.to_dict()["code"] == "non_finite_price"
```

An amount the money scale cannot hold stops the run with `money_overflow`
instead of wrapping (the engine also prints a one-line Rust panic notice to
standard error; the exception is the signal):

```python
import numpy as np

from raptorbt import BacktestConfig, Strategy, run_strategies
from raptorbt.errors import RaptorBTError


class Idle(Strategy):
    pass


close = np.array([100.0, 101.0, 102.0])
tape = dict(timestamps=np.arange(3, dtype=np.int64) * 60_000_000_000, open=close, high=close + 1.0,
            low=close - 1.0, close=close, volume=np.full(3, 1000.0))
try:
    run_strategies([Idle()], {"AAA": tape}, BacktestConfig(initial_capital=1e300), bar_stamp="close")
    raise AssertionError("1e300 cannot be booked at two decimals")
except RaptorBTError as e:
    assert e.code == "money_overflow" and isinstance(e, ArithmeticError)
```

Reading a journal. For any order, the story is the records that name it: the
decision (`command`), a refusal and the rejection it led to if there was one, the request and its result,
and every report the venue sent, with the times they were known. The strategy's
own id and the venue-facing id are tied together in the request's tags:

```python
from raptorbt import BacktestConfig, LiveNode, Strategy, journal as rj
from raptorbt.execution import ExecutionReport
from raptorbt.strategy import orders

MIN = 60_000_000_000


class Venue:
    namespace = "N1"
    def __init__(self):
        self.requests = []

    def submit(self, request):
        self.requests.append(request)

    def cancel(self, client_order_id):
        pass

    def modify(self, client_order_id, *, quantity=None, limit_price=None, trigger_price=None):
        pass


class Sender(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.submit_order(orders.Limit(side="buy", price=99.0, quantity=10), client_id="entry")
            self.submit_order(orders.Market(side="buy", quantity=999), client_id="big")


def check(order, ctx):
    return "too_big" if order.units and order.units > 100 else None


def story(records, strategy_id):
    """(seq, kind, what) for every record about the order the strategy calls `strategy_id`."""
    venue_id = next((r.client_order_id for r in records if r.kind == "adapter_request"
                     and r.payload["request"]["tags"].get("client_id") == strategy_id), None)
    rows = []
    for r in records:
        named = r.client_order_id in (strategy_id, venue_id) and r.client_order_id is not None
        in_batch = r.kind == "command" and any(c.get("client_id") == strategy_id for c in r.payload["commands"])
        if named or in_batch:
            detail = (r.payload.get("code") or r.payload.get("reason") or r.payload.get("status")
                      or r.payload.get("report") or r.payload.get("op"))
            rows.append((r.seq, r.kind, detail))
    return rows


journal, venue = rj.Journal({"run": "demo"}, memory=True), Venue()
node = LiveNode([Sender()], ["AAA"], BacktestConfig(fees=0.0), bar_stamp="close", execution=venue,
                journal=journal, order_check=check)
node.push_bar("AAA", MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)
cid = venue.requests[0].client_order_id
node.push_report(cid, ExecutionReport.accepted("V1", MIN), arrival_ns=MIN + 40)
node.push_report(cid, ExecutionReport.fill("X1", 4.0, 99.0, 2 * MIN), arrival_ns=2 * MIN + 90)

records = journal.since(0)
assert [(kind, detail) for _, kind, detail in story(records, "entry")] == [
    ("command", None), ("adapter_request", "submit"), ("adapter_result", "ok"),
    ("report", "accepted"), ("report", "fill")]
assert [(kind, detail) for _, kind, detail in story(records, "big")] == [
    ("command", None), ("refusal", "too_big"), ("rejection", "too_big")]   # the decision, then its outcome
fill = [r for r in records if r.kind == "report" and r.payload["report"] == "fill"][0]
assert (fill.ts_event, fill.ts_arrival) == (2 * MIN, 2 * MIN + 90)      # when it happened, when we knew
assert rj.verify(records, {"run": "demo"}) == journal.head()[1]         # the chain is intact
```

A supervisor's diagnosis from `health()`. A request that raised leaves the order
unconfirmed (visible with a journal); a fenced node says so:

```python
from raptorbt import BacktestConfig, LiveNode, Strategy, journal as rj
from raptorbt.errors import StateError
from raptorbt.execution import Fenced
from raptorbt.strategy import orders

MIN = 60_000_000_000


class Flaky:
    namespace = "N1"
    can_fence = True
    """A venue whose first request fails in the way that leaves the outcome unknown."""

    def __init__(self):
        self.failed, self.epoch = False, 1

    def submit(self, request):
        if not self.failed:
            self.failed = True
            raise ConnectionError("connection reset")

    def cancel(self, client_order_id):
        pass

    def modify(self, client_order_id, *, quantity=None, limit_price=None, trigger_price=None):
        pass

    def fence(self, epoch):
        if epoch < self.epoch:
            raise Fenced(f"epoch {self.epoch} owns the account")


class Buyer(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.submit_order(orders.Market(side="buy", quantity=1))


def diagnose(health):
    if health["fenced"]:
        return "stop: a newer epoch owns the account"
    if health["state"] == "recovering":
        return "reconcile, then resume"
    if health.get("unconfirmed"):
        return "ask the venue about " + ", ".join(health["unconfirmed"])
    return "ok"


venue = Flaky()
node = LiveNode([Buyer()], ["AAA"], BacktestConfig(fees=0.0), bar_stamp="close", execution=venue, epoch=1,
                journal=rj.Journal({"run": "demo"}, memory=True))
try:
    node.push_bar("AAA", MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)
except ConnectionError:
    pass                                                         # unknown outcome, not a refusal
assert diagnose(node.health()) == "ask the venue about s0-0"

venue.epoch = 2                                                  # somebody else took over
try:
    node.heartbeat()
except StateError as e:
    assert e.code == "fenced"
assert diagnose(node.health()) == "stop: a newer epoch owns the account"
```

## 6. Composition

- Each code's owning skill is in `e.skill` and in the catalogue: data in
  [market-data](../market-data/SKILL.md), orders in [orders](../orders/SKILL.md),
  limits in [risk](../risk/SKILL.md), venues in [execution](../execution/SKILL.md),
  nodes, journals and adapters in [live-trading](../live-trading/SKILL.md),
  admission and pre-live checks in [deployment](../deployment/SKILL.md).
- A wrong result without an error is usually a modelling choice, not a bug:
  fill timing, slippage, fees, `bar_stamp`, netting against hedging. Check those
  in [backtesting](../backtesting/SKILL.md) and [results](../results/SKILL.md)
  before suspecting the engine.

## 7. Constraints and correctness

- Branch on `code`, never on message text; messages improve between releases.
- An error leaves state unchanged; an exception from a hook or an adapter may
  not. After any exception from a push on a live node, read `health()` before
  pushing again.
- A journal is the evidence. Do not edit, truncate or "repair" a journal file by
  hand: `rj.read(path)` verifies checksums and the chain, and a torn last frame
  is cut by `open_for_recovery`, not by you.
- A discrepancy accepted without a fact is a decision; write it with
  `node.annotate(...)`.

**Engine gaps found while writing these skills** (state of the engine at the
time of writing; verified by running them; none is papered over above):

1. **A journal that holds a push which raised because an adapter's `submit`
   raised an unknown-outcome exception (for example `ConnectionError`) cannot be
   restored**: the replay raises `venue_unavailable` before it reproduces the
   `refusal` and `error` records the original wrote. Refusals (`OrderRefused`),
   `Unsupported` and an exception from a strategy replay correctly. The shipped
   broker adapters queue and return, so they do not raise this way; a custom
   adapter should queue the request and return, raising only `OrderRefused` or
   `Unsupported` from `submit`.
2. **`run_strategy_backtest` with an out-of-range `initial_capital`** raises the
   raw `pyo3_runtime.PanicException` (`money_overflow: ...`) instead of
   `MoneyOverflow`; `run_strategies` and `LiveNode` convert it. Catch
   `BaseException` at that one call if you accept untrusted capital, or validate
   the amount first.
3. `node.health()["unconfirmed"]` is empty on a node with no journal even when an
   order's request raised.
4. **Two `warm_up()` calls back to back that meet at one close time** (the
   second's first bar, of another symbol, has the close time of the first's last
   bar, and nothing was journaled between them) are journaled like one call, and a
   restore replays them as one. It raises nothing and `state_digest()` matches, but
   the strategy's hooks hear those bars differently (a hook sees the other symbol's
   bar), so state a strategy keeps itself may differ. Give such history in one call.
5. **A trailing stop from `BacktestConfig.set_trailing_stop` is reported as a
   stop**: its trade's `exit_reason` is `"StopLoss"`, the same as a fixed stop's
   (bar and tick runners). Tell them apart by the exit price against the fixed
   level, as `examples/tick-level/tick_scalper.py` does.

## 8. Common mistakes

Branching on the message instead of the code:

<!-- skill: invalid -->
```python
import numpy as np

from raptorbt import Strategy, run_strategy_backtest


class Idle(Strategy):
    pass


close = np.array([100.0, float("nan"), 102.0])
tape = dict(timestamps=np.arange(3, dtype=np.int64) * 60_000_000_000, open=close, high=close + 1.0,
            low=close - 1.0, close=close, volume=np.full(3, 1000.0))
try:
    run_strategy_backtest(Idle(), **tape, bar_stamp="close")
except ValueError as e:
    assert "invalid" in str(e).lower(), "the message says 'is not finite'; e.code says non_finite_price"
```

- Not passing `bar_stamp`: it is required (`missing_bar_stamp`). State it
  (`"open"` or `"close"`).
- Keying a venue report by the strategy's id (`"O-0"`) instead of the request's
  `client_order_id` (`"s0-0"`): `unknown_client_order_id`.
- Treating `order_rejected` as an error to catch. It is an event: read
  `event.reject_reason`.
- Using `enter()` or `close_position()` after installing an `order_check`.
- Expecting a resting order to fill on the bar it was placed; it matches from
  the next bar.
- Giving both `quantity` and `size_frac` (`conflicting_order_args`), or neither
  on an order that opens a position (`invalid_qty`).
- Pushing market data from a hook (`runtime_busy`), or after `finish`
  (`node_finished`).
- Waiting for a bar built from ticks after `advance_time`, or for `on_quote` after
  a tick with one side of the book: the bar closes on the symbol's next tick of a
  later window, and a quote needs both `bid` and `ask`
  ([market-data](../market-data/SKILL.md)).
- Restoring with a different strategy, feed or warm-up than wrote the journal
  (`replay_diverged`), or with a strategy that reads the clock.
- Resuming with `accept="all"` to get past `recovering`.
- Reusing an adapter `namespace` across runs, so a run reads another run's
  orders from the broker's day book.
- Comparing `pnl` across runs with different `money_decimals`, or fees with and
  without a `fee_schedule`.
- Passing `float` for money in a journal header or `state()`: records carry no
  floats, and `state()` must be JSON.

## 9. Validation

1. After a fix, re-run the smallest reproduction and assert the `code` it used
   to raise is gone, not that nothing raises.
2. `python -c "import raptorbt.errors as e; print(e.markdown())"` matches
   `docs/errors.md` (it is generated; do not edit it).
3. For a live-node incident: capture `node.health()`, copy the journal file
   before touching it, verify the chain with `raptorbt.journal.read`, and
   rebuild a test node from the copy with `LiveNode.restore` to reproduce.
4. `python -m raptorbt.skills check --strict` after editing any skill.

## 10. Related skills

- [Core](../core/SKILL.md)
- [Orders](../orders/SKILL.md)
- [Execution](../execution/SKILL.md)
- [Live trading](../live-trading/SKILL.md)
- [Deployment](../deployment/SKILL.md)
