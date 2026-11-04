# Errors

<!-- Generated from raptorbt.errors.CODES: regenerate with `python -c 'import raptorbt.errors as e; print(e.markdown(), end=)' > docs/errors.md`. -->

Every error RaptorBT raises on purpose is a `raptorbt.errors.RaptorBTError`.
It carries `code`, `operation`, `parameter`, `value`, `valid` (the accepted
alternatives, or the rule when the set is open), `skill` (the documentation
domain) and `to_dict()`.
The first line of `str(error)` is the short message; the lines after it list
what was requested, what is supported and the related skill.

Each class also subclasses the builtin it replaced, so `except ValueError`
(or `RuntimeError`, `KeyError`) keeps working.

| Class | Also a |
| --- | --- |
| `InvalidParameter` | `ValueError` |
| `InvalidType` | `TypeError` |
| `InvalidData` | `ValueError` |
| `InvalidOrder` | `ValueError` |
| `InvalidSubscription` | `ValueError` |
| `UnsupportedOperation` | `ValueError` |
| `StateError` | `RuntimeError` |
| `SessionFinished` | `RuntimeError`, `ValueError` |
| `UnknownSymbol` | `KeyError`, `LookupError`, `ValueError` |

## Codes

| Code | Class | Meaning | Alternatives | Related skill |
| --- | --- | --- | --- | --- |
| `account_too_large` | `InvalidParameter` | More strategies share one account than a group may hold. | at most 500 strategies per account | `raptorbt-portfolio` |
| `adapter_cannot_adopt` | `StateError` | restore() was given an adapter that cannot account for an order the node holds as working. | an adapter with restore_orders(rows); an adapter with keeps_own_records = True | `raptorbt-live-trading` |
| `adapter_namespace_mismatch` | `StateError` | restore() was given an adapter whose namespace is not the one the journal was written for. | the adapter whose namespace the journal records | `raptorbt-live-trading` |
| `adapter_namespace_missing` | `InvalidParameter` | An execution adapter must expose a non-empty namespace that marks the orders of one node. | a non-empty str namespace on the adapter | `raptorbt-live-trading` |
| `adoption_refused` | `InvalidParameter` | A position cannot be adopted as given: a short on a cash account, a posted margin on a cash account or a cost basis on a margin account. | a long position on a cash account; a short position or a posted margin on a margin account; a cost basis (cost_minor) on a cash account | `raptorbt-live-trading` |
| `below_one_lot` | `InvalidParameter` | The sizing rule yields less than one lot; nothing is rounded up or to zero. | more capital, equity or available margin; a smaller lot_size or price | `raptorbt-risk` |
| `bracket_under_hedging` | `InvalidOrder` | A reduce-only child order (a bracket's stop or target) was submitted under hedging, where every order opens its own position and a reduce-only order is always refused. | `"oms_type='netting'"`; stop_price and target_price on the entry order | `raptorbt-orders` |
| `conflicting_bar` | `StateError` | A bar was pushed for a symbol and timestamp that already has a bar with other values; a delivered bar is not revised. | a delivered bar is not revised: push the next bar | `raptorbt-live-trading` |
| `conflicting_order_args` | `InvalidOrder` | Two options that exclude each other were both given (e.g. units and size_frac). | units (or quantity) or size_frac, not both; every or every_bars, not both | `raptorbt-orders` |
| `correction_not_reversible` | `StateError` | A correction or bust cannot be applied to the book as it stands: reconcile the position. | reconcile the position with the venue | `raptorbt-live-trading` |
| `declaration_outside_run` | `StateError` | subscribe() or ctx.indicators was used outside on_init / on_start. | call subscribe() and ctx.indicators in on_init or on_start | `raptorbt-subscriptions` |
| `decreasing_timestamps` | `InvalidData` | Timestamps go backwards; bars must be in time order. | timestamps in non-decreasing order | `raptorbt-market-data` |
| `depth_without_ticks` | `InvalidData` | depth was given for a symbol that has no ticks. | ticks for every symbol that has depth | `raptorbt-market-data` |
| `duplicate_bar` | `StateError` | A bar was pushed again for a symbol and timestamp that already has an identical bar; the retry is refused, not delivered twice. | push each bar once | `raptorbt-live-trading` |
| `duplicate_client_order_id` | `InvalidOrder` | An order id was reused for a different order. | a new client_id per order; the same id only to repeat an identical order | `raptorbt-live-trading` |
| `duplicate_strategy` | `InvalidParameter` | The same strategy instance was passed twice in one call. | one instance per strategy: pass the class, or a new instance for each entry | `raptorbt-strategy` |
| `duplicate_symbols` | `InvalidParameter` | A symbol appears more than once. | each symbol once | `raptorbt-market-data` |
| `empty_data` | `InvalidData` | There is no data to run on (zero bars or rows). | at least one bar, tick or instrument | `raptorbt-market-data` |
| `empty_strategies` | `InvalidParameter` | The strategy list is empty. | at least one Strategy, (strategy, symbols) pair or entry dict | `raptorbt-strategy` |
| `epoch_required` | `StateError` | restore() of a journal that holds an epoch was given none: the node that wrote it would not be fenced. | epoch= greater than the journal's | `raptorbt-live-trading` |
| `exit_close_failed` | `StateError` | The closes of an exit rule were refused or cancelled the allowed number of times and the position is still open. | fix what the venue refuses (the refusal reasons are on the handle), then close the position yourself | `raptorbt-live-trading` |
| `exit_rule_not_cancellable` | `StateError` | An exit rule the venue holds cannot be taken back; it stays held. | an adapter with cancel_exit_rule(request); close the position | `raptorbt-live-trading` |
| `exit_rule_unconfirmed` | `StateError` | The venue did not say whether it holds, changed or dropped an exit rule. | ask the venue whether it holds the rule (reconcile) | `raptorbt-live-trading` |
| `fee_schedule_gap` | `InvalidParameter` | A fee schedule has no row for an instrument's or fill's venue and segment, or the instrument has no InstrumentSpec venue to choose a row from, or a fill or settlement falls on a date before the schedule's first row (the error names the earliest date); the run is refused rather than charged nothing or charged a rate that was not in force. | give the InstrumentSpec a venue the schedule prices; set config.fee_segment; add the missing row to the schedule | `raptorbt-execution` |
| `fence_unsupported` | `InvalidParameter` | epoch= was given with an adapter whose can_fence is not true, so the epoch would protect nothing. | an adapter with can_fence = True; `'epoch=None'` | `raptorbt-live-trading` |
| `fenced` | `StateError` | A newer execution epoch owns the account; this node sends nothing more. | stop this process: a newer epoch owns the account | `raptorbt-live-trading` |
| `group_open` | `StateError` | A timestamp group is open: flush() it before opening another, reconciling, resuming, reading health, taking a checkpoint or a state digest. | flush() the open group first | `raptorbt-live-trading` |
| `group_timestamp_mismatch` | `StateError` | A bar was pushed into an open group declared at one timestamp with another timestamp; it is not part of the group and is refused. | bars of the group's timestamp; flush() and begin_group(new timestamp) | `raptorbt-live-trading` |
| `hedging_unsupported_by_venue` | `InvalidParameter` | oms_type='hedging' was given with an adapter whose venue nets positions. | `'netting'` | `raptorbt-live-trading` |
| `history_exhausted` | `StateError` | A live node's history cannot hold a row: the ring is full of rows a strategy asked to keep, or the row was overwritten. | raise history= on the node; ask the strategy to keep fewer rows (subscribe(..., history=)) | `raptorbt-live-trading` |
| `incompatible_order_options` | `InvalidOrder` | A time-in-force or flag does not apply to this order kind. | tif at_open or at_close with a market order; post_only with a limit order; offset_kind='ticks' with an instrument that has a price_increment | `raptorbt-orders` |
| `instrument_in_use` | `StateError` | An instrument cannot be retired while a strategy holds a position or a working order on it. | close the position and cancel the working orders on the symbol, then retire it | `raptorbt-live-trading` |
| `invalid_account_spec` | `InvalidParameter` | An AccountSpec field has a value the account does not accept. | initial_capital > 0; account_type: cash, margin; leverage > 0; max_drawdown_pct and max_daily_loss_pct in (0, 100]; max_daily_loss, max_gross_exposure > 0; max_positions: a whole number > 0 | `raptorbt-portfolio` |
| `invalid_account_type` | `InvalidParameter` | account_type is not recognised. | `'cash'`; `'margin'` | `raptorbt-execution` |
| `invalid_accounts` | `InvalidParameter` | accounts is not a dict of account name -> AccountSpec. | dict[str, AccountSpec] | `raptorbt-portfolio` |
| `invalid_adjust` | `InvalidParameter` | continuous_series needs adjust to be 'raw', 'back' or 'ratio'. | `'raw'`; `'back'`; `'ratio'` | `raptorbt-market-data` |
| `invalid_adjustment_policy` | `InvalidParameter` | Corporate actions were given without an adjustment_policy, or the policy is not raw or adjusted. | `'raw'`; `'adjusted'` | `raptorbt-execution` |
| `invalid_allocation` | `InvalidParameter` | allocation is not recognised. | `'equal_weight'`; `'full'` | `raptorbt-portfolio` |
| `invalid_argument_type` | `InvalidType` | An argument or a strategy class is of the wrong type: a non-typed order to submit_order, a custom indicator without update/ready/value, or a hook defined under both its name and its deprecated alias. | a typed order (orders.Market, orders.Limit, ...); an indicator object with update(bar), ready and value; one spelling of a hook, not both | `raptorbt-strategy` |
| `invalid_array` | `InvalidData` | A field is not a numeric array of the required dimension. | a numeric 1-D numpy array (depth levels: 2-D) | `raptorbt-market-data` |
| `invalid_bar_ns` | `InvalidParameter` | bar_ns is not a positive whole number of nanoseconds. | a positive whole number of nanoseconds, e.g. 60_000_000_000 for one minute | `raptorbt-timeframes` |
| `invalid_bar_spec` | `InvalidSubscription` | A bar step or unit is not valid. | step >= 1; unit: ms, s, m, h, d, w, month, y, tick, volume, value, renko, tick_imbalance, tick_runs, volume_imbalance, volume_runs, value_imbalance, value_runs | `raptorbt-timeframes` |
| `invalid_bar_stamp` | `InvalidParameter` | bar_stamp is not 'open' or 'close'. | `'open'`; `'close'` | `raptorbt-timeframes` |
| `invalid_bar_step` | `InvalidSubscription` | A bar subscription step is below 1. | a whole number >= 1 | `raptorbt-timeframes` |
| `invalid_batch_size` | `InvalidParameter` | batch_size is not a positive integer or None. | a positive integer; `None` | `raptorbt-backtesting` |
| `invalid_binding` | `InvalidParameter` | A role is bound to the wrong shape: a Universe to anything but a list of distinct symbols, any other role to anything but one symbol. | one symbol (str) for an Underlying; a list of distinct symbols for a Universe | `raptorbt-live-trading` |
| `invalid_calendar` | `InvalidParameter` | A calendar document given to SessionSchedule.from_calendars is not valid JSON in the documented format, names no entry for the venue, or has overlapping or out-of-range windows or dates. | a JSON document with utc_offset and an entry for the venue; windows and dates inside range; one zone for every document | `raptorbt-instruments` |
| `invalid_checkpoint` | `InvalidData` | The checkpoint or journal records cannot be used to rebuild the node. | a checkpoint from LiveNode.checkpoint() of this version; records= for a checkpoint taken without its log | `raptorbt-live-trading` |
| `invalid_combine_mode` | `InvalidParameter` | combine_mode is not recognised. | `'any'`; `'all'`; `'majority'`; `'independent'`; `'weighted'` | `raptorbt-backtesting` |
| `invalid_config` | `InvalidParameter` | A configuration value is invalid (engine-side check). | a configuration value inside the range the message names | `raptorbt-core` |
| `invalid_corporate_action` | `InvalidParameter` | A corporate action is invalid: an unknown kind, a ratio that is not positive, a negative dividend, or a pay date before the ex-date. | kind: split, bonus, dividend; ratio > 0 for a split or bonus; amount >= 0 for a dividend; pay_ts at or after ts | `raptorbt-execution` |
| `invalid_curve_samples` | `InvalidParameter` | curve_samples is not a positive integer or None. | a positive integer; `None` | `raptorbt-live-trading` |
| `invalid_data` | `InvalidData` | Other malformed input data (division by zero, index out of range, conversion). | the arrays and fields the message names | `raptorbt-market-data` |
| `invalid_data_type` | `InvalidSubscription` | subscribe() was given a data_type it does not know. | `'bar'`; `'trade'`; `'quote'`; `'book'` | `raptorbt-subscriptions` |
| `invalid_direction` | `InvalidParameter` | direction is not 1 (long) or -1 (short). | `1`; `-1` | `raptorbt-backtesting` |
| `invalid_execution` | `InvalidParameter` | execution is not an execution adapter, or the run cannot send its orders to one. | `'None'`; `'SimulatedExecution()'`; `'PaperExecution()'`; an ExecutionAdapter (submit, cancel, modify) | `raptorbt-live-trading` |
| `invalid_execution_report` | `InvalidData` | A venue report contradicts the order it is about, or cannot be applied to it. | a report consistent with the order: no over-fill, a repeated exec_id with the same terms, no reject after a fill | `raptorbt-live-trading` |
| `invalid_exit_rule` | `InvalidParameter` | An exit rule is empty, has a non-positive or non-finite threshold, names no or several targets, or fired with no position. | ExitRule(stop_loss_pct | take_profit_pct | stop_loss | take_profit | premium_at_or_above | premium_at_or_below) with at least one condition; positive finite thresholds; stop_loss_pct <= 100 | `raptorbt-live-trading` |
| `invalid_expiry_pick` | `InvalidParameter` | OptionChain.pick_expiry was given an unknown rule, or 'date' without date_ns. | `'nearest'`; `'nth'`; `'weekly'`; `'monthly'`; `'date'` | `raptorbt-instruments` |
| `invalid_fee_schedule` | `InvalidParameter` | A fee schedule or rate card document is invalid, or fee_segment names a row the schedule does not have. | a fee schedule or rate card document in the documented format; `'FeeSchedule.exchange_presets()'` | `raptorbt-execution` |
| `invalid_fill_timing` | `InvalidParameter` | fill_timing is not recognised. | `'same_bar_close'`; `'next_bar_open'`; `'same_bar_open_lookahead'` | `raptorbt-execution` |
| `invalid_history` | `InvalidParameter` | A history or count is not a positive integer. | a positive integer | `raptorbt-subscriptions` |
| `invalid_indicator_input` | `InvalidParameter` | An indicator input field other than the close was requested. | `'close'` | `raptorbt-indicators` |
| `invalid_initial_positions` | `InvalidParameter` | initial_positions names a strategy or position that does not fit the node. | {strategy_index: {symbol: {'quantity': ..., 'avg_price': ..., 'margin_minor': ..., 'cost_minor': ...}}}; a strategy index and a symbol of the node | `raptorbt-live-trading` |
| `invalid_input` | `InvalidParameter` | An argument or array was refused by the computation it was passed to. | the argument shape and range the message names | `raptorbt-core` |
| `invalid_input_type` | `InvalidData` | A data, ticks or depth argument is not a dict of arrays. | `'dict'` | `raptorbt-market-data` |
| `invalid_instrument_metadata` | `InvalidParameter` | An instrument's venue or currency is an empty string. | a non-empty string; `None` | `raptorbt-instruments` |
| `invalid_instrument_spec` | `InvalidParameter` | The spec given for an instrument names another symbol. | an InstrumentSpec whose symbol is the symbol added | `raptorbt-live-trading` |
| `invalid_leverage` | `InvalidParameter` | Margin accounts need leverage > 0. | a number > 0 | `raptorbt-execution` |
| `invalid_margin_model` | `InvalidParameter` | A margin model or the line it prices is invalid: an unknown name, a missing rule for an instrument type, legs that do not fit, or a config.margin_model given to a cash account. | a model name from MarginModel (no_margin, cash_only, leverage, fraction_of_notional, option_simple, table, by_type, by_product, defined_risk, spread_credit, from_spec); a rule for every instrument type used; no margin_model on a cash account | `raptorbt-risk` |
| `invalid_money_decimals` | `InvalidParameter` | config.money_decimals is outside 0..=9: the decimals cash, fees and realised P&L are booked in. | a whole number from 0 to 9 | `raptorbt-execution` |
| `invalid_offset_kind` | `InvalidOrder` | Trailing offset kind is not recognised. | `'price'`; `'bps'`; `'ticks'` | `raptorbt-orders` |
| `invalid_oms_type` | `InvalidParameter` | oms_type is not recognised. | `'netting'`; `'hedging'` | `raptorbt-execution` |
| `invalid_option_right` | `InvalidParameter` | An option's right is not call or put. | `'call'`; `'put'` | `raptorbt-instruments` |
| `invalid_option_type` | `InvalidParameter` | option_type is not recognised. | `'call'`; `'ce'`; `'c'`; `'put'`; `'pe'`; `'p'` | `raptorbt-backtesting` |
| `invalid_order_check` | `InvalidParameter` | order_check is not callable. | a callable (order, ctx) -> None | str | OrderRefusal; `None` | `raptorbt-live-trading` |
| `invalid_order_check_scope` | `InvalidParameter` | order_check_scope is not 'all' or 'typed'. | `'all'`; `'typed'` | `raptorbt-live-trading` |
| `invalid_order_intent` | `InvalidOrder` | Intents queued on one bar are duplicated or unknown. | one enter() and one close_position() per bar; not both while in position | `raptorbt-orders` |
| `invalid_order_kind` | `InvalidOrder` | Order kind is not recognised. | `'market'`; `'limit'`; `'stop_market'`; `'stop_limit'`; `'market_if_touched'`; `'limit_if_touched'`; `'market_to_limit'`; `'trailing_stop_market'`; `'trailing_stop_limit'` | `raptorbt-orders` |
| `invalid_order_link` | `InvalidOrder` | A parent or one-cancels-other link cannot be made. | link_oco(a, b, ...) with at least two ids of orders of one symbol; parent: the client id of an order already submitted for the same symbol | `raptorbt-orders` |
| `invalid_order_reference` | `InvalidOrder` | An order or position id does not exist. | a position id from ctx.positions() | `raptorbt-orders` |
| `invalid_order_refusal` | `InvalidParameter` | order_check returned something other than None, a code, or an object with a str code; or the code is empty, over 64 bytes, or the 1025th distinct one. | None to accept; a str code of 1 to 64 bytes; an object with a str code; at most 1024 distinct codes | `raptorbt-live-trading` |
| `invalid_order_side` | `InvalidOrder` | Order side is not 'buy' or 'sell'. | `'buy'`; `'sell'` | `raptorbt-orders` |
| `invalid_order_tags` | `InvalidOrder` | An order's tags are not a dict of at most 16 str pairs (keys 1-64 bytes, values up to 256), or use the reserved key 'client_id'. | tags: dict of at most 16 str -> str pairs, keys 1-64 bytes, values up to 256, no 'client_id' | `raptorbt-orders` |
| `invalid_order_value` | `InvalidOrder` | A price, size, offset, protective level or count is out of range, NaN or infinite. | price, trigger, quantity, offset, stop_price, target_price, slices and every > 0 and finite; size_frac in (0, 1]; limit_offset >= 0 | `raptorbt-orders` |
| `invalid_parameter` | `InvalidParameter` | A parameter value is invalid (engine-side check). | the argument the message names | `raptorbt-core` |
| `invalid_primary_bars` | `InvalidSubscription` | primary_bars is not a valid (step, unit) pair. | (step, unit) with step >= 1 and unit a bar unit, e.g. (5, 'm') | `raptorbt-timeframes` |
| `invalid_roll_sizing` | `InvalidParameter` | roll_sizing is not 'notional' or 'lots'. | `'notional'`; `'lots'` | `raptorbt-instruments` |
| `invalid_session_date` | `InvalidParameter` | A session date is not a real yyyymmdd date. | a real calendar date as yyyymmdd | `raptorbt-instruments` |
| `invalid_session_schedule` | `InvalidParameter` | A SessionSchedule is invalid: a bad window, weekday, holiday date, offset or square-off minute. | windows: {'mon'..'sun' or 0..6: [(open_minute, close_minute), ...]} with 0 <= open < close <= 1440; holidays: yyyymmdd ints; squareoff_minute inside a window | `raptorbt-instruments` |
| `invalid_settlement` | `InvalidParameter` | An instrument's settlement style or settlement price is invalid. | settlement_style: cash, physical; settlement_price: auto, last_price, underlying, or a number >= 0 | `raptorbt-instruments` |
| `invalid_size_type` | `InvalidParameter` | size_type is not recognised. | `'percent'`; `'contracts'`; `'notional'`; `'risk'` | `raptorbt-backtesting` |
| `invalid_sizing_input` | `InvalidParameter` | A sizing input is not a positive finite number (or a fraction out of range). | a positive finite number (fractions in (0, 1], percent in (0, 100]) | `raptorbt-risk` |
| `invalid_squareoff_time` | `InvalidParameter` | squareoff_time is not a 24-hour 'HH:MM' string or None. | 'HH:MM' (24-hour, local), e.g. '15:25'; `None` | `raptorbt-execution` |
| `invalid_strategies` | `InvalidParameter` | strategies must name existing strategies of the node, each once. | indexes of the node's strategies, each once | `raptorbt-live-trading` |
| `invalid_strategy` | `InvalidParameter` | A strategy entry is not a Strategy instance or subclass. | Strategy instance; Strategy subclass; (strategy, symbols) pair | `raptorbt-strategy` |
| `invalid_strategy_entry` | `InvalidParameter` | A strategy entry is a tuple that is not (strategy, symbols). | `'strategy'`; (strategy, symbols); (strategy, symbols, {'account': name}); {'strategy': s, 'symbols': [...], 'account': name} | `raptorbt-strategy` |
| `invalid_strike_selection` | `InvalidParameter` | strike_selection is not recognised. | `'atm'`; `'otm1'`; `'otm2'`; `'itm1'`; `'itm2'` | `raptorbt-backtesting` |
| `invalid_structure` | `InvalidParameter` | A structure, leg, selector, exit or submission argument is invalid. | the leg, selector, exit or submission argument the message names | `raptorbt-instruments` |
| `invalid_time_in_force` | `InvalidOrder` | Time in force is not recognised. | `'auto'`; `'gtc'`; `'day'`; `'gtd'`; `'ioc'`; `'fok'`; `'at_open'`; `'at_close'` | `raptorbt-orders` |
| `invalid_timeframe` | `InvalidSubscription` | A timeframe is not one of the supported strings or a valid bar spec. | `'1m'`; `'3m'`; `'5m'`; `'10m'`; `'15m'`; `'30m'`; `'1h'`; `'2h'`; `'4h'`; `'1d'`; `'1w'` | `raptorbt-timeframes` |
| `invalid_trade_from` | `InvalidParameter` | trade_from is not an integer event time in nanoseconds. | integer nanoseconds since the epoch; `None` | `raptorbt-backtesting` |
| `invalid_warmup` | `InvalidParameter` | warmup is not 'live', 'muted' or None. | `'live'`; `'muted'`; `None` | `raptorbt-live-trading` |
| `journal_evicted` | `StateError` | A journal read asked for records a bounded memory has dropped; read them from the durable sink. | read the records from the durable sink; a larger memory= bound | `raptorbt-live-trading` |
| `journal_failed` | `StateError` | The journal could not take a record, so the node stopped before acting on it. | fix the sink and LiveNode.restore from what is durable | `raptorbt-live-trading` |
| `length_mismatch` | `InvalidData` | Arrays that must share one length do not. | every array the length of timestamps | `raptorbt-market-data` |
| `mcx_session_unknown` | `InvalidParameter` | An MCX instrument with no session names no known contract class (select agricultural COTTON, KAPAS, COTWASOIL at 09:00-21:00; other agricultural at 09:00-17:00; bullion, base metals, energy), so no session is implied: pass session=. | pass session= (a SessionSchedule); name an underlying of a known contract class (COTTON, KAPAS, COTWASOIL: 09:00-21:00; other agricultural: 09:00-17:00; bullion, base metals, energy) | `raptorbt-instruments` |
| `misaligned_legs` | `InvalidData` | Legs traded bar by bar (a basket, a pair) do not share their timestamps. | the same timestamps on every leg | `raptorbt-market-data` |
| `missing_bar_ns` | `InvalidParameter` | bar_stamp='open' was given without the bar width bar_ns. | bar_ns=<bar width in nanoseconds>, or a dict symbol -> nanoseconds; bar_stamp='close' (needs no width) | `raptorbt-timeframes` |
| `missing_bar_stamp` | `InvalidParameter` | Bars were given without bar_stamp, which is required. | `'open'`; `'close'` | `raptorbt-timeframes` |
| `missing_field` | `InvalidData` | A required field is missing from a symbol's arrays. | bars: timestamps, open, high, low, close, volume; ticks: timestamps, ltp; depth: timestamps, bid_prices, bid_sizes, ask_prices, ask_sizes | `raptorbt-market-data` |
| `missing_order_field` | `InvalidOrder` | A field this order kind or time in force needs was not given. | limit_price for limit kinds; trigger_price for stop and touched kinds; offset for trailing kinds; expire_ns for tif='gtd'; units for a Twap | `raptorbt-orders` |
| `mixed_holding` | `InvalidParameter` | Strategies in one run declare different holding requirements (intraday with different flatten times, or intraday beside overnight), so no single square-off time can be derived; set config.squareoff_time or run them apart. | `'config.squareoff_time'`; run the strategies apart | `raptorbt-execution` |
| `money_overflow` | `MoneyOverflow` | A booked amount is not finite or a balance or sum is past the integer range of minor units; the run stops instead of wrapping. | amounts that are finite and inside the range of minor units; a smaller config.money_decimals | `raptorbt-execution` |
| `no_continuous_bars` | `InvalidData` | A continuous series has no bars in a contract's window, or an older contract has none before a roll. | bars for each contract inside its own window | `raptorbt-market-data` |
| `no_instrument_spec` | `UnknownSymbol` | ctx.instrument was asked for a symbol with no InstrumentSpec. | instruments={symbol: InstrumentSpec}; a role name the strategy declares | `raptorbt-instruments` |
| `no_reconcile` | `UnsupportedOperation` | The adapter cannot say what the venue holds (no open_orders, positions or balances). | an adapter with open_orders(), positions() or balances(); reconcile='skip' (journaled) | `raptorbt-live-trading` |
| `no_requirements` | `InvalidParameter` | ctx.role was used by a strategy that declares no requirements. | declare requirements = Requirements(roles={...}) on the strategy | `raptorbt-instruments` |
| `no_symbols` | `InvalidParameter` | No symbols were given, or the data/ticks dict is empty. | at least one symbol | `raptorbt-market-data` |
| `node_finished` | `StateError` | The live node is finished; create a new one. | create a new LiveNode, or LiveNode.restore from the journal | `raptorbt-live-trading` |
| `node_poisoned` | `StateError` | A call stopped part way through a booking; the node's books cannot be trusted. Rebuild it from the venue's positions. | discard the node and rebuild it from the venue's positions (initial_positions=) | `raptorbt-live-trading` |
| `non_finite_price` | `InvalidData` | A bar's open, high, low or close is NaN or infinite; the message names the column and first index. | finite open, high, low and close | `raptorbt-market-data` |
| `not_subscribed_data` | `InvalidSubscription` | ctx.data was asked for quotes or books the strategy did not subscribe to. | subscribe(instrument, data_type='quote' or 'book') in on_init or on_start | `raptorbt-subscriptions` |
| `not_subscribed_timeframe` | `InvalidSubscription` | ctx.data was asked for a timeframe the strategy did not subscribe to. | subscribe(instrument, timeframe=...) in on_init or on_start | `raptorbt-subscriptions` |
| `off_lot_quantity` | `InvalidOrder` | An order's quantity is not a whole number of the instrument's lots (or size steps). It is refused in every mode so the simulation and a venue see the same terms; `valid` lists the nearest valid quantities. | a whole number of lots (the error lists the nearest) | `raptorbt-orders` |
| `off_tick_price` | `InvalidOrder` | A limit, trigger, stop or target price is not a whole number of the instrument's ticks. It is refused in every mode so the simulation and a venue see the same terms; `valid` lists the nearest valid prices. | a whole number of ticks (the error lists the nearest) | `raptorbt-orders` |
| `order_check_intent` | `InvalidOrder` | enter() or close_position() was used with an order_check installed; they carry no order for it to see. | submit_order(...) with a typed order, not enter() or close_position(); order_check_scope="typed" when the check judges typed orders only | `raptorbt-live-trading` |
| `order_validity_unsupported` | `InvalidOrder` | A gtc or gtd order on an instrument whose exchange segment holds only day orders (NSE equity derivatives, currency, BSE, MCX), or a gtd dated past the contract's maturity where the segment holds them, or a day order on an instrument whose venue no session preset covers and that has no session. | `'day'`; `'ioc'`; `'fok'` | `raptorbt-orders` |
| `portfolio_total_unavailable` | `UnsupportedOperation` | A ctx.portfolio total needs the shared runtime (run_strategies). | `'run_strategies'`; `'run_portfolio_strategy'` | `raptorbt-portfolio` |
| `primary_bars_without_ticks` | `InvalidSubscription` | primary_bars was given but there are no ticks to aggregate. | ticks={symbol: {'timestamps': ..., 'ltp': ...}}; primary_bars=None to run on the bars given | `raptorbt-subscriptions` |
| `recovering` | `StateError` | The node was rebuilt from its journal and has not resumed: reconcile with the venue, then resume. | reconcile(), push the reports you hold, resend what the venue lacks, then resume() | `raptorbt-live-trading` |
| `replay_diverged` | `StateError` | A replay produced a record that is not the one the journal holds: the node is not the one that wrote it, or a strategy is not deterministic. | restore with the same feed, symbols, warmup, config and strategies that wrote the journal; deterministic strategies (no clock, randomness or environment) | `raptorbt-live-trading` |
| `roll_close_failed` | `StateError` | The close of an auto-roll ended without filling, so the open in the new contract was not sent. | fix what the venue refused, then close the position in the old contract yourself; override on_roll_failed | `raptorbt-live-trading` |
| `roll_open_failed` | `StateError` | The open of an auto-roll ended without filling: the position is closed in the old contract and open in neither. | open the position in the new contract yourself; override on_roll_failed | `raptorbt-live-trading` |
| `roll_price_missing` | `StateError` | An auto-roll sized by notional could not price the new contract: it has no bar yet. | feed the new contract a bar before the roll; `"roll_sizing='lots'"` | `raptorbt-instruments` |
| `roll_target_not_subscribed` | `InvalidSubscription` | A continuous role rolled to a contract that is not a symbol of the run. | include the next contract among the symbols; give a LiveNode the contract's InstrumentSpec | `raptorbt-live-trading` |
| `run_finished` | `StateError` | The strategy's run is finished. | start a new run | `raptorbt-backtesting` |
| `runtime_busy` | `StateError` | The runtime is in use; a call re-entered it from a hook. | queue the push and make it after the hook returns | `raptorbt-backtesting` |
| `session_finished` | `SessionFinished` | The session is finished; create a new one. | create a new session | `raptorbt-backtesting` |
| `session_not_covered` | `InvalidData` | A bar falls on a date outside those a session schedule covers (the MCX preset, or a schedule loaded from calendar documents), so whether the market traded is unknown, not closed; the date and the covered ranges are named. | load a calendar document for the period (SessionSchedule.from_calendars); bars inside the dates the schedule covers | `raptorbt-instruments` |
| `squareoff_needs_timezone` | `InvalidParameter` | config.squareoff_time is a local time of day but the instrument has no session schedule and the config names no session_tz_offset_ns. | config.session_tz_offset_ns (IST_OFFSET_NS, or 0 for UTC); an InstrumentSpec with a SessionSchedule | `raptorbt-execution` |
| `stale_epoch` | `StateError` | restore() was given an epoch that is not newer than the one the journal already holds. | an epoch greater than the journal's | `raptorbt-live-trading` |
| `structure_contract_missing` | `InvalidParameter` | The chain lists no contract (or expiry) a structure's leg selects at the event time. | a strike, expiry or right the chain lists at the event time | `raptorbt-instruments` |
| `structure_limit_breached` | `InvalidParameter` | The legs' prices make the net premium worse than net_limit. | a net_limit the legs' prices can meet; order_type='market' without net_limit | `raptorbt-orders` |
| `structure_margin_exceeded` | `InvalidParameter` | The margin model needs more for the group than is available. | fewer lots; more available margin; a structure that needs less margin | `raptorbt-risk` |
| `structure_missing_mark` | `StateError` | A price or mark a structure needs was not supplied. | marks: a price for every leg's symbol | `raptorbt-instruments` |
| `universe_instrument` | `InvalidParameter` | ctx.instrument was given a Universe role, which names many symbols; ask for one of them. | one symbol of the role: ctx.role(name) lists them | `raptorbt-instruments` |
| `universe_size` | `InvalidParameter` | A Universe role is bound to fewer symbols than its min or more than its max. | a list with between the role's min and max symbols | `raptorbt-live-trading` |
| `unknown_account` | `InvalidParameter` | A strategy or call names an account that is not in accounts. | a name from accounts={name: AccountSpec(...)} | `raptorbt-portfolio` |
| `unknown_client_order_id` | `InvalidParameter` | A report names an order that was never sent. | the client_order_id of a request the node sent (s0-3), as the adapter received it | `raptorbt-live-trading` |
| `unknown_order` | `InvalidOrder` | A cancel or modify named a client id this run never submitted. A submitted order that has finished is a documented no-op, not this error. | a client id this run submitted (the error lists them) | `raptorbt-orders` |
| `unknown_role` | `UnknownSymbol` | ctx.role was asked for a role the strategy's requirements do not declare. | a role the strategy's requirements declare | `raptorbt-instruments` |
| `unknown_role_binding` | `InvalidParameter` | bindings names a role that no strategy declares. | a role some strategy declares in requirements | `raptorbt-live-trading` |
| `unknown_skill` | `UnknownSymbol` | raptorbt.skills.load was asked for a skill that does not exist. | a name from raptorbt.skills.index() | `raptorbt-core` |
| `unknown_symbol` | `UnknownSymbol` | A symbol was named that has no data in this run. | a symbol the run has data for | `raptorbt-market-data` |
| `unresolved_role` | `UnknownSymbol` | ctx.role could not resolve a requirements role at this time. | supply the data or instrument specs the role needs; bindings={role: symbol} | `raptorbt-instruments` |
| `unsupported_corporate_actions` | `UnsupportedOperation` | corporate_actions were given to a runner that cannot apply them; they are refused, never ignored. | `'run_single_backtest'`; `'run_strategy_backtest'`; `'run_strategies'` | `raptorbt-execution` |
| `unsupported_in_venue_run` | `UnsupportedOperation` | The request has no equivalent when orders go to an execution adapter. | do not trade on warmup bars (warmup='muted'); `'PaperExecution'`; resend only orders the node sent | `raptorbt-live-trading` |
| `unsupported_on_bars` | `UnsupportedOperation` | A setting that only the tick path implements (order_latency_ns, partial_fills) was given to a run that has bars; on bars it would change nothing. | run on ticks (ticks=); leave order_latency_ns at 0 and partial_fills False | `raptorbt-core` |
| `unsupported_short_ticks` | `UnsupportedOperation` | Tick backtests are long-only; quantity must be >= 0. | quantity >= 0; run_strategies with ticks for shorts | `raptorbt-backtesting` |
| `untradable_instrument` | `UnsupportedOperation` | The instrument kind cannot be traded. | a tradable kind: equity, contract, option, currency pair | `raptorbt-instruments` |
| `venue_fill_not_booked` | `StateError` | The venue filled an order but the book could not take the fill: reconcile the position. | reconcile the position with the venue (LiveNode.reconcile) | `raptorbt-live-trading` |
| `venue_requires_units` | `InvalidOrder` | An order sent to an execution adapter gave its size as a capital fraction or the whole position. | units= (a quantity), not size_frac or the whole position | `raptorbt-live-trading` |
| `venue_unavailable` | `StateError` | A request may not have reached the venue; the order stays as it was until the venue reports. | ask the venue what it holds (reconcile); resend only an order the venue does not hold | `raptorbt-live-trading` |
