# Errors

<!-- Generated from raptorbt.errors.CODES: regenerate with `python -c 'import raptorbt.errors as e; print(e.markdown(), end=)' > docs/errors.md`. -->

Every error RaptorBT raises on purpose is a `raptorbt.errors.RaptorBTError`.
It carries `code`, `operation`, `parameter`, `value`, `valid` (the accepted
alternatives, or `None`), `skill` (the documentation domain) and `to_dict()`.
The first line of `str(error)` is the short message; the lines after it list
what was requested, what is supported and the related skill.

Each class also subclasses the builtin it replaced, so `except ValueError`
(or `RuntimeError`, `KeyError`) keeps working.

| Class | Also a |
| --- | --- |
| `InvalidParameter` | `ValueError` |
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
| `account_too_large` | `InvalidParameter` | More strategies share one account than a group may hold. | depends on the call | `raptorbt-portfolio` |
| `below_one_lot` | `InvalidParameter` | The sizing rule yields less than one lot; nothing is rounded up or to zero. | depends on the call | `raptorbt-risk` |
| `conflicting_order_args` | `InvalidOrder` | Two options that exclude each other were both given (e.g. units and size_frac). | depends on the call | `raptorbt-orders` |
| `declaration_outside_run` | `StateError` | subscribe() or ctx.indicators was used outside on_init / on_start. | depends on the call | `raptorbt-subscriptions` |
| `decreasing_timestamps` | `InvalidData` | Timestamps go backwards; bars must be in time order. | depends on the call | `raptorbt-market-data` |
| `depth_without_ticks` | `InvalidData` | depth was given for a symbol that has no ticks. | depends on the call | `raptorbt-market-data` |
| `duplicate_client_order_id` | `InvalidOrder` | An order id was reused for a different order. | depends on the call | `raptorbt-live-trading` |
| `duplicate_strategy` | `InvalidParameter` | The same strategy instance was passed twice in one call. | depends on the call | `raptorbt-strategy` |
| `duplicate_symbols` | `InvalidParameter` | A symbol appears more than once. | depends on the call | `raptorbt-market-data` |
| `empty_data` | `InvalidData` | There is no data to run on (zero bars or rows). | depends on the call | `raptorbt-market-data` |
| `empty_strategies` | `InvalidParameter` | The strategy list is empty. | depends on the call | `raptorbt-strategy` |
| `fee_schedule_gap` | `InvalidParameter` | A fee schedule has no row for the fill's venue and segment; the run is refused rather than charged nothing. | depends on the call | `raptorbt-execution` |
| `incompatible_order_options` | `InvalidOrder` | A time-in-force or flag does not apply to this order kind. | depends on the call | `raptorbt-orders` |
| `insufficient_data` | `InvalidData` | Fewer elements than the calculation needs. | depends on the call | `raptorbt-market-data` |
| `invalid_account_spec` | `InvalidParameter` | An AccountSpec field has a value the account does not accept. | depends on the call | `raptorbt-portfolio` |
| `invalid_account_type` | `InvalidParameter` | account_type is not recognised. | `'cash'`, `'margin'` | `raptorbt-execution` |
| `invalid_accounts` | `InvalidParameter` | accounts is not a dict of account name -> AccountSpec. | depends on the call | `raptorbt-portfolio` |
| `invalid_adjustment_policy` | `InvalidParameter` | Corporate actions were given without an adjustment_policy, or the policy is not raw or adjusted. | depends on the call | `raptorbt-execution` |
| `invalid_allocation` | `InvalidParameter` | allocation is not recognised. | `'equal_weight'`, `'full'` | `raptorbt-portfolio` |
| `invalid_array` | `InvalidData` | A field is not a numeric array of the required dimension. | depends on the call | `raptorbt-market-data` |
| `invalid_bar_ns` | `InvalidParameter` | bar_ns is not a positive whole number of nanoseconds. | depends on the call | `raptorbt-timeframes` |
| `invalid_bar_spec` | `InvalidSubscription` | A bar step or unit is not valid. | depends on the call | `raptorbt-timeframes` |
| `invalid_bar_stamp` | `InvalidParameter` | bar_stamp is not 'open', 'close' or None. | `'open'`, `'close'`, `None` | `raptorbt-timeframes` |
| `invalid_bar_step` | `InvalidSubscription` | A bar subscription step is below 1. | depends on the call | `raptorbt-timeframes` |
| `invalid_batch_size` | `InvalidParameter` | batch_size is not a positive integer or None. | depends on the call | `raptorbt-backtesting` |
| `invalid_config` | `InvalidParameter` | A configuration value is invalid (engine-side check). | depends on the call | `raptorbt-core` |
| `invalid_corporate_action` | `InvalidParameter` | A corporate action is invalid: an unknown kind, a ratio that is not positive, a negative dividend, or a pay date before the ex-date. | depends on the call | `raptorbt-execution` |
| `invalid_curve_samples` | `InvalidParameter` | curve_samples is not a positive integer or None. | depends on the call | `raptorbt-live-trading` |
| `invalid_data` | `InvalidData` | Other malformed input data (division by zero, index out of range, conversion). | depends on the call | `raptorbt-market-data` |
| `invalid_data_type` | `InvalidSubscription` | subscribe() was given a data_type it does not know. | `'bar'`, `'trade'`, `'quote'`, `'book'` | `raptorbt-subscriptions` |
| `invalid_direction` | `InvalidParameter` | direction is not 1 (long) or -1 (short). | `1`, `-1` | `raptorbt-backtesting` |
| `invalid_execution` | `InvalidParameter` | execution is not an execution adapter, or the run cannot send its orders to one. | depends on the call | `raptorbt-live-trading` |
| `invalid_execution_report` | `InvalidData` | A venue report contradicts the order it is about, or cannot be applied to it. | depends on the call | `raptorbt-live-trading` |
| `invalid_expiry_pick` | `InvalidParameter` | OptionChain.pick_expiry was given an unknown rule, or 'date' without date_ns. | `'nearest'`, `'nth'`, `'weekly'`, `'monthly'`, `'date'` | `raptorbt-instruments` |
| `invalid_fee_schedule` | `InvalidParameter` | A fee schedule or rate card document is invalid, or fee_schedule is set without a fee_segment. | depends on the call | `raptorbt-execution` |
| `invalid_fill_timing` | `InvalidParameter` | fill_timing is not recognised. | `'same_bar_close'`, `'next_bar_open'`, `'same_bar_open_lookahead'` | `raptorbt-execution` |
| `invalid_history` | `InvalidParameter` | A history or count is not a positive integer. | depends on the call | `raptorbt-subscriptions` |
| `invalid_indicator_input` | `InvalidParameter` | An indicator input field other than the close was requested. | `'close'` | `raptorbt-indicators` |
| `invalid_initial_positions` | `InvalidParameter` | initial_positions names a strategy or position that does not fit the node. | depends on the call | `raptorbt-live-trading` |
| `invalid_input_type` | `InvalidData` | A data, ticks or depth argument is not a dict of arrays. | depends on the call | `raptorbt-market-data` |
| `invalid_instrument_metadata` | `InvalidParameter` | An instrument's venue or currency is an empty string. | depends on the call | `raptorbt-instruments` |
| `invalid_leverage` | `InvalidParameter` | Margin accounts need leverage > 0. | depends on the call | `raptorbt-execution` |
| `invalid_margin_model` | `InvalidParameter` | A margin model or the line it prices is invalid: an unknown name, a missing rule for an instrument type, legs that do not fit, or a config.margin_model given to a cash account. | depends on the call | `raptorbt-risk` |
| `invalid_money_decimals` | `InvalidParameter` | config.money_decimals is outside 0..=9: the decimals cash, fees and realised P&L are booked in. | depends on the call | `raptorbt-execution` |
| `invalid_offset_kind` | `InvalidOrder` | Trailing offset kind is not recognised. | `'price'`, `'bps'`, `'ticks'` | `raptorbt-orders` |
| `invalid_oms_type` | `InvalidParameter` | oms_type is not recognised. | `'netting'`, `'hedging'` | `raptorbt-execution` |
| `invalid_option_right` | `InvalidParameter` | An option's right is not call or put. | `'call'`, `'put'` | `raptorbt-instruments` |
| `invalid_option_type` | `InvalidParameter` | option_type is not recognised. | `'call'`, `'ce'`, `'c'`, `'put'`, `'pe'`, `'p'` | `raptorbt-backtesting` |
| `invalid_order_intent` | `InvalidOrder` | Intents queued on one bar are duplicated or unknown. | depends on the call | `raptorbt-orders` |
| `invalid_order_kind` | `InvalidOrder` | Order kind is not recognised. | `'market'`, `'limit'`, `'stop_market'`, `'stop_limit'`, `'market_if_touched'`, `'limit_if_touched'`, `'market_to_limit'`, `'trailing_stop_market'`, `'trailing_stop_limit'` | `raptorbt-orders` |
| `invalid_order_link` | `InvalidOrder` | A parent or one-cancels-other link cannot be made. | depends on the call | `raptorbt-orders` |
| `invalid_order_reference` | `InvalidOrder` | An order or position id does not exist. | depends on the call | `raptorbt-orders` |
| `invalid_order_side` | `InvalidOrder` | Order side is not 'buy' or 'sell'. | `'buy'`, `'sell'` | `raptorbt-orders` |
| `invalid_order_value` | `InvalidOrder` | A price, size, offset or count is out of range. | depends on the call | `raptorbt-orders` |
| `invalid_parameter` | `InvalidParameter` | A parameter value is invalid (engine-side check). | depends on the call | `raptorbt-core` |
| `invalid_primary_bars` | `InvalidSubscription` | primary_bars is not a valid (step, unit) pair. | depends on the call | `raptorbt-timeframes` |
| `invalid_session_date` | `InvalidParameter` | A session date is not a real yyyymmdd date. | depends on the call | `raptorbt-instruments` |
| `invalid_session_schedule` | `InvalidParameter` | A SessionSchedule is invalid: a bad window, weekday, holiday date, offset or square-off minute. | depends on the call | `raptorbt-instruments` |
| `invalid_settlement` | `InvalidParameter` | An instrument's settlement style or settlement price is invalid. | depends on the call | `raptorbt-instruments` |
| `invalid_size_type` | `InvalidParameter` | size_type is not recognised. | `'percent'`, `'contracts'`, `'notional'`, `'risk'` | `raptorbt-backtesting` |
| `invalid_sizing_input` | `InvalidParameter` | A sizing input is not a positive finite number (or a fraction out of range). | depends on the call | `raptorbt-risk` |
| `invalid_squareoff_time` | `InvalidParameter` | squareoff_time is not a 24-hour 'HH:MM' string or None. | depends on the call | `raptorbt-execution` |
| `invalid_strategy` | `InvalidParameter` | A strategy entry is not a Strategy instance or subclass. | depends on the call | `raptorbt-strategy` |
| `invalid_strategy_entry` | `InvalidParameter` | A strategy entry is a tuple that is not (strategy, symbols). | depends on the call | `raptorbt-strategy` |
| `invalid_strike_selection` | `InvalidParameter` | strike_selection is not recognised. | `'atm'`, `'otm1'`, `'otm2'`, `'itm1'`, `'itm2'` | `raptorbt-backtesting` |
| `invalid_structure` | `InvalidParameter` | A structure, leg, selector, exit or submission argument is invalid. | depends on the call | `raptorbt-instruments` |
| `invalid_time_in_force` | `InvalidOrder` | Time in force is not recognised. | `'gtc'`, `'day'`, `'gtd'`, `'ioc'`, `'fok'`, `'at_open'`, `'at_close'` | `raptorbt-orders` |
| `invalid_timeframe` | `InvalidSubscription` | A timeframe is not one of the supported strings or a valid bar spec. | `'1m'`, `'3m'`, `'5m'`, `'10m'`, `'15m'`, `'30m'`, `'1h'`, `'2h'`, `'4h'`, `'1d'`, `'1w'` | `raptorbt-timeframes` |
| `length_mismatch` | `InvalidData` | Arrays that must share one length do not. | depends on the call | `raptorbt-market-data` |
| `missing_bar_ns` | `InvalidParameter` | bar_stamp='open' was given without the bar width bar_ns. | depends on the call | `raptorbt-timeframes` |
| `missing_field` | `InvalidData` | A required field is missing from a symbol's arrays. | depends on the call | `raptorbt-market-data` |
| `missing_order_field` | `InvalidOrder` | A field this order kind or time in force needs was not given. | depends on the call | `raptorbt-orders` |
| `money_overflow` | `MoneyOverflow` | A booked amount is not finite or a balance or sum is past the integer range of minor units; the run stops instead of wrapping. | depends on the call | `raptorbt-execution` |
| `no_symbols` | `InvalidParameter` | No symbols were given, or the data/ticks dict is empty. | depends on the call | `raptorbt-market-data` |
| `node_finished` | `StateError` | The live node is finished; create a new one. | depends on the call | `raptorbt-live-trading` |
| `non_finite_price` | `InvalidData` | A bar's open, high, low or close is NaN or infinite; the message names the column and first index. | depends on the call | `raptorbt-market-data` |
| `not_subscribed_data` | `InvalidSubscription` | ctx.data was asked for quotes or books the strategy did not subscribe to. | depends on the call | `raptorbt-subscriptions` |
| `not_subscribed_timeframe` | `InvalidSubscription` | ctx.data was asked for a timeframe the strategy did not subscribe to. | depends on the call | `raptorbt-subscriptions` |
| `paper_unsolicited_position_change` | `StateError` | The simulated venue changed a position with no order behind it. | depends on the call | `raptorbt-live-trading` |
| `portfolio_total_unavailable` | `UnsupportedOperation` | A ctx.portfolio total needs the shared runtime (run_strategies). | depends on the call | `raptorbt-portfolio` |
| `primary_bars_without_ticks` | `InvalidSubscription` | primary_bars was given but there are no ticks to aggregate. | depends on the call | `raptorbt-subscriptions` |
| `run_finished` | `StateError` | The strategy's run is finished. | depends on the call | `raptorbt-backtesting` |
| `runtime_busy` | `StateError` | The runtime is in use; a call re-entered it from a hook. | depends on the call | `raptorbt-backtesting` |
| `runtime_not_ready` | `StateError` | A call was made before the runtime was finalized (or after it was built). | depends on the call | `raptorbt-backtesting` |
| `session_finished` | `SessionFinished` | The session is finished; create a new one. | depends on the call | `raptorbt-backtesting` |
| `structure_contract_missing` | `InvalidParameter` | The chain lists no contract (or expiry) a structure's leg selects at the event time. | depends on the call | `raptorbt-instruments` |
| `structure_limit_breached` | `InvalidParameter` | The legs' prices make the net premium worse than net_limit. | depends on the call | `raptorbt-orders` |
| `structure_margin_exceeded` | `InvalidParameter` | The margin model needs more for the group than is available. | depends on the call | `raptorbt-risk` |
| `structure_missing_mark` | `StateError` | A price or mark a structure needs was not supplied. | depends on the call | `raptorbt-instruments` |
| `unknown_account` | `InvalidParameter` | A strategy or call names an account that is not in accounts. | depends on the call | `raptorbt-portfolio` |
| `unknown_client_order_id` | `InvalidParameter` | A report names an order that was never sent. | depends on the call | `raptorbt-live-trading` |
| `unknown_symbol` | `UnknownSymbol` | A symbol was named that has no data in this run. | depends on the call | `raptorbt-market-data` |
| `unsupported_in_venue_run` | `UnsupportedOperation` | The request has no equivalent when orders go to an execution adapter. | depends on the call | `raptorbt-live-trading` |
| `unsupported_short_ticks` | `UnsupportedOperation` | Tick backtests are long-only; quantity must be >= 0. | depends on the call | `raptorbt-backtesting` |
| `untradable_instrument` | `UnsupportedOperation` | The instrument kind cannot be traded. | depends on the call | `raptorbt-instruments` |
| `venue_fill_not_booked` | `StateError` | The venue filled an order but the book could not take the fill: reconcile the position. | depends on the call | `raptorbt-live-trading` |
| `venue_requires_units` | `InvalidOrder` | An order sent to an execution adapter gave its size as a capital fraction or the whole position. | depends on the call | `raptorbt-live-trading` |
| `venue_unavailable` | `StateError` | A request may not have reached the venue; the order stays as it was until the venue reports. | depends on the call | `raptorbt-live-trading` |
