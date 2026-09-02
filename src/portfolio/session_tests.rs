//! Tests for the multi-instrument event session.
//!
//! Split out of `session.rs`, which the file-size rules cap; included back
//! into that module so `super::*` and private items still resolve.

use super::*;
use crate::instruments::{InstrumentKind, InstrumentSpec, OptionRight};

fn bars(start_ts: i64, closes: &[f64]) -> Vec<KernelBar> {
    closes
        .iter()
        .enumerate()
        .map(|(i, &c)| KernelBar {
            timestamp: start_ts + i as i64 * 10,
            open: c,
            high: c + 1.0,
            low: c - 1.0,
            close: c,
            volume: 1_000.0,
        })
        .collect()
}

fn session_two_instruments() -> EventSession {
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let mut session = EventSession::new(config);
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    let b = session.add_instrument("BBB".into(), Direction::Long, None, None, PositionPolicy::Net);
    // Interleaved timestamps: AAA at 0,10,20..., BBB offset by 5.
    session.set_bars(a, bars(0, &[100.0, 101.0, 102.0]));
    session.set_bars(b, bars(5, &[50.0, 51.0, 52.0]));
    session.seal();
    session
}

#[test]
fn schedule_interleaves_deterministically() {
    let mut session = session_two_instruments();
    let mut order = Vec::new();
    while let Some(entry) = session.current() {
        order.push((entry.instrument, entry.local_idx, entry.timestamp()));
        session.apply_current(StepInput::default());
    }
    assert_eq!(order, vec![(0, 0, 0), (1, 0, 5), (0, 1, 10), (1, 1, 15), (0, 2, 20), (1, 2, 25)]);
}

#[test]
fn shared_pool_constrains_second_instrument() {
    let mut session = session_two_instruments();
    // Enter AAA with everything on its first bar.
    session.apply_current(StepInput { entry: true, ..StepInput::default() });
    assert!(session.kernel(0).is_in_position());
    let cash_after_a = session.cash();
    assert!(cash_after_a < 1.0, "pool should be nearly spent, got {cash_after_a}");

    // BBB tries to enter with an empty pool: zero-size rejection.
    let events = session.apply_current(StepInput { entry: true, ..StepInput::default() });
    assert!(events.iter().any(|e| matches!(e, EngineEvent::EntryRejected { .. })));
    assert!(!session.kernel(1).is_in_position());
}

#[test]
fn equity_marks_both_instruments() {
    let mut session = session_two_instruments();
    // Enter AAA with half the pool.
    session.apply_current(StepInput { entry: true, size_mult: Some(0.5), ..StepInput::default() });
    // BBB enters with what remains.
    session.apply_current(StepInput { entry: true, ..StepInput::default() });
    assert!(session.kernel(0).is_in_position());
    assert!(session.kernel(1).is_in_position());

    // Run out the schedule; both drift up 2 points.
    while session.current().is_some() {
        session.apply_current(StepInput::default());
    }
    let equity = session.equity();
    assert!(equity > 100_000.0, "both positions gained, equity {equity}");

    let outcome = session.finish();
    assert_eq!(outcome.result.trades.len(), 2); // both force-closed at end
    assert_eq!(outcome.instruments.len(), 2);
    assert!(outcome.instruments.iter().all(|o| o.trades == 1));
    let total_pnl: f64 = outcome.instruments.iter().map(|o| o.pnl).sum();
    assert!(total_pnl > 0.0);
}

/// Risk-gated twin of [`session_two_instruments`].
fn session_two_instruments_gated(
    max_positions: Option<usize>,
    max_drawdown_pct: Option<f64>,
) -> EventSession {
    let config =
        BacktestConfig { fees: 0.0, max_positions, max_drawdown_pct, ..BacktestConfig::default() };
    let mut session = EventSession::new(config);
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    let b = session.add_instrument("BBB".into(), Direction::Long, None, None, PositionPolicy::Net);
    session.set_bars(a, bars(0, &[100.0, 101.0, 102.0]));
    session.set_bars(b, bars(5, &[50.0, 51.0, 52.0]));
    session.seal();
    session
}

#[test]
fn max_positions_caps_across_instruments() {
    // Regression: each kernel used to check only its own ledger, so a
    // limit of 1 allowed one position *per instrument*.
    let mut session = session_two_instruments_gated(Some(1), None);
    session.apply_current(StepInput { entry: true, size_mult: Some(0.25), ..StepInput::default() });
    assert!(session.kernel(0).is_in_position());

    let events = session.apply_current(StepInput {
        entry: true,
        size_mult: Some(0.25),
        ..StepInput::default()
    });
    assert!(
        events.iter().any(|e| matches!(
            e,
            EngineEvent::EntryRejected { reason: RejectReason::MaxPositions, .. }
        )),
        "the portfolio slot is taken, got {events:?}"
    );
    assert!(!session.kernel(1).is_in_position());

    let open = (0..2).filter(|&i| session.kernel(i).is_in_position()).count();
    assert_eq!(open, 1, "max_positions=1 must mean one position portfolio-wide");
}

#[test]
fn max_positions_frees_a_slot_on_exit() {
    let mut session = session_two_instruments_gated(Some(1), None);
    session.apply_current(StepInput { entry: true, size_mult: Some(0.25), ..StepInput::default() });
    // Close AAA, then let BBB take the freed slot.
    let position_id = session.kernel(0).position_snapshots()[0].position_id;
    session.kernel_mut(0).request_close(position_id);
    session.apply_current(StepInput { entry: false, ..StepInput::default() });
    while session.current().is_some() {
        let events = session.apply_current(StepInput {
            entry: true,
            size_mult: Some(0.25),
            ..StepInput::default()
        });
        if events.iter().any(|e| matches!(e, EngineEvent::Entered { .. })) {
            break;
        }
    }
    assert!(!session.kernel(0).is_in_position(), "AAA closed");
    assert!(session.kernel(1).is_in_position(), "BBB took the freed slot");
}

#[test]
fn max_positions_gates_the_order_path() {
    // Resting orders match *inside* `step()`, so a session-side
    // pre-check could not see them. Injecting the count covers this
    // path through the kernel's own gate.
    use crate::execution::orders::{OrderKind, OrderSide, QtySpec, TimeInForce};

    let mut session = session_two_instruments_gated(Some(1), None);
    session.apply_current(StepInput { entry: true, size_mult: Some(0.25), ..StepInput::default() });
    assert!(session.kernel(0).is_in_position(), "AAA holds the only slot");

    // BBB rests a marketable buy limit; it matches on its next bar.
    session.kernel_mut(1).submit_order(
        OrderSide::Buy,
        QtySpec::Units(10.0),
        OrderKind::Limit { price: 60.0 },
        TimeInForce::Gtc,
        0,
        5,
        "probe".to_string(),
        None,
        None,
    );
    let mut rejected = false;
    while session.current().is_some() {
        let events = session.apply_current(StepInput::default());
        if events.iter().any(|e| {
            matches!(e, EngineEvent::OrderRejected { reason, .. } if *reason == "max_positions")
        }) {
            rejected = true;
        }
    }
    assert!(rejected, "the resting order must be refused portfolio-wide");
    assert!(!session.kernel(1).is_in_position());
}

#[test]
fn unset_max_positions_is_unconstrained() {
    let mut session = session_two_instruments_gated(None, None);
    session.apply_current(StepInput { entry: true, size_mult: Some(0.25), ..StepInput::default() });
    let events = session.apply_current(StepInput {
        entry: true,
        size_mult: Some(0.25),
        ..StepInput::default()
    });
    assert!(!events.iter().any(|e| matches!(e, EngineEvent::EntryRejected { .. })));
    assert!(session.kernel(0).is_in_position());
    assert!(session.kernel(1).is_in_position());
}

#[test]
fn drawdown_halt_blocks_entries_on_all_instruments() {
    // AAA collapses; the portfolio drawdown gate halts BBB too, even
    // though BBB never traded.
    let config =
        BacktestConfig { fees: 0.0, max_drawdown_pct: Some(15.0), ..BacktestConfig::default() };
    let mut session = EventSession::new(config);
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    let b = session.add_instrument("BBB".into(), Direction::Long, None, None, PositionPolicy::Net);
    session.set_bars(a, bars(0, &[100.0, 50.0, 40.0, 40.0]));
    session.set_bars(b, bars(5, &[50.0, 50.0, 50.0, 50.0]));
    session.seal();

    // AAA goes all-in, then collapses.
    session.apply_current(StepInput { entry: true, ..StepInput::default() });
    let mut reasons = Vec::new();
    while session.current().is_some() {
        for event in session.apply_current(StepInput { entry: true, ..StepInput::default() }) {
            if let EngineEvent::EntryRejected { reason, .. } = event {
                reasons.push(reason);
            }
        }
    }
    assert!(session.is_halted(), "the drawdown gate must latch");
    assert!(
        reasons.contains(&RejectReason::DrawdownHalt),
        "expected a drawdown rejection, got {reasons:?}"
    );
}

#[test]
fn drawdown_halt_reports_its_own_reason_not_margin_call() {
    // A cash-account portfolio has no margin switch to trip; the halt
    // must not borrow the margin-call reason. BBB stays flat so its
    // entry attempts actually reach the gate.
    let config =
        BacktestConfig { fees: 0.0, max_drawdown_pct: Some(15.0), ..BacktestConfig::default() };
    let mut session = EventSession::new(config);
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    let b = session.add_instrument("BBB".into(), Direction::Long, None, None, PositionPolicy::Net);
    session.set_bars(a, bars(0, &[100.0, 50.0, 40.0, 40.0]));
    session.set_bars(b, bars(5, &[50.0, 50.0, 50.0, 50.0]));
    session.seal();

    session.apply_current(StepInput { entry: true, ..StepInput::default() });
    let mut reasons = Vec::new();
    while session.current().is_some() {
        for event in session.apply_current(StepInput { entry: true, ..StepInput::default() }) {
            if let EngineEvent::EntryRejected { reason, .. } = event {
                reasons.push(reason);
            }
        }
    }
    assert!(reasons.contains(&RejectReason::DrawdownHalt));
    assert!(
        !reasons.contains(&RejectReason::MarginCall),
        "a drawdown halt must not be labeled a margin call, got {reasons:?}"
    );
    assert!(!session.kernel(0).is_margin_halted());
}

#[test]
fn drawdown_halt_records_halted_at_on_the_account() {
    let config =
        BacktestConfig { fees: 0.0, max_drawdown_pct: Some(15.0), ..BacktestConfig::default() };
    let mut session = EventSession::new(config);
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    session.set_bars(a, bars(0, &[100.0, 50.0, 40.0, 40.0]));
    session.seal();

    session.apply_current(StepInput { entry: true, ..StepInput::default() });
    while session.current().is_some() {
        session.apply_current(StepInput::default());
    }
    let outcome = session.finish();
    assert!(outcome.halted);
    // The account is the single source of truth for *where*, whether the
    // halt came from a margin call or the drawdown gate.
    assert_eq!(outcome.halted_at, Some(1), "latched on the collapsing event");
}

#[test]
fn a_latched_drawdown_halt_suppresses_a_later_margin_call() {
    // Halts are latch-once: `halt_all` records the first cause, and the
    // maintenance check skips an already-halted account. Pins the
    // documented consequence of routing both causes through the account.
    let config =
        BacktestConfig { fees: 0.0, max_drawdown_pct: Some(15.0), ..BacktestConfig::default() };
    let mut session = EventSession::with_account(config, AccountMode::Margin { leverage: 50.0 });
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    session.set_bars(a, bars(0, &[100.0, 40.0, 30.0, 30.0]));
    session.seal();

    session.apply_current(StepInput { entry: true, ..StepInput::default() });
    let mut margin_calls = 0;
    while session.current().is_some() {
        margin_calls += session
            .apply_current(StepInput::default())
            .iter()
            .filter(|e| matches!(e, EngineEvent::MarginCall { .. }))
            .count();
    }
    let outcome = session.finish();
    assert!(outcome.halted);
    assert!(
        margin_calls <= 1,
        "a halted portfolio must not keep emitting margin calls, got {margin_calls}"
    );
}

/// Margin-mode twin of [`session_two_instruments`].
fn session_two_instruments_margin(leverage: f64, short_second: bool) -> EventSession {
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let mut session = EventSession::with_account(config, AccountMode::Margin { leverage });
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    let second_dir = if short_second { Direction::Short } else { Direction::Long };
    let b = session.add_instrument("BBB".into(), second_dir, None, None, PositionPolicy::Net);
    session.set_bars(a, bars(0, &[100.0, 101.0, 102.0]));
    session.set_bars(b, bars(5, &[50.0, 51.0, 52.0]));
    session.seal();
    session
}

#[test]
fn cash_mode_arithmetic_unchanged() {
    // Drift tripwire: the cash path must be bit-identical to the
    // single-pool implementation this replaced.
    let mut session = session_two_instruments();
    session.apply_current(StepInput { entry: true, size_mult: Some(0.5), ..StepInput::default() });
    session.apply_current(StepInput { entry: true, ..StepInput::default() });
    while session.current().is_some() {
        session.apply_current(StepInput::default());
    }
    // Exact equality, not approximate: these feed the golden metrics.
    assert_eq!(session.cash(), session.free_capital());
    let curve = session.equity_curve.clone();
    assert_eq!(curve.len(), 6);
    // Marked at full position value throughout, as the cash model does.
    assert_eq!(curve[0], 100_000.0);
    assert!(curve.iter().all(|v| v.is_finite()));
    // Both legs gain; the curve ends above where it started.
    assert!(curve[5] > curve[0], "curve {curve:?}");
}

#[test]
fn margin_pool_shared_across_kernels() {
    // The headline: under leverage the second instrument still has room,
    // where `shared_pool_constrains_second_instrument` shows it does not.
    // Size AAA at a quarter of capital. In cash mode that buys 250
    // units and leaves 75k; under 5x it buys 1250 units for the same
    // 25k of locked margin — and BBB still draws on the shared balance.
    let mut session = session_two_instruments_margin(5.0, false);
    session.apply_current(StepInput { entry: true, size_mult: Some(0.25), ..StepInput::default() });
    assert!(session.kernel(0).is_in_position());
    assert_eq!(session.kernel(0).position_snapshots()[0].size, 1_250.0);
    // Locks reserve capital without debiting cash.
    assert_eq!(session.cash(), 100_000.0);
    assert_eq!(session.free_capital(), 75_000.0);

    let events = session.apply_current(StepInput {
        entry: true,
        size_mult: Some(0.25),
        ..StepInput::default()
    });
    assert!(
        !events.iter().any(|e| matches!(e, EngineEvent::EntryRejected { .. })),
        "the shared balance should still fund the second instrument"
    );
    assert!(session.kernel(1).is_in_position());
    // BBB's sizing saw the portfolio's free capital, not the raw balance:
    // 25% of 75k at 50.0 under 5x margin.
    assert_eq!(session.kernel(1).position_snapshots()[0].size, 1_875.0);
    assert_eq!(session.free_capital(), 56_250.0);
}

#[test]
fn margin_mode_sizes_larger_than_cash() {
    let mut cash = session_two_instruments();
    cash.apply_current(StepInput { entry: true, ..StepInput::default() });
    let cash_size = cash.kernel(0).position_snapshots()[0].size;

    let mut margin = session_two_instruments_margin(5.0, false);
    margin.apply_current(StepInput { entry: true, ..StepInput::default() });
    let margin_size = margin.kernel(0).position_snapshots()[0].size;

    let ratio = margin_size / cash_size;
    assert!((ratio - 5.0).abs() < 0.01, "5x leverage should size ~5x, got {ratio}");
}

#[test]
fn margin_equity_is_direction_aware() {
    // AAA long and BBB short, both drifting up: the short loses, but
    // cash-mode marking would price the short's `position_value` as a
    // gain. Only direction-aware marking nets them correctly.
    let mut session = session_two_instruments_margin(2.0, true);
    session.apply_current(StepInput { entry: true, size_mult: Some(0.25), ..StepInput::default() });
    session.apply_current(StepInput { entry: true, size_mult: Some(0.25), ..StepInput::default() });
    assert!(session.kernel(0).is_in_position());
    assert!(session.kernel(1).is_in_position());

    // Run out the schedule so both legs are marked at their last close.
    while session.current().is_some() {
        session.apply_current(StepInput::default());
    }
    let short_pnl = session.kernel(1).unrealized_value(52.0);
    assert!(short_pnl < 0.0, "a short into a rising market must be a loss");
    // Cash-mode marking would add the short's *position value*, which
    // grows as the price rises — reporting a loss as a gain.
    let cash_style = session.cash()
        + session.kernel(0).position_value(102.0)
        + session.kernel(1).position_value(52.0);
    let equity = session.equity();
    assert!(
        equity < cash_style,
        "direction-aware marking must price the losing short below the \
         cash model: equity {equity} vs cash-style {cash_style}"
    );
}

#[test]
fn fully_funded_hedged_book_has_no_maintenance_requirement() {
    // A leverage-1.0 book locks the whole notional, so it cannot be
    // impaired and must never margin-call. Maintenance is charged on
    // *gross* notional, so without the fully-funded carve-out a hedged
    // long/short portfolio trips immediately — and the halt latches,
    // silently blocking every later entry.
    let mut session = session_two_instruments_margin(1.0, true);
    session.apply_current(StepInput { entry: true, size_mult: Some(0.5), ..StepInput::default() });
    session.apply_current(StepInput { entry: true, size_mult: Some(0.5), ..StepInput::default() });
    assert!(session.kernel(0).is_in_position());
    assert!(session.kernel(1).is_in_position());
    assert_eq!(session.kernel(0).maintenance_requirement(102.0), 0.0);
    assert_eq!(session.kernel(1).maintenance_requirement(52.0), 0.0);

    while session.current().is_some() {
        session.apply_current(StepInput::default());
    }
    assert!(!session.is_halted(), "a fully-funded hedged book must not margin-call");
}

#[test]
fn portfolio_margin_call_halts_all_kernels() {
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let mut session = EventSession::with_account(config, AccountMode::Margin { leverage: 50.0 });
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    let b = session.add_instrument("BBB".into(), Direction::Long, None, None, PositionPolicy::Net);
    // AAA collapses after entry; BBB is untouched and never enters.
    session.set_bars(a, bars(0, &[100.0, 40.0, 30.0]));
    session.set_bars(b, bars(5, &[50.0, 50.0, 50.0]));
    session.seal();

    session.apply_current(StepInput { entry: true, ..StepInput::default() });
    assert!(session.kernel(0).is_in_position());

    let mut calls = 0;
    while session.current().is_some() {
        let events = session.apply_current(StepInput::default());
        calls += events.iter().filter(|e| matches!(e, EngineEvent::MarginCall { .. })).count();
    }
    assert_eq!(calls, 1, "the call latches, so it fires exactly once");
    assert!(session.is_halted());

    // The untouched instrument is halted too — one shared account.
    assert!(session.kernel(1).is_margin_halted());
}

#[test]
fn maintenance_requirement_sums_per_instrument_rates() {
    // Each instrument contributes its own spec rate; a blended rate
    // would misprice the portfolio requirement.
    let mut session = session_two_instruments_margin(4.0, false);
    session.apply_current(StepInput { entry: true, size_mult: Some(0.25), ..StepInput::default() });
    session.apply_current(StepInput { entry: true, size_mult: Some(0.25), ..StepInput::default() });
    let required: f64 = [
        session.kernel(0).maintenance_requirement(102.0),
        session.kernel(1).maintenance_requirement(52.0),
    ]
    .iter()
    .sum();
    assert!(required > 0.0);
    // Default maint rate is half of init (1/4), i.e. 12.5% of notional.
    let notional = session.kernel(0).position_value(102.0).abs()
        + session.kernel(1).position_value(52.0).abs();
    assert!((required / notional - 0.125).abs() < 1e-9, "got {}", required / notional);
}

#[test]
fn finish_reports_rejected_entries_and_halt() {
    // A margin call halts both instruments; every later entry attempt is
    // a counted constraint refusal, on the untouched instrument too.
    // (Zero-size sizing is deliberately *not* counted — see the kernel.)
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let mut session = EventSession::with_account(config, AccountMode::Margin { leverage: 50.0 });
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    let b = session.add_instrument("BBB".into(), Direction::Long, None, None, PositionPolicy::Net);
    session.set_bars(a, bars(0, &[100.0, 40.0, 30.0, 30.0]));
    session.set_bars(b, bars(5, &[50.0, 50.0, 50.0, 50.0]));
    session.seal();

    // AAA enters, then collapses into a call; both then keep signaling.
    session.apply_current(StepInput { entry: true, ..StepInput::default() });
    while session.current().is_some() {
        session.apply_current(StepInput { entry: true, ..StepInput::default() });
    }
    assert!(session.is_halted());

    let outcome = session.finish();
    assert!(outcome.rejected_entries > 0, "rejections must be reported, not hardcoded to zero");
    let per_instrument: usize = outcome.instruments.iter().map(|o| o.rejected_entries).sum();
    assert_eq!(outcome.rejected_entries, per_instrument);
    // The instrument that never traded was halted by the shared account.
    assert!(outcome.instruments[1].rejected_entries > 0);
    assert!(outcome.halted);
    assert!(outcome.halted_at.is_some());
}

#[test]
fn finish_reports_no_halt_on_a_clean_run() {
    let mut session = session_two_instruments();
    session.apply_current(StepInput { entry: true, size_mult: Some(0.5), ..StepInput::default() });
    while session.current().is_some() {
        session.apply_current(StepInput::default());
    }
    let outcome = session.finish();
    assert!(!outcome.halted);
    assert_eq!(outcome.halted_at, None);
    assert_eq!(outcome.rejected_entries, 0);
}

#[test]
fn finish_reports_margin_halt() {
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let mut session = EventSession::with_account(config, AccountMode::Margin { leverage: 50.0 });
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    session.set_bars(a, bars(0, &[100.0, 40.0, 30.0]));
    session.seal();
    session.apply_current(StepInput { entry: true, ..StepInput::default() });
    while session.current().is_some() {
        session.apply_current(StepInput::default());
    }
    let outcome = session.finish();
    assert!(outcome.halted);
    assert!(outcome.halted_at.is_some());
}

#[test]
fn locked_margin_released_on_close() {
    let mut session = session_two_instruments_margin(5.0, false);
    session.apply_current(StepInput { entry: true, size_mult: Some(0.25), ..StepInput::default() });
    session.apply_current(StepInput { entry: true, size_mult: Some(0.25), ..StepInput::default() });
    assert!(session.free_capital() < session.cash());

    while session.current().is_some() {
        session.apply_current(StepInput::default());
    }
    let outcome = session.finish();
    assert_eq!(outcome.result.trades.len(), 2);
    // Every lock is released, so the closing balance is exactly the
    // starting capital plus realized PnL (fees are zero in this config).
    let total_pnl: f64 = outcome.instruments.iter().map(|o| o.pnl).sum();
    let final_balance =
        *outcome.result.equity_curve.last().expect("the schedule produced equity samples");
    assert!(
        (final_balance - (100_000.0 + total_pnl)).abs() < 1e-6,
        "balance {final_balance} should reconcile to 100000 + {total_pnl}"
    );
}

fn tick_data(rows: &[(i64, f64, f64, f64)]) -> TickData {
    // (timestamp, ltp, bid, ask); 0.0 means absent.
    TickData {
        timestamps: rows.iter().map(|r| r.0).collect(),
        ltp: rows.iter().map(|r| r.1).collect(),
        bid: rows.iter().map(|r| r.2).collect(),
        ask: rows.iter().map(|r| r.3).collect(),
        buy_qty_delta: rows.iter().map(|_| 1.0).collect(),
        sell_qty_delta: rows.iter().map(|_| 0.0).collect(),
        oi: rows.iter().map(|_| 0.0).collect(),
    }
}

#[test]
fn seal_keeps_trade_and_quote_events() {
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let mut session = EventSession::new(config);
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    session.set_ticks(a, tick_data(&[(10, 100.0, 99.0, 101.0), (20, 102.0, 0.0, 0.0)]));
    session.seal();

    let mut kinds = Vec::new();
    while let Some(entry) = session.current() {
        kinds.push(match entry.data {
            ScheduleData::Bar(_) => "bar",
            ScheduleData::Trade(_) => "trade",
            ScheduleData::Quote(_) => "quote",
            ScheduleData::Depth(_) => "book",
        });
        session.apply_current(StepInput::default());
    }
    // Row 1 yields a trade and a quote; row 2 only a trade (no book).
    assert_eq!(kinds, vec!["trade", "quote", "trade"]);
}

#[test]
fn trade_precedes_the_quote_of_the_same_feed_row() {
    // The print is what the book state at that row followed, so a strategy
    // reading the book inside a trade handler sees the *prior* quote — not
    // the one the print itself moved.
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let mut session = EventSession::new(config);
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    session.set_ticks(a, tick_data(&[(10, 100.0, 99.0, 101.0)]));
    session.seal();

    let first = session.current().expect("an event");
    assert!(matches!(first.data, ScheduleData::Trade(_)));
    session.apply_current(StepInput::default());
    let second = session.current().expect("a second event");
    assert!(matches!(second.data, ScheduleData::Quote(_)));
    assert_eq!(first.timestamp(), second.timestamp());
}

#[test]
fn quotes_do_not_sample_the_equity_curve() {
    // Metrics must not depend on how chatty the feed is.
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let with_quotes = {
        let mut s = EventSession::new(config.clone());
        let a = s.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
        s.set_ticks(a, tick_data(&[(10, 100.0, 99.0, 101.0), (20, 101.0, 100.0, 102.0)]));
        s.seal();
        while s.current().is_some() {
            s.apply_current(StepInput::default());
        }
        s.finish()
    };
    let without_quotes = {
        let mut s = EventSession::new(config);
        let a = s.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
        s.set_ticks(a, tick_data(&[(10, 100.0, 0.0, 0.0), (20, 101.0, 0.0, 0.0)]));
        s.seal();
        while s.current().is_some() {
            s.apply_current(StepInput::default());
        }
        s.finish()
    };
    assert_eq!(
        with_quotes.result.equity_curve.len(),
        without_quotes.result.equity_curve.len(),
        "quote events must not lengthen the equity curve"
    );
    assert_eq!(with_quotes.result.equity_curve, without_quotes.result.equity_curve);
}

#[test]
fn tick_entries_lend_and_drain_the_shared_account() {
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let mut session = EventSession::new(config);
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    session.set_ticks(a, tick_data(&[(10, 100.0, 0.0, 0.0), (20, 110.0, 0.0, 0.0)]));
    session.seal();

    session.apply_current(StepInput { entry: true, size_mult: Some(0.5), ..StepInput::default() });
    assert!(session.kernel(0).is_in_position());
    // Half the pool went into the position and the rest came back.
    assert!(session.cash() < 100_000.0 && session.cash() > 0.0);

    while session.current().is_some() {
        session.apply_current(StepInput::default());
    }
    let outcome = session.finish();
    assert_eq!(outcome.result.trades.len(), 1, "force-closed at the last print");
    assert!(outcome.instruments[0].pnl > 0.0, "the print rose from 100 to 110");
}

#[test]
fn max_positions_gates_tick_entries_portfolio_wide() {
    let config = BacktestConfig { fees: 0.0, max_positions: Some(1), ..BacktestConfig::default() };
    let mut session = EventSession::new(config);
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    let b = session.add_instrument("BBB".into(), Direction::Long, None, None, PositionPolicy::Net);
    session.set_ticks(a, tick_data(&[(10, 100.0, 0.0, 0.0), (30, 101.0, 0.0, 0.0)]));
    session.set_ticks(b, tick_data(&[(20, 50.0, 0.0, 0.0), (40, 51.0, 0.0, 0.0)]));
    session.seal();

    session.apply_current(StepInput { entry: true, size_mult: Some(0.25), ..StepInput::default() });
    let events = session.apply_current(StepInput {
        entry: true,
        size_mult: Some(0.25),
        ..StepInput::default()
    });
    assert!(events.iter().any(|e| matches!(
        e,
        EngineEvent::EntryRejected { reason: RejectReason::MaxPositions, .. }
    )));
    let open = (0..2).filter(|&i| session.kernel(i).is_in_position()).count();
    assert_eq!(open, 1);
}

fn depth_snapshot(ts: i64, bid: (f64, f64), ask: (f64, f64)) -> DepthTick {
    DepthTick::from_levels(
        ts,
        &[crate::data::BookLevel { price: bid.0, size: bid.1 }],
        &[crate::data::BookLevel { price: ask.0, size: ask.1 }],
    )
}

#[test]
fn depth_events_merge_and_reach_the_kernel_book() {
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let mut session = EventSession::new(config);
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    session.set_ticks(a, tick_data(&[(20, 100.0, 0.0, 0.0)]));
    session.set_depth(a, vec![depth_snapshot(10, (99.0, 500.0), (101.0, 400.0))]);
    session.seal();

    // The book precedes the print that follows it in time.
    let first = session.current().expect("an event");
    assert!(matches!(first.data, ScheduleData::Depth(_)));
    session.apply_current(StepInput::default());

    let book = &session.kernel(0).book;
    assert_eq!(book.best_bid(), Some(99.0));
    assert_eq!(book.size_at(crate::data::BookSide::Bid, 99.0), Some(500.0));
}

#[test]
fn depth_events_do_not_sample_the_equity_curve() {
    // Same reasoning as quotes, and depth feeds are chattier still.
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let curve_with_depth = {
        let mut s = EventSession::new(config.clone());
        let a = s.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
        s.set_ticks(a, tick_data(&[(20, 100.0, 0.0, 0.0), (40, 101.0, 0.0, 0.0)]));
        s.set_depth(
            a,
            vec![
                depth_snapshot(10, (99.0, 500.0), (101.0, 400.0)),
                depth_snapshot(30, (100.0, 500.0), (102.0, 400.0)),
            ],
        );
        s.seal();
        while s.current().is_some() {
            s.apply_current(StepInput::default());
        }
        s.finish().result.equity_curve
    };
    let curve_without = {
        let mut s = EventSession::new(config);
        let a = s.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
        s.set_ticks(a, tick_data(&[(20, 100.0, 0.0, 0.0), (40, 101.0, 0.0, 0.0)]));
        s.seal();
        while s.current().is_some() {
            s.apply_current(StepInput::default());
        }
        s.finish().result.equity_curve
    };
    assert_eq!(curve_with_depth, curve_without);
}

#[test]
fn depth_events_do_not_fill_resting_orders() {
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let mut session = EventSession::new(config);
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    session.set_ticks(a, tick_data(&[(30, 100.0, 0.0, 0.0)]));
    // A book straddling the limit is intent, not a trade.
    session.set_depth(a, vec![depth_snapshot(10, (89.0, 500.0), (91.0, 400.0))]);
    session.seal();

    session.kernel_mut(0).submit_order(
        crate::execution::orders::OrderSide::Buy,
        crate::execution::orders::QtySpec::Units(10.0),
        crate::execution::orders::OrderKind::Limit { price: 90.0 },
        crate::execution::orders::TimeInForce::Gtc,
        0,
        0,
        "d".to_string(),
        None,
        None,
    );
    while session.current().is_some() {
        session.apply_current(StepInput::default());
    }
    assert!(!session.kernel(0).is_in_position(), "a book must not fill an order");
}

#[test]
fn push_tick_matches_batch_replay() {
    // A live session pushing rows one at a time must land on the exact
    // numbers a batch replay of the same rows produces.
    let rows = [(10, 100.0, 99.0, 101.0), (20, 102.0, 0.0, 0.0), (30, 105.0, 104.0, 106.0)];
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };

    let batch = {
        let mut s = EventSession::new(config.clone());
        let a = s.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
        s.set_ticks(a, tick_data(&rows));
        s.seal();
        let mut first = true;
        while s.current().is_some() {
            let input = if first {
                first = false;
                StepInput { entry: true, size_mult: Some(0.5), ..StepInput::default() }
            } else {
                StepInput::default()
            };
            s.apply_current(input);
        }
        s.finish()
    };

    let streamed = {
        let mut s = EventSession::new(config);
        let a = s.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
        s.seal();
        let mut first = true;
        for (ts, ltp, bid, ask) in rows {
            // tick_data() stamps buy/sell deltas of 1.0/0.0 per row.
            s.push_tick(a, ts, ltp, bid, ask, 1.0, 0.0);
            while s.current().is_some() {
                let input = if first {
                    first = false;
                    StepInput { entry: true, size_mult: Some(0.5), ..StepInput::default() }
                } else {
                    StepInput::default()
                };
                s.apply_current(input);
            }
        }
        s.finish()
    };

    assert_eq!(batch.result.equity_curve, streamed.result.equity_curve);
    assert_eq!(batch.result.trades.len(), streamed.result.trades.len());
    assert_eq!(batch.instruments[0].pnl, streamed.instruments[0].pnl);
}

#[test]
fn pushes_append_behind_warmup_bars() {
    // Batch bars attached before the first push merge ahead of it, and the
    // per-instrument ordinal keeps counting across the seam — order matching
    // depends on local_idx staying monotone.
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let mut session = EventSession::new(config);
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    session.set_bars(a, bars(0, &[100.0, 101.0]));

    // No explicit seal: the first push seals implicitly.
    session.push_tick(a, 30, 102.0, 0.0, 0.0, 0.0, 0.0);

    let mut seen = Vec::new();
    while let Some(entry) = session.current() {
        let kind = match entry.data {
            ScheduleData::Bar(_) => "bar",
            ScheduleData::Trade(_) => "trade",
            _ => "other",
        };
        seen.push((kind, entry.local_idx, entry.timestamp()));
        session.apply_current(StepInput::default());
    }
    assert_eq!(seen, vec![("bar", 0, 0), ("bar", 1, 10), ("trade", 2, 30)]);
}

#[test]
fn remaining_counts_unapplied_events() {
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let mut session = EventSession::new(config);
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    session.seal();
    assert_eq!(session.remaining(), 0);

    // One row carrying both a print and a book appends two events.
    assert_eq!(session.push_tick(a, 10, 100.0, 99.0, 101.0, 0.0, 0.0), 2);
    assert_eq!(session.remaining(), 2);
    session.apply_current(StepInput::default());
    assert_eq!(session.remaining(), 1);
    session.apply_current(StepInput::default());
    assert_eq!(session.remaining(), 0);

    session.push_bar(
        a,
        KernelBar {
            timestamp: 20,
            open: 100.0,
            high: 101.0,
            low: 99.0,
            close: 100.5,
            volume: 10.0,
        },
    );
    assert_eq!(session.remaining(), 1);
}

#[test]
fn adoption_after_an_applied_event_is_refused() {
    // Adopting mid-run understates max drawdown and understates it in the
    // flattering direction: the equity curve is written streaming, so the
    // flat pre-adoption stretch holds the running peak below where it
    // belongs and the later decline measures against the wrong high-water
    // mark. On a 6-bar 100->95 fixture, a real 0.495% drawdown reports as
    // 0.199%. The samples are already wrong by the time metrics run, so the
    // ordering is refused rather than repaired.
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let mut session = EventSession::new(config);
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    session.set_bars(a, bars(0, &[100.0, 99.0, 98.0]));
    session.seal();
    session.apply_current(StepInput::default());

    let err = session.adopt_position(a, 0, 90.0, 100.0).unwrap_err();
    assert!(
        err.contains("before the first applied event"),
        "expected an ordering refusal, got: {err}"
    );
}

#[test]
fn adoption_survives_a_quote_only_event() {
    // The gate is the equity curve, not the event cursor. A quote advances
    // the cursor but samples no equity (marking on one would append a zero
    // return per quote and distort annualized metrics by how chatty the feed
    // is), so adopting after a quote corrupts nothing. Live feeds routinely
    // deliver quotes before the first print, and a broker holdings callback
    // can return after them — gating on the cursor would break that.
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let mut session = EventSession::new(config);
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    session.seal();
    // ltp = 0.0 means "no trade print", so this appends only the quote.
    assert_eq!(session.push_tick(a, 1_000, 0.0, 99.0, 101.0, 0.0, 0.0), 1);
    session.apply_current(StepInput::default());

    assert!(
        session.adopt_position(a, 0, 90.0, 100.0).is_ok(),
        "a quote samples no equity, so adoption after one must still be allowed"
    );
}

// ── Position-group margin: sold legs that hedge each other ─────────────────

fn option_spec(symbol: &str, strike: f64, right: OptionRight) -> InstrumentSpec {
    InstrumentSpec {
        lot_size: 30.0,
        span_pct: 0.0975,
        exposure_pct: 0.02,
        expiration_ns: Some(1_000_000),
        ..InstrumentSpec::new(
            symbol,
            InstrumentKind::Option {
                strike,
                right,
                underlying: Some("BANKNIFTY".to_string()),
                binary: false,
            },
        )
    }
}

/// A margin session at leverage 1.0 holding the legs given as
/// (symbol, strike, right, direction, premium series).
fn option_session(
    capital: f64,
    legs: &[(&str, f64, OptionRight, Direction, &[f64])],
) -> EventSession {
    let config =
        BacktestConfig { fees: 0.0, initial_capital: capital, ..BacktestConfig::default() };
    let mut session = EventSession::with_account(config, AccountMode::Margin { leverage: 1.0 });
    for (i, (symbol, strike, right, direction, premiums)) in legs.iter().enumerate() {
        let idx = session.add_instrument(
            symbol.to_string(),
            *direction,
            Some(option_spec(symbol, *strike, *right)),
            None,
            PositionPolicy::Net,
        );
        // Offset each leg's timestamps so the schedule interleaves legs in order.
        session.set_bars(idx, bars(i as i64, premiums));
    }
    session.seal();
    session
}

/// Enter one leg per fraction, in schedule order. Fractions are chosen so
/// each leg lands on exactly one lot: an unsized entry takes every lot the
/// pool can carry and starves the leg behind it.
fn enter_all(session: &mut EventSession, fractions: &[f64]) {
    for &fraction in fractions {
        let events = session.apply_current(StepInput {
            entry: true,
            size_mult: Some(fraction),
            ..StepInput::default()
        });
        assert!(
            events.iter().any(|e| matches!(e, EngineEvent::Entered { .. })),
            "expected an entry, got {events:?}"
        );
    }
}

#[test]
fn a_sold_straddle_locks_the_group_figure_not_two_naked_deposits() {
    let mut session = option_session(
        420_000.0,
        &[
            ("CE", 57_000.0, OptionRight::Call, Direction::Short, &[1_006.15, 1_006.15]),
            ("PE", 57_000.0, OptionRight::Put, Direction::Short, &[551.05, 551.05]),
        ],
    );
    // CE: 0.5 × 4,20,000 / 6,697.5 = 31 → one lot. PE: the pool has
    // 2,19,075 free and one naked lot needs 2,00,925 → 0.95 lands on one lot.
    enter_all(&mut session, &[0.5, 0.95]);
    assert!(session.kernel(0).is_in_position() && session.kernel(1).is_in_position());
    assert_eq!(session.kernel(0).open_size(), 30.0);
    assert_eq!(session.kernel(1).open_size(), 30.0);
    // One lot each. Naked: 2 × (0.1175 × 57,000 × 30) = 4,01,850. Group:
    // span once 1,66,725 + exposure 68,400 − premium 46,716 = 1,88,409.
    let locked = 420_000.0 - session.free_capital();
    assert!((locked - 188_409.0).abs() < 1.0, "locked {locked}");
    let per_kernel: f64 = (0..2).map(|i| session.kernel(i).locked_margin()).sum();
    assert!((per_kernel - locked).abs() < 1e-6, "kernels {per_kernel} vs account {locked}");
}

#[test]
fn a_new_sold_leg_must_still_be_carriable_on_its_own() {
    // 4,00,000 carries the first naked deposit (2,00,925) but not a second
    // before the group benefit exists: the second leg is refused for margin.
    let mut session = option_session(
        400_000.0,
        &[
            ("CE", 57_000.0, OptionRight::Call, Direction::Short, &[1_006.15]),
            ("PE", 57_000.0, OptionRight::Put, Direction::Short, &[551.05]),
        ],
    );
    enter_all(&mut session, &[0.51]);
    assert_eq!(session.kernel(0).open_size(), 30.0);
    let events = session.apply_current(StepInput {
        entry: true,
        size_mult: Some(1.0),
        ..StepInput::default()
    });
    assert!(
        events.iter().any(|e| matches!(
            e,
            EngineEvent::EntryRejected { reason: RejectReason::InsufficientMargin, .. }
        )),
        "{events:?}"
    );
}

#[test]
fn a_covered_spread_locks_a_fraction_of_the_naked_deposit() {
    let mut session = option_session(
        400_000.0,
        &[
            ("SHORT_PUT", 23_850.0, OptionRight::Put, Direction::Short, &[120.0, 120.0]),
            ("LONG_PUT", 23_800.0, OptionRight::Put, Direction::Long, &[102.05, 102.05]),
        ],
    );
    // Sold put: 0.25 × 4,00,000 / 2,802 = 35 → one lot. Bought put sizes at
    // its premium: 0.05 × 3,15,925 / 102.05 = 154 → five lots, covering it.
    enter_all(&mut session, &[0.25, 0.05]);
    let short_lots = session.kernel(0).open_size();
    let long_lots = session.kernel(1).open_size();
    assert_eq!(short_lots, 30.0);
    assert!(long_lots >= short_lots, "the wing must cover the sold leg, got {long_lots}");
    // Regrouped: the sold leg's lock is the width plus exposure plus the
    // net debit, far below its naked deposit (0.1175 × 23,850 × 30 = 84,075).
    let naked = 0.1175 * 23_850.0 * short_lots;
    let locked_short = session.kernel(0).locked_margin();
    assert!(locked_short < naked / 2.0, "sold leg locks {locked_short}, naked {naked}");
    // The bought wing still locks exactly its premium.
    let locked_long = session.kernel(1).locked_margin();
    assert!((locked_long - 102.05 * long_lots).abs() < 1e-6, "{locked_long}");
}

#[test]
fn a_hedged_pair_is_not_margin_called_on_a_move_two_naked_deposits_would_be() {
    let mut session = option_session(
        410_000.0,
        &[
            ("CE", 57_000.0, OptionRight::Call, Direction::Short, &[1_006.15, 5_000.0, 5_000.0]),
            ("PE", 57_000.0, OptionRight::Put, Direction::Short, &[551.05, 551.05, 551.05]),
        ],
    );
    // CE: 0.5 × 4,10,000 / 6,697.5 = 30 → one lot. PE: 2,09,075 free, one
    // naked lot needs 2,00,925 → 0.97 lands on one lot.
    enter_all(&mut session, &[0.5, 0.97]);
    // The call runs against the seller: loss (5,000 − 1,006.15) × 30 =
    // 1,19,815 → equity 2,90,185, under two naked deposits (4,01,850) but
    // over the group figure (1,88,409). No call.
    let mut calls = 0;
    while session.current().is_some() {
        calls += session
            .apply_current(StepInput::default())
            .iter()
            .filter(|e| matches!(e, EngineEvent::MarginCall { .. }))
            .count();
    }
    assert_eq!(calls, 0, "a hedged pair must be maintained at its group figure");
    assert!(!session.is_halted());
}
