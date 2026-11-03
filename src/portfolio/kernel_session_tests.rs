//! Session enforcement on the kernel: closed-market refusals, matching
//! outside the session, DAY expiry at the close and the square-off.

use crate::core::session_schedule::SessionSchedule;
use crate::core::types::{BacktestConfig, Direction, ExitReason, FillTiming};
use crate::execution::orders::{OrderKind, OrderSide, QtySpec, TimeInForce, TrailOffset};
use crate::execution::{FillPrice, SlippageModel};
use crate::instruments::{InstrumentKind, InstrumentSpec};
use crate::portfolio::kernel::{EngineEvent, EngineKernel, KernelBar, StepInput};
use crate::portfolio::risk::RejectReason;

const MIN: i64 = 60_000_000_000;
const HOUR: i64 = 60 * MIN;
const DAY: i64 = 24 * HOUR;
const IST: i64 = 5 * HOUR + 30 * MIN;

/// 2026-01-15 (a Thursday) 00:00 UTC as nanoseconds, plus whole days.
fn day_start(days_after: i64) -> i64 {
    20_468 * DAY + days_after * DAY
}

/// UTC nanoseconds of an IST wall-clock time, `days_after` days after Thu 15 Jan 2026.
fn ist(days_after: i64, hour: i64, minute: i64) -> i64 {
    day_start(days_after) + hour * HOUR + minute * MIN - IST
}

fn kernel(schedule: Option<SessionSchedule>, direction: Direction) -> EngineKernel {
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let mut spec = InstrumentSpec::new("X", InstrumentKind::Cash);
    spec.session = schedule;
    EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "X".to_string(),
        direction,
        None,
    )
    .with_instrument(spec)
}

fn bar(ts: i64, low: f64, close: f64) -> KernelBar {
    KernelBar { timestamp: ts, open: close, high: close + 1.0, low, close, volume: 1000.0 }
}

fn entry() -> StepInput {
    StepInput { entry: true, ..StepInput::default() }
}

fn rejection_of(events: &[EngineEvent]) -> Option<&'static str> {
    events.iter().find_map(|e| match e {
        EngineEvent::EntryRejected { reason, .. } => Some(reason.as_str()),
        EngineEvent::OrderRejected { reason, .. } => Some(*reason),
        _ => None,
    })
}

/// The NSE calendar with a 15:20 square-off, which most tests here are about
/// (the preset itself carries none).
fn nse() -> Option<SessionSchedule> {
    Some(SessionSchedule::nse_equity().with_squareoff_minute(Some(15 * 60 + 20)).unwrap())
}

/// The NSE calendar without a square-off, for tests that are not about it.
fn nse_no_squareoff() -> Option<SessionSchedule> {
    Some(SessionSchedule::nse_equity())
}

#[test]
fn an_entry_signal_while_closed_is_rejected_and_inside_it_trades() {
    let mut k = kernel(nse(), Direction::Long);
    let closed = k.step(0, &bar(ist(0, 8, 0), 99.0, 100.0), entry());
    assert_eq!(rejection_of(&closed), Some("market_closed"));
    assert!(!k.is_in_position());
    assert_eq!(RejectReason::MarketClosed.as_str(), "market_closed");

    let open = k.step(1, &bar(ist(0, 10, 0), 99.0, 100.0), entry());
    assert!(matches!(open.as_slice(), [EngineEvent::Entered { .. }]));
}

#[test]
fn a_holiday_and_a_weekend_are_closed_days() {
    let holiday = nse().map(|s| s.with_holidays([20260116]).unwrap());
    let mut k = kernel(holiday, Direction::Long);
    // Fri 16 Jan is the holiday; Sat 17 Jan is a weekend.
    for (i, day) in [1, 2].into_iter().enumerate() {
        let events = k.step(i, &bar(ist(day, 10, 0), 99.0, 100.0), entry());
        assert_eq!(rejection_of(&events), Some("market_closed"), "day {day}");
    }
}

#[test]
fn every_order_kind_is_refused_while_closed() {
    let trail = TrailOffset::Price(1.0);
    let kinds = [
        OrderKind::Market,
        OrderKind::Limit { price: 99.0 },
        OrderKind::StopMarket { trigger: 99.0 },
        OrderKind::StopLimit { trigger: 99.0, price: 98.0 },
        OrderKind::MarketIfTouched { trigger: 99.0 },
        OrderKind::LimitIfTouched { trigger: 99.0, price: 98.0 },
        OrderKind::MarketToLimit,
        OrderKind::TrailingStopMarket { offset: trail },
        OrderKind::TrailingStopLimit { offset: trail, limit_offset: 0.5 },
    ];
    for kind in kinds {
        let mut k = kernel(nse(), Direction::Long);
        let id = k.submit_order(
            OrderSide::Buy,
            QtySpec::Units(1.0),
            kind,
            TimeInForce::Gtc,
            0,
            ist(0, 16, 0),
            "c".into(),
            None,
            None,
        );
        let events = k.step(0, &bar(ist(0, 16, 1), 90.0, 100.0), StepInput::default());
        assert_eq!(rejection_of(&events), Some("market_closed"), "{kind:?}");
        assert!(!k.is_in_position(), "{kind:?}");
        assert!(k.order(id).unwrap().status.is_terminal(), "{kind:?}");
    }
}

#[test]
fn an_order_inside_the_session_is_accepted() {
    let mut k = kernel(nse(), Direction::Long);
    k.submit_order(
        OrderSide::Buy,
        QtySpec::Units(1.0),
        OrderKind::Limit { price: 95.0 },
        TimeInForce::Gtc,
        0,
        ist(0, 10, 0),
        "c".into(),
        None,
        None,
    );
    let events = k.step(0, &bar(ist(0, 10, 1), 99.0, 100.0), StepInput::default());
    assert!(events.iter().any(|e| matches!(e, EngineEvent::OrderAccepted { .. })));
    assert_eq!(rejection_of(&events), None);
}

#[test]
fn a_resting_order_is_skipped_outside_the_session_and_waits() {
    let mut k = kernel(nse_no_squareoff(), Direction::Long);
    k.submit_order(
        OrderSide::Buy,
        QtySpec::Units(1.0),
        OrderKind::Limit { price: 95.0 },
        TimeInForce::Gtc,
        0,
        ist(0, 10, 0),
        "c".into(),
        None,
        None,
    );
    k.step(0, &bar(ist(0, 10, 1), 99.0, 100.0), StepInput::default());
    // A bar after the close that trades through the limit does not fill it.
    let after_close = k.step(1, &bar(ist(0, 16, 0), 90.0, 92.0), StepInput::default());
    assert!(!after_close.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })));
    assert!(!k.is_in_position());
    // Next morning's bar does.
    let next_day = k.step(2, &bar(ist(1, 10, 0), 90.0, 92.0), StepInput::default());
    assert!(next_day.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })));
    assert!(k.is_in_position());
}

#[test]
fn the_bar_closing_at_the_session_close_is_still_in_session() {
    // No square-off here, which would otherwise own this event.
    let no_squareoff = nse().map(|s| s.with_squareoff_minute(None).unwrap());
    let mut k = kernel(no_squareoff, Direction::Long);
    k.submit_order(
        OrderSide::Buy,
        QtySpec::Units(1.0),
        OrderKind::Limit { price: 95.0 },
        TimeInForce::Gtc,
        0,
        ist(0, 10, 0),
        "c".into(),
        None,
        None,
    );
    k.step(0, &bar(ist(0, 10, 1), 99.0, 100.0), StepInput::default());
    // 15:29-15:30 bar, stamped 15:30: its data is in session.
    let last = k.step(1, &bar(ist(0, 15, 30), 90.0, 92.0), StepInput::default());
    assert!(last.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })));
}

#[test]
fn a_day_order_lapses_at_the_close_not_at_midnight() {
    let place = |k: &mut EngineKernel| {
        k.submit_order(
            OrderSide::Buy,
            QtySpec::Units(1.0),
            OrderKind::Limit { price: 50.0 },
            TimeInForce::Day,
            0,
            ist(0, 10, 0),
            "d".into(),
            None,
            None,
        );
        k.step(0, &bar(ist(0, 10, 1), 99.0, 100.0), StepInput::default());
    };
    let expired = |events: &[EngineEvent]| {
        events.iter().any(|e| matches!(e, EngineEvent::OrderExpired { .. }))
    };

    let mut with_session = kernel(nse_no_squareoff(), Direction::Long);
    place(&mut with_session);
    assert!(!expired(&with_session.step(
        1,
        &bar(ist(0, 15, 30), 99.0, 100.0),
        StepInput::default()
    )));
    assert!(expired(&with_session.step(
        2,
        &bar(ist(0, 15, 45), 99.0, 100.0),
        StepInput::default()
    )));

    // Without a calendar the same order lives to the local date rollover.
    let mut without = kernel(None, Direction::Long);
    place(&mut without);
    assert!(!expired(&without.step(1, &bar(ist(0, 15, 45), 99.0, 100.0), StepInput::default())));
}

fn squareoff_run(direction: Direction) -> Vec<(usize, ExitReason)> {
    let mut k = kernel(nse(), direction);
    k.step(0, &bar(ist(0, 10, 0), 99.0, 100.0), entry());
    let mut exits = Vec::new();
    let stamps = [
        ist(0, 15, 19),
        ist(0, 15, 20),
        ist(0, 15, 21),
        ist(0, 15, 25),
        ist(4, 10, 0),
        ist(4, 15, 20),
        ist(4, 15, 22),
    ];
    for (i, ts) in stamps.into_iter().enumerate() {
        let input = if ts == ist(4, 10, 0) { entry() } else { StepInput::default() };
        for event in k.step(i + 1, &bar(ts, 99.0, 100.0), input) {
            if let EngineEvent::Exited { trade, .. } = event {
                exits.push((i + 1, trade.exit_reason));
            }
        }
    }
    exits
}

#[test]
fn squareoff_closes_a_long_once_per_session() {
    // Day one's position closes on the 15:20 event only; day five re-enters
    // at 10:00 and is closed at its own 15:20.
    assert_eq!(
        squareoff_run(Direction::Long),
        [(2, ExitReason::Squareoff), (6, ExitReason::Squareoff)]
    );
}

#[test]
fn squareoff_closes_a_short_too() {
    assert_eq!(
        squareoff_run(Direction::Short),
        [(2, ExitReason::Squareoff), (6, ExitReason::Squareoff)]
    );
}

#[test]
fn entries_stay_refused_for_the_rest_of_the_squared_off_day() {
    let mut k = kernel(nse(), Direction::Long);
    let events = k.step(0, &bar(ist(0, 15, 20), 99.0, 100.0), entry());
    assert_eq!(rejection_of(&events), Some("squareoff"));
    for (i, (h, m)) in [(15, 21), (15, 29), (16, 0)].into_iter().enumerate() {
        let later = k.step(i + 1, &bar(ist(0, h, m), 99.0, 100.0), entry());
        assert_eq!(rejection_of(&later), Some("squareoff"), "{h}:{m}");
    }
    assert!(!k.is_in_position());
    // The next trading day is open for entries again, and squares off again.
    let next = k.step(9, &bar(ist(1, 10, 0), 99.0, 100.0), entry());
    assert!(matches!(next.as_slice(), [EngineEvent::Entered { .. }]));
}

#[test]
fn a_typed_entry_after_the_squareoff_is_refused_but_a_close_is_not() {
    let mut k = kernel(nse(), Direction::Long);
    k.step(0, &bar(ist(0, 10, 0), 99.0, 100.0), entry());
    k.step(1, &bar(ist(0, 15, 20), 99.0, 100.0), StepInput::default());
    assert!(!k.is_in_position(), "squared off");
    k.submit_order(
        OrderSide::Buy,
        QtySpec::Units(1.0),
        OrderKind::Market,
        TimeInForce::Gtc,
        2,
        ist(0, 15, 22),
        "late".into(),
        None,
        None,
    );
    let events = k.step(2, &bar(ist(0, 15, 22), 99.0, 100.0), StepInput::default());
    assert_eq!(rejection_of(&events), Some("squareoff"));
    assert!(!k.is_in_position());
}

#[test]
fn a_deferred_entry_filling_after_the_squareoff_is_refused() {
    let config = BacktestConfig {
        fees: 0.0,
        fill_timing: Some(FillTiming::NextBarOpen),
        ..BacktestConfig::default()
    };
    let fee_model = config.fee_model();
    let mut spec = InstrumentSpec::new("X", InstrumentKind::Cash);
    spec.session = nse();
    let mut k = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "X".to_string(),
        Direction::Long,
        None,
    )
    .with_instrument(spec);
    // Decided at 15:19, fills at the 15:20 open: the square-off event.
    k.step(0, &bar(ist(0, 15, 19), 99.0, 100.0), entry());
    let events = k.step(1, &bar(ist(0, 15, 20), 99.0, 100.0), StepInput::default());
    assert_eq!(rejection_of(&events), Some("squareoff"));
    assert!(!k.is_in_position());
}

#[test]
fn squareoff_cancels_working_entry_orders_and_keeps_reduce_only_ones() {
    let mut k = kernel(nse(), Direction::Long);
    let place = |k: &mut EngineKernel, price: f64, reduce_only: bool, tag: &str| {
        k.submit_order_full(
            if reduce_only { OrderSide::Sell } else { OrderSide::Buy },
            QtySpec::Units(1.0),
            OrderKind::Limit { price },
            TimeInForce::Gtc,
            0,
            ist(0, 10, 0),
            tag.into(),
            None,
            None,
            false,
            reduce_only,
            None,
        )
    };
    let entry_order = place(&mut k, 50.0, false, "entry");
    let exit_order = place(&mut k, 500.0, true, "exit");
    k.step(0, &bar(ist(0, 10, 1), 99.0, 100.0), StepInput::default());
    let events = k.step(1, &bar(ist(0, 15, 20), 99.0, 100.0), StepInput::default());
    assert!(events.iter().any(
        |e| matches!(e, EngineEvent::OrderCanceled { order_id, .. } if *order_id == entry_order)
    ));
    assert!(k.order(entry_order).unwrap().status.is_terminal());
    assert!(!k.order(exit_order).unwrap().status.is_terminal());
}

#[test]
fn squareoff_does_not_fire_on_a_holiday() {
    let holiday = nse().map(|s| s.with_holidays([20260115]).unwrap());
    let mut k = kernel(holiday, Direction::Long);
    k.step(0, &bar(ist(0, 15, 25), 99.0, 100.0), StepInput::default());
    assert!(!k.session_state.squareoff_now);
}

#[test]
fn an_instrument_without_a_schedule_steps_exactly_as_before() {
    let stamps = [ist(0, 3, 0), ist(0, 10, 0), ist(0, 15, 20), ist(0, 23, 0), ist(2, 12, 0)];
    let run = |spec_less: bool| -> String {
        let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
        let fee_model = config.fee_model();
        let mut k = EngineKernel::new(
            config,
            fee_model,
            SlippageModel::None,
            FillPrice::Close,
            "X".to_string(),
            Direction::Long,
            None,
        );
        if !spec_less {
            k = k.with_instrument(InstrumentSpec::new("X", InstrumentKind::Cash));
        }
        let mut log = String::new();
        for (i, ts) in stamps.into_iter().enumerate() {
            let input = StepInput { entry: i % 2 == 0, exit: i % 2 == 1, ..StepInput::default() };
            log.push_str(&format!("{:?}\n", k.step(i, &bar(ts, 99.0, 100.0 + i as f64), input)));
        }
        log
    };
    assert_eq!(run(true), run(false));
}

#[test]
fn a_configured_squareoff_time_closes_once_a_day_and_blocks_reentry_without_a_calendar() {
    let config = BacktestConfig {
        fees: 0.0,
        squareoff_time_minutes: Some(15 * 60 + 25),
        session_tz_offset_ns: IST,
        ..BacktestConfig::default()
    };
    let fee_model = config.fee_model();
    let mut k = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "X".to_string(),
        Direction::Long,
        None,
    );
    k.step(0, &bar(ist(0, 10, 0), 99.0, 100.0), entry());
    assert!(k.is_in_position());
    // Before the cutoff nothing happens; the first event at or after it closes the position.
    assert!(k.step(1, &bar(ist(0, 15, 24), 99.0, 100.0), StepInput::default()).is_empty());
    let events = k.step(2, &bar(ist(0, 15, 25), 99.0, 100.0), StepInput::default());
    assert!(
        matches!(events.as_slice(), [EngineEvent::Exited { trade, .. }] if trade.exit_reason == ExitReason::Squareoff),
        "{events:?}"
    );
    // The rest of that day refuses entries; the next day trades again.
    let blocked = k.step(3, &bar(ist(0, 15, 28), 99.0, 100.0), entry());
    assert_eq!(rejection_of(&blocked), Some("squareoff"));
    let next_day = k.step(4, &bar(ist(1, 10, 0), 99.0, 100.0), entry());
    assert!(matches!(next_day.as_slice(), [EngineEvent::Entered { .. }]));
}
