//! A venue fill is a fact. These tests book reports into a venue-driven
//! kernel and check the book follows the venue exactly: the price and size as
//! reported, a fill never dropped for a limit or for arriving out of order,
//! and the same book whatever order the reports arrive in.

use super::*;
use crate::accounts::Money;
use crate::execution::orders::{ExecutionReport, OrderKind, OrderSide, TimeInForce};
use crate::instruments::{InstrumentKind, InstrumentSpec};
use crate::portfolio::kernel_venue::VenueError;
use crate::portfolio::risk::RejectReason;

fn kernel_with(config: BacktestConfig, spec: Option<InstrumentSpec>) -> EngineKernel {
    let fee_model = config.fee_model();
    let kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "X".to_string(),
        Direction::Long,
        None,
    );
    match spec {
        Some(spec) => kernel.with_instrument(spec),
        None => kernel,
    }
}

fn kernel() -> EngineKernel {
    kernel_with(BacktestConfig { fees: 0.0, ..BacktestConfig::default() }, None)
}

fn submit(k: &mut EngineKernel, side: OrderSide, units: f64, reduce_only: bool, id: &str) -> u64 {
    k.submit_venue_order(
        side,
        units,
        OrderKind::Market,
        TimeInForce::Gtc,
        0,
        0,
        id.to_string(),
        false,
        reduce_only,
        None,
        None,
        None,
    )
}

fn accept(k: &mut EngineKernel, order: u64) {
    let accepted = ExecutionReport::Accepted { venue_order_id: format!("V{order}"), ts_ns: 1 };
    k.apply_venue_report(0, 1, order, &accepted).unwrap();
}

fn fill_report(exec: &str, qty: f64, price: f64) -> ExecutionReport {
    ExecutionReport::Fill { exec_id: exec.to_string(), qty, price, fee: 0.0, ts_ns: 2 }
}

fn fill(
    k: &mut EngineKernel,
    order: u64,
    exec: &str,
    qty: f64,
    price: f64,
) -> Result<Vec<EngineEvent>, VenueError> {
    k.apply_venue_report(1, 2, order, &fill_report(exec, qty, price)).map(|(_, events)| events)
}

fn filled_sizes(events: &[EngineEvent]) -> f64 {
    events
        .iter()
        .map(|e| match e {
            EngineEvent::OrderFilled { size, .. } => *size,
            _ => 0.0,
        })
        .sum()
}

fn open_long(k: &mut EngineKernel, units: f64, price: f64) -> u64 {
    let id = submit(k, OrderSide::Buy, units, false, "enter");
    accept(k, id);
    fill(k, id, "E1", units, price).unwrap();
    id
}

#[test]
fn a_closing_fill_larger_than_the_position_reverses_it_and_books_every_unit() {
    let mut k = kernel();
    open_long(&mut k, 10.0, 100.0);
    let sell = submit(&mut k, OrderSide::Sell, 15.0, false, "exit");
    accept(&mut k, sell);

    let events = fill(&mut k, sell, "E2", 15.0, 105.0).unwrap();

    assert_eq!(filled_sizes(&events), 15.0, "{events:?}");
    let pos = k.position_snapshot().expect("the remainder opened a short");
    assert_eq!((pos.direction, pos.size), (Direction::Short, 5.0));
    assert_eq!(k.order(sell).unwrap().filled_qty, 15.0);
    assert!(k.unbooked_fills().is_empty());
    let trade = events.iter().find_map(|e| match e {
        EngineEvent::Exited { trade, .. } => Some(trade),
        _ => None,
    });
    assert_eq!(trade.unwrap().size, 10.0);
}

#[test]
fn a_venue_price_is_booked_exactly_not_snapped_to_the_tick() {
    let spec = InstrumentSpec {
        price_increment: 0.05,
        ..InstrumentSpec::new("X", InstrumentKind::Contract { underlying: None })
    };
    let mut k = kernel_with(BacktestConfig { fees: 0.0, ..BacktestConfig::default() }, Some(spec));
    let buy = submit(&mut k, OrderSide::Buy, 10.0, false, "enter");
    accept(&mut k, buy);
    let events = fill(&mut k, buy, "E1", 10.0, 100.03).unwrap();
    let entered = events.iter().find_map(|e| match e {
        EngineEvent::Entered { price, .. } => Some(*price),
        _ => None,
    });
    assert_eq!(entered, Some(100.03), "{events:?}");
    assert_eq!(k.order(buy).unwrap().avg_fill_price(), Some(100.03));
    assert_eq!(k.position_snapshot().unwrap().entry_price, 100.03);

    let sell = submit(&mut k, OrderSide::Sell, 10.0, false, "exit");
    accept(&mut k, sell);
    let events = fill(&mut k, sell, "E2", 10.0, 100.07).unwrap();
    let exit = events.iter().find_map(|e| match e {
        EngineEvent::Exited { trade, .. } => Some(trade.exit_price),
        _ => None,
    });
    assert_eq!(exit, Some(100.07), "{events:?}");
}

#[test]
fn a_reduce_only_fill_before_its_entry_is_kept_and_booked_when_the_entry_arrives() {
    let in_order = {
        let mut k = kernel();
        let entry = submit(&mut k, OrderSide::Buy, 10.0, false, "enter");
        let stop = submit(&mut k, OrderSide::Sell, 10.0, true, "stop");
        accept(&mut k, entry);
        accept(&mut k, stop);
        fill(&mut k, entry, "E1", 10.0, 100.0).unwrap();
        fill(&mut k, stop, "S1", 10.0, 95.0).unwrap();
        k
    };

    let mut k = kernel();
    let entry = submit(&mut k, OrderSide::Buy, 10.0, false, "enter");
    let stop = submit(&mut k, OrderSide::Sell, 10.0, true, "stop");
    accept(&mut k, entry);
    accept(&mut k, stop);
    let early = fill(&mut k, stop, "S1", 10.0, 95.0).unwrap();
    assert!(
        early.iter().any(|e| matches!(e, EngineEvent::FillDeferred { size, .. } if *size == 10.0)),
        "{early:?}"
    );
    assert_eq!(k.unbooked_fills().len(), 1);
    assert!(!k.is_in_position());

    let late = fill(&mut k, entry, "E1", 10.0, 100.0).unwrap();
    assert!(late.iter().any(|e| matches!(e, EngineEvent::Exited { .. })), "{late:?}");
    assert!(k.unbooked_fills().is_empty());
    assert!(!k.is_in_position());
    assert_eq!(k.cash_money(), in_order.cash_money());
}

#[test]
fn a_second_entry_on_the_side_held_is_refused_before_sending_and_booked_if_the_venue_fills_it() {
    let mut k = kernel();
    open_long(&mut k, 10.0, 100.0);
    assert_eq!(
        k.venue_pretrade(OrderSide::Buy, 5.0, false, Some(100.0)),
        Err(RejectReason::PositionOpen)
    );
    assert_eq!(
        k.venue_pretrade(OrderSide::Buy, 5.0, true, Some(100.0)),
        Err(RejectReason::ReduceOnly)
    );
    assert_eq!(k.venue_pretrade(OrderSide::Sell, 5.0, false, Some(100.0)), Ok(()));

    // The venue filled it anyway: the position is what the venue says.
    let second = submit(&mut k, OrderSide::Buy, 5.0, false, "again");
    accept(&mut k, second);
    let events = fill(&mut k, second, "E2", 5.0, 101.0).unwrap();
    assert_eq!(k.position_snapshot().unwrap().size, 15.0);
    assert!(
        events.iter().any(
            |e| matches!(e, EngineEvent::FillBreach { reason, .. } if *reason == "position_open")
        ),
        "{events:?}"
    );
}

#[test]
fn a_fill_past_a_size_limit_is_booked_and_the_breach_reported() {
    let config =
        BacktestConfig { fees: 0.0, max_order_qty: Some(5.0), ..BacktestConfig::default() };
    let mut k = kernel_with(config, None);
    let buy = submit(&mut k, OrderSide::Buy, 10.0, false, "enter");
    accept(&mut k, buy);
    let events = fill(&mut k, buy, "E1", 10.0, 100.0).unwrap();
    assert_eq!(k.position_snapshot().unwrap().size, 10.0);
    assert!(
        events.iter().any(
            |e| matches!(e, EngineEvent::FillBreach { reason, .. } if *reason == "order_size")
        ),
        "{events:?}"
    );
}

#[test]
fn a_fill_the_account_cannot_afford_is_booked_and_the_breach_reported() {
    let mut k = kernel();
    let buy = submit(&mut k, OrderSide::Buy, 1.0e9, false, "enter");
    accept(&mut k, buy);
    let events = fill(&mut k, buy, "E1", 1.0e9, 100.0).unwrap();
    assert_eq!(k.position_snapshot().unwrap().size, 1.0e9);
    assert!(
        events.iter().any(|e| matches!(e, EngineEvent::FillBreach { reason, .. }
            if *reason == "insufficient_capital")),
        "{events:?}"
    );
}

#[test]
fn a_fill_is_booked_at_its_full_size_whatever_the_lot() {
    let spec = InstrumentSpec {
        lot_size: 25.0,
        ..InstrumentSpec::new("X", InstrumentKind::Contract { underlying: None })
    };
    let mut k = kernel_with(BacktestConfig { fees: 0.0, ..BacktestConfig::default() }, Some(spec));
    let buy = submit(&mut k, OrderSide::Buy, 30.0, false, "enter");
    accept(&mut k, buy);
    let events = fill(&mut k, buy, "E1", 30.0, 100.0).unwrap();
    assert_eq!(k.position_snapshot().unwrap().size, 30.0, "{events:?}");
    assert_eq!(filled_sizes(&events), 30.0);
}

#[test]
fn two_oco_legs_that_both_fill_leave_the_book_where_the_broker_is() {
    // Not reduce-only: the second leg sells into a flat book, so the broker
    // is short and the book follows.
    let mut k = kernel();
    open_long(&mut k, 10.0, 100.0);
    let stop = submit(&mut k, OrderSide::Sell, 10.0, false, "stop");
    let target = submit(&mut k, OrderSide::Sell, 10.0, false, "target");
    accept(&mut k, stop);
    accept(&mut k, target);
    fill(&mut k, stop, "S1", 10.0, 95.0).unwrap();
    fill(&mut k, target, "T1", 10.0, 110.0).unwrap();
    let pos = k.position_snapshot().unwrap();
    assert_eq!((pos.direction, pos.size), (Direction::Short, 10.0));

    // Reduce-only: the second fill cannot reduce anything. It is kept and
    // reported, never dropped.
    let mut k = kernel();
    open_long(&mut k, 10.0, 100.0);
    let stop = submit(&mut k, OrderSide::Sell, 10.0, true, "stop");
    let target = submit(&mut k, OrderSide::Sell, 10.0, true, "target");
    accept(&mut k, stop);
    accept(&mut k, target);
    fill(&mut k, stop, "S1", 10.0, 95.0).unwrap();
    let events = fill(&mut k, target, "T1", 10.0, 110.0).unwrap();
    assert!(events.iter().any(|e| matches!(e, EngineEvent::FillDeferred { .. })), "{events:?}");
    assert_eq!(k.unbooked_fills().len(), 1);
    assert_eq!(k.unbooked_fills()[0].qty, 10.0);
}

mod book_convergence {
    use super::*;
    use proptest::prelude::*;

    /// An entry of 10 in two executions and a reduce-only stop of 10 in two,
    /// with their acceptances.
    fn script() -> Vec<(bool, ExecutionReport)> {
        let rep = |exec: &str, qty: f64, price: f64| fill_report(exec, qty, price);
        let acc = |id: &str| ExecutionReport::Accepted { venue_order_id: id.into(), ts_ns: 1 };
        vec![
            (true, acc("VE")),
            (true, rep("E1", 4.0, 100.0)),
            (true, rep("E2", 6.0, 101.0)),
            (false, acc("VS")),
            (false, rep("S1", 7.0, 95.0)),
            (false, rep("S2", 3.0, 96.0)),
        ]
    }

    fn run(reports: &[(bool, ExecutionReport)]) -> EngineKernel {
        let mut k = kernel();
        let entry = submit(&mut k, OrderSide::Buy, 10.0, false, "enter");
        let stop = submit(&mut k, OrderSide::Sell, 10.0, true, "stop");
        for (is_entry, report) in reports {
            let order = if *is_entry { entry } else { stop };
            k.apply_venue_report(1, 2, order, report).unwrap();
        }
        k
    }

    fn signature(k: &EngineKernel) -> (Money, bool, usize, Vec<u64>) {
        (
            k.cash_money(),
            k.is_in_position(),
            k.unbooked_fills().len(),
            k.orders.all().iter().map(|o| o.filled_qty.to_bits()).collect(),
        )
    }

    proptest! {
        #[test]
        fn reports_in_any_order_and_repeated_leave_the_same_book(
            seed in any::<u64>(),
            dup in prop::collection::vec(any::<prop::sample::Index>(), 0..4),
        ) {
            let base = script();
            let in_order = run(&base);
            let mut noisy = base.clone();
            for pick in &dup {
                noisy.push(base[pick.index(base.len())].clone());
            }
            let mut x = seed | 1;
            for i in (1..noisy.len()).rev() {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                noisy.swap(i, (x >> 33) as usize % (i + 1));
            }
            let shuffled = run(&noisy);
            prop_assert_eq!(signature(&in_order), signature(&shuffled));
            prop_assert!(!shuffled.is_in_position());
            prop_assert!(shuffled.unbooked_fills().is_empty());
        }
    }
}

#[test]
fn the_bookings_kept_to_restate_executions_are_bounded_and_indexed() {
    use crate::portfolio::kernel_venue::{VenueBooking, VenueBookings, BOOKING_WINDOW};
    let booking = |order_id: u64, exec: &str, entry: bool, position_id: u64| VenueBooking {
        order_id,
        exec_id: exec.to_string(),
        qty: 1.0,
        price: 1.0,
        booked_qty: 1.0,
        booked_price: 1.0,
        entry,
        position_id,
        cash: Money::ZERO,
        locked: 0.0,
        cost: Money::ZERO,
        fees: Money::ZERO,
        shift: false,
    };
    let mut b = VenueBookings::default();
    for n in 0..(BOOKING_WINDOW as u64 + 10) {
        b.push(booking(n, "e", true, n));
    }
    assert_eq!(b.len(), BOOKING_WINDOW, "the oldest bookings are dropped past the window");
    assert!(b.find(0, "e").is_none(), "an execution older than the window is no longer held");
    assert!(b.find(BOOKING_WINDOW as u64 + 9, "e").is_some());
    assert!(
        b.window_starts_after(3),
        "a correction older than the window is told apart from one never booked"
    );
    assert!(!b.window_starts_after(BOOKING_WINDOW as u64 + 9));
    let mut c = VenueBookings::default();
    c.push(booking(1, "a", true, 7));
    c.push(booking(1, "b", false, 7));
    let (at, _) = c.find(1, "a").unwrap();
    assert!(c.reduced_since(at, 7));
}
