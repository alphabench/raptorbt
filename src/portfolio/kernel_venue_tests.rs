//! A venue fill is a fact. These tests book reports into a venue-driven
//! kernel and check the book follows the venue exactly: the price and size as
//! reported, a fill never dropped for a limit or for arriving out of order,
//! and the same book whatever order the reports arrive in.

use super::*;
use crate::accounts::Money;
use crate::execution::orders::{ExecutionReport, OrderKind, OrderSide, TimeInForce};
use crate::instruments::{InstrumentKind, InstrumentSpec};
use crate::portfolio::kernel_venue::{VenueError, MAX_UNBOOKED_FILLS};
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
fn a_second_entry_on_the_side_held_is_sent_and_adds_to_the_net_position() {
    let mut k = kernel();
    open_long(&mut k, 10.0, 100.0);
    assert_eq!(k.venue_pretrade(None, OrderSide::Buy, 5.0, false, Some(100.0), None), Ok(()));
    assert_eq!(
        k.venue_pretrade(None, OrderSide::Buy, 5.0, true, Some(100.0), None),
        Err(RejectReason::ReduceOnly)
    );
    assert_eq!(k.venue_pretrade(None, OrderSide::Sell, 5.0, false, Some(100.0), None), Ok(()));

    let second = submit(&mut k, OrderSide::Buy, 5.0, false, "again");
    accept(&mut k, second);
    let events = fill(&mut k, second, "E2", 5.0, 101.0).unwrap();
    assert_eq!(k.position_snapshot().unwrap().size, 15.0);
    assert!(!events.iter().any(|e| matches!(e, EngineEvent::FillBreach { .. })), "{events:?}");
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
    assert!(b.find_all(0, "e").is_empty(), "an execution older than the window is no longer held");
    assert!(!b.find_all(BOOKING_WINDOW as u64 + 9, "e").is_empty());
    assert!(
        b.window_starts_after(3),
        "a correction older than the window is told apart from one never booked"
    );
    assert!(!b.window_starts_after(BOOKING_WINDOW as u64 + 9));
    let mut c = VenueBookings::default();
    c.push(booking(1, "a", true, 7));
    c.push(booking(1, "b", false, 7));
    let (at, _) = c.find_all(1, "a")[0];
    assert!(c.reduced_since(at, 7));
}

fn hedged() -> EngineKernel {
    kernel().with_position_policy(crate::portfolio::ledger::PositionPolicy::Independent)
}

fn bust(k: &mut EngineKernel, order: u64, exec: &str) -> Result<(), VenueError> {
    k.apply_venue_report(
        1,
        3,
        order,
        &ExecutionReport::Bust { exec_id: exec.to_string(), ts_ns: 3 },
    )
    .map(|_| ())
}

#[test]
fn a_bust_of_a_fill_that_reversed_a_position_is_refused_and_changes_nothing() {
    // The fill reduced the long and opened a short: two bookings of one execution. Taking back
    // only the short would leave the reduction booked, a profit from a fill that never happened.
    let mut k = kernel();
    open_long(&mut k, 10.0, 100.0);
    let sell = submit(&mut k, OrderSide::Sell, 15.0, false, "exit");
    accept(&mut k, sell);
    fill(&mut k, sell, "E2", 15.0, 110.0).unwrap();
    let cash = k.cash_money();
    let held = k.position_snapshots();

    let err = bust(&mut k, sell, "E2").unwrap_err();

    assert!(matches!(err, VenueError::NotReversible { .. }), "{err:?}");
    assert!(err.to_string().contains("more than one position change"), "{err}");
    assert_eq!(k.cash_money(), cash, "the book was not touched");
    assert_eq!(k.position_snapshots().len(), held.len());
    assert_eq!(k.order(sell).unwrap().filled_qty, 15.0, "nor was the order");
}

#[test]
fn under_hedging_an_opposite_order_opens_its_own_position_and_a_bust_takes_back_exactly_that_one() {
    let mut k = hedged();
    open_long(&mut k, 10.0, 100.0);
    let cash_long = k.cash_money();
    let sell = submit(&mut k, OrderSide::Sell, 15.0, false, "hedge");
    accept(&mut k, sell);
    fill(&mut k, sell, "E2", 15.0, 110.0).unwrap();
    let held = k.position_snapshots();
    assert_eq!(held.len(), 2, "a hedging account keeps the long and opens a short beside it");
    assert_eq!((held[0].direction, held[0].size), (Direction::Long, 10.0));
    assert_eq!((held[1].direction, held[1].size), (Direction::Short, 15.0));

    bust(&mut k, sell, "E2").unwrap();

    let after = k.position_snapshots();
    assert_eq!(after.len(), 1);
    assert_eq!((after[0].direction, after[0].size), (Direction::Long, 10.0));
    assert_eq!(k.cash_money(), cash_long, "the short's cash came back to the cent");
}

#[test]
fn an_order_that_names_a_position_reduces_that_position_only() {
    let mut k = hedged();
    let a = open_long(&mut k, 10.0, 100.0);
    let b = submit(&mut k, OrderSide::Buy, 4.0, false, "second");
    accept(&mut k, b);
    fill(&mut k, b, "E2", 4.0, 101.0).unwrap();
    let ids: Vec<u64> = k.position_snapshots().iter().map(|p| p.position_id).collect();
    assert_eq!(ids.len(), 2);
    let _ = a;

    // Closing the newer position by name does not touch the older one.
    let close = submit(&mut k, OrderSide::Sell, 4.0, true, "close-newer");
    k.set_order_position(close, Some(ids[1]));
    accept(&mut k, close);
    fill(&mut k, close, "E3", 4.0, 103.0).unwrap();

    let held = k.position_snapshots();
    assert_eq!(held.len(), 1);
    assert_eq!((held[0].position_id, held[0].size), (ids[0], 10.0));

    // A fill larger than the position it names is not a reversal: the excess is kept unbooked.
    let big = submit(&mut k, OrderSide::Sell, 12.0, true, "close-older");
    k.set_order_position(big, Some(ids[0]));
    accept(&mut k, big);
    fill(&mut k, big, "E4", 12.0, 104.0).unwrap();
    assert!(k.position_snapshots().is_empty());
    assert_eq!(k.unbooked_fills().len(), 1);
    assert!((k.unbooked_fills()[0].qty - 2.0).abs() < 1e-9);
}

#[test]
fn an_order_naming_a_position_that_is_gone_is_kept_unbooked_not_opened_the_other_way() {
    let mut k = hedged();
    open_long(&mut k, 5.0, 100.0);
    let close = submit(&mut k, OrderSide::Sell, 5.0, true, "close");
    k.set_order_position(close, Some(9_999));
    accept(&mut k, close);
    fill(&mut k, close, "E2", 5.0, 101.0).unwrap();
    assert_eq!(k.position_snapshots().len(), 1, "the long is untouched and nothing was opened");
    assert_eq!(k.unbooked_fills().len(), 1);
}

#[test]
fn unbooked_fills_are_capped_with_a_structured_error() {
    let mut k = kernel();
    // Reduce-only fills with no position to reduce cannot book; each is kept, up to the cap.
    for n in 0..MAX_UNBOOKED_FILLS {
        let id = submit(&mut k, OrderSide::Sell, 1.0, true, &format!("r{n}"));
        accept(&mut k, id);
        fill(&mut k, id, "E", 1.0, 100.0).unwrap();
    }
    assert_eq!(k.unbooked_fills().len(), MAX_UNBOOKED_FILLS);
    let id = submit(&mut k, OrderSide::Sell, 1.0, true, "one-too-many");
    accept(&mut k, id);
    let err = fill(&mut k, id, "E", 1.0, 100.0).unwrap_err();
    assert!(matches!(err, VenueError::FillNotBooked { .. }), "{err:?}");
    assert_eq!(k.unbooked_fills().len(), MAX_UNBOOKED_FILLS, "it did not grow past the cap");
}

#[test]
fn a_correction_refused_at_the_unbooked_cap_leaves_the_order_position_and_cash_as_they_were() {
    let mut k = kernel();
    let entry = open_long(&mut k, 10.0, 100.0);
    // Reduce-only fills that name a position that is gone cannot book; each is kept, to the cap.
    for n in 0..MAX_UNBOOKED_FILLS {
        let id = submit(&mut k, OrderSide::Sell, 1.0, true, &format!("r{n}"));
        k.set_order_position(id, Some(9_999));
        accept(&mut k, id);
        fill(&mut k, id, "E", 1.0, 100.0).unwrap();
    }
    assert_eq!(k.unbooked_fills().len(), MAX_UNBOOKED_FILLS);
    let cash = k.cash();
    let held = format!("{:?}", k.position_snapshots());
    let counted = k.order(entry).unwrap().exec_effective("E1");

    let correction =
        ExecutionReport::Correction { exec_id: "E1".to_string(), qty: 8.0, price: 101.0, ts_ns: 3 };
    let err = k.apply_venue_report(2, 3, entry, &correction).unwrap_err();

    assert!(matches!(err, VenueError::FillNotBooked { .. }), "{err:?}");
    assert_eq!(k.cash(), cash, "the cash is as it was");
    assert_eq!(format!("{:?}", k.position_snapshots()), held, "the position is as it was");
    assert_eq!(
        k.order(entry).unwrap().exec_effective("E1"),
        counted,
        "the order still counts the original fill"
    );
    assert_eq!(k.unbooked_fills().len(), MAX_UNBOOKED_FILLS);
}

#[test]
fn an_unpriced_market_order_is_still_held_to_the_position_and_size_limits() {
    let config = BacktestConfig {
        fees: 0.0,
        max_position_qty: Some(10.0),
        max_order_qty: Some(500.0),
        ..BacktestConfig::default()
    };
    let k = kernel_with(config, None);
    assert_eq!(
        k.venue_pretrade(None, OrderSide::Buy, 1000.0, false, None, None),
        Err(RejectReason::OrderSize)
    );
    assert_eq!(
        k.venue_pretrade(None, OrderSide::Buy, 400.0, false, None, None),
        Err(RejectReason::MaxPosition)
    );
    assert_eq!(k.venue_pretrade(None, OrderSide::Buy, 10.0, false, None, None), Ok(()));
}

#[test]
fn a_working_limit_order_counts_at_its_own_price_with_no_mark() {
    let mut k = kernel();
    k.submit_venue_order(
        OrderSide::Buy,
        5.0,
        OrderKind::Limit { price: 100.0 },
        TimeInForce::Gtc,
        0,
        0,
        "rest".to_string(),
        false,
        false,
        None,
        None,
        None,
    );
    assert_eq!(k.working_notional(None), 500.0);
}
