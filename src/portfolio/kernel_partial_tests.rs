//! Partial fills: a typed order fills across prints, never past a print.
//!
//! Included from `kernel.rs` like `kernel_tests.rs`, so private kernel
//! items resolve.

use super::*;

fn partial_kernel() -> EngineKernel {
    let config = BacktestConfig { partial_fills: true, fees: 0.0, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    )
}

fn zero_fee_kernel() -> EngineKernel {
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    )
}

fn bar(idx: i64, price: Price) -> KernelBar {
    KernelBar {
        timestamp: idx,
        open: price,
        high: price + 1.0,
        low: price - 1.0,
        close: price,
        volume: 1000.0,
    }
}

fn print(ts: i64, price: Price, size: f64) -> TradeTick {
    TradeTick { timestamp: ts, price, size, signed_size: 0.0, oi: 0.0 }
}

fn submit(
    kernel: &mut EngineKernel,
    side: OrderSide,
    qty: QtySpec,
    tif: TimeInForce,
    tag: &str,
) -> u64 {
    kernel.submit_order(side, qty, OrderKind::Market, tif, 0, 0, tag.into(), None, None)
}

fn fills(events: &[EngineEvent]) -> Vec<f64> {
    events
        .iter()
        .filter_map(|e| match e {
            EngineEvent::OrderFilled { size, .. } => Some(*size),
            _ => None,
        })
        .collect()
}

#[test]
fn an_opening_order_fills_across_prints_into_one_averaged_position() {
    let mut kernel = partial_kernel();
    let id = submit(&mut kernel, OrderSide::Buy, QtySpec::Units(100.0), TimeInForce::Gtc, "o");
    let events = kernel.step_trade(1, &print(1, 100.0, 40.0), StepInput::default());
    assert_eq!(fills(&events), vec![40.0]);
    assert_eq!(kernel.order(id).unwrap().status, OrderStatus::PartiallyFilled);
    assert_eq!(kernel.order(id).unwrap().filled_qty, 40.0);
    // A second print of the SAME order is not "position_open": it adds.
    let events = kernel.step_trade(2, &print(2, 110.0, 60.0), StepInput::default());
    assert_eq!(fills(&events), vec![60.0]);
    assert!(!events.iter().any(|e| matches!(e, EngineEvent::OrderRejected { .. })), "{events:?}");
    assert_eq!(kernel.order(id).unwrap().status, OrderStatus::Filled);
    let snap = kernel.position_snapshot().unwrap();
    assert!((snap.size - 100.0).abs() < 1e-9);
    // Size-weighted average: (40*100 + 60*110) / 100 = 106.
    assert!((snap.entry_price - 106.0).abs() < 1e-9, "{}", snap.entry_price);
    // Cash paid the two slices at their own prices.
    assert!((kernel.cash() - (100_000.0 - 4_000.0 - 6_600.0)).abs() < 1e-6);
    // A different order is still a second entry under netting.
    let other = submit(&mut kernel, OrderSide::Buy, QtySpec::Units(10.0), TimeInForce::Gtc, "p");
    let events = kernel.step_trade(3, &print(3, 110.0, 10.0), StepInput::default());
    assert!(events.iter().any(|e| matches!(e, EngineEvent::OrderRejected { order_id, reason: "position_open", .. } if *order_id == other)), "{events:?}");
}

#[test]
fn a_derived_stop_follows_the_average_and_an_explicit_one_stays() {
    // Percent stop derived from the entry: shifts with the average.
    let config = BacktestConfig {
        partial_fills: true,
        fees: 0.0,
        stop: StopConfig::Fixed { percent: 0.10 },
        ..BacktestConfig::default()
    };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "T".into(),
        Direction::Long,
        None,
    );
    submit(&mut kernel, OrderSide::Buy, QtySpec::Units(100.0), TimeInForce::Gtc, "o");
    kernel.step_trade(1, &print(1, 100.0, 50.0), StepInput::default());
    let stop_after_first = kernel.position_snapshot().unwrap().stop_price.unwrap();
    assert!((stop_after_first - 90.0).abs() < 1e-9);
    kernel.step_trade(2, &print(2, 120.0, 50.0), StepInput::default());
    let stop = kernel.position_snapshot().unwrap().stop_price.unwrap();
    // Average moved 100 -> 110, so the stop moved 90 -> 100.
    assert!((stop - 100.0).abs() < 1e-9, "{stop}");

    // Explicit stop attached to the order: untouched by the average.
    let mut kernel = partial_kernel();
    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(100.0),
        OrderKind::Market,
        TimeInForce::Gtc,
        0,
        0,
        "e".into(),
        Some(80.0),
        None,
    );
    kernel.step_trade(1, &print(1, 100.0, 50.0), StepInput::default());
    kernel.step_trade(2, &print(2, 120.0, 50.0), StepInput::default());
    assert_eq!(kernel.position_snapshot().unwrap().stop_price, Some(80.0));
}

#[test]
fn a_closing_order_reduces_across_prints_and_never_reverses() {
    let mut kernel = partial_kernel();
    submit(&mut kernel, OrderSide::Buy, QtySpec::Units(100.0), TimeInForce::Gtc, "o");
    kernel.step_trade(1, &print(1, 100.0, 100.0), StepInput::default());
    assert!(kernel.is_in_position());
    // Sell 150 against a 100 position: 40 prints, then 60 closes it; the
    // surplus 50 is canceled rather than opening a short.
    let id = submit(&mut kernel, OrderSide::Sell, QtySpec::Units(150.0), TimeInForce::Gtc, "x");
    let events = kernel.step_trade(2, &print(2, 105.0, 40.0), StepInput::default());
    assert_eq!(fills(&events), vec![40.0]);
    assert!((kernel.position_snapshot().unwrap().size - 60.0).abs() < 1e-9);
    let trades: Vec<&Trade> = events
        .iter()
        .filter_map(|e| match e {
            EngineEvent::Exited { trade, .. } => Some(trade),
            _ => None,
        })
        .collect();
    assert_eq!(trades.len(), 1);
    assert!((trades[0].size - 40.0).abs() < 1e-9);
    assert!((trades[0].pnl - 200.0).abs() < 1e-9, "40 x (105-100) = 200, got {}", trades[0].pnl);
    let events = kernel.step_trade(3, &print(3, 106.0, 500.0), StepInput::default());
    assert_eq!(fills(&events), vec![60.0]);
    assert!(!kernel.is_in_position());
    assert_eq!(kernel.order(id).unwrap().status, OrderStatus::Filled);
    assert!((kernel.order(id).unwrap().filled_qty - 100.0).abs() < 1e-9);
    let events = kernel.step_trade(4, &print(4, 106.0, 500.0), StepInput::default());
    assert!(!kernel.is_in_position(), "the surplus must never reverse: {events:?}");
    // Cash: 100_000 - 10_000 + 40*105 + 60*106 = 100_560.
    assert!((kernel.cash() - 100_560.0).abs() < 1e-6, "{}", kernel.cash());
}

#[test]
fn ioc_keeps_one_slice_and_fok_takes_nothing_small() {
    let mut kernel = partial_kernel();
    let ioc = submit(&mut kernel, OrderSide::Buy, QtySpec::Units(100.0), TimeInForce::Ioc, "i");
    let events = kernel.step_trade(1, &print(1, 100.0, 30.0), StepInput::default());
    assert_eq!(fills(&events), vec![30.0]);
    assert!(events
        .iter()
        .any(|e| matches!(e, EngineEvent::OrderCanceled { order_id, .. } if *order_id == ioc)));
    assert_eq!(kernel.order(ioc).unwrap().status, OrderStatus::Canceled);
    assert!((kernel.order(ioc).unwrap().filled_qty - 30.0).abs() < 1e-9);
    assert!((kernel.position_snapshot().unwrap().size - 30.0).abs() < 1e-9);

    let mut kernel = partial_kernel();
    let fok = submit(&mut kernel, OrderSide::Buy, QtySpec::Units(100.0), TimeInForce::Fok, "f");
    let events = kernel.step_trade(1, &print(1, 100.0, 30.0), StepInput::default());
    assert!(fills(&events).is_empty(), "{events:?}");
    assert_eq!(kernel.order(fok).unwrap().status, OrderStatus::Canceled);
    assert!(!kernel.is_in_position());
    let mut kernel = partial_kernel();
    submit(&mut kernel, OrderSide::Buy, QtySpec::Units(100.0), TimeInForce::Fok, "g");
    let events = kernel.step_trade(1, &print(1, 100.0, 100.0), StepInput::default());
    assert_eq!(fills(&events), vec![100.0]);
}

#[test]
fn day_expiry_keeps_the_filled_part() {
    let mut kernel = partial_kernel();
    let day_ns = 86_400_000_000_000i64;
    let id = kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(100.0),
        OrderKind::Limit { price: 100.0 },
        TimeInForce::Day,
        0,
        0,
        "d".into(),
        None,
        None,
    );
    kernel.step_trade(1, &print(1, 100.0, 40.0), StepInput::default());
    assert_eq!(kernel.order(id).unwrap().status, OrderStatus::PartiallyFilled);
    let events = kernel.step_trade(2, &print(day_ns + 1, 100.0, 1.0), StepInput::default());
    assert!(events.iter().any(|e| matches!(e, EngineEvent::OrderExpired { .. })), "{events:?}");
    assert_eq!(kernel.order(id).unwrap().status, OrderStatus::Expired);
    assert!((kernel.position_snapshot().unwrap().size - 40.0).abs() < 1e-9);
}

#[test]
fn whole_fills_when_the_flag_is_off_and_on_bars() {
    let mut kernel = zero_fee_kernel();
    submit(&mut kernel, OrderSide::Buy, QtySpec::Units(100.0), TimeInForce::Gtc, "w");
    let events = kernel.step_trade(1, &print(1, 100.0, 5.0), StepInput::default());
    assert_eq!(fills(&events), vec![100.0], "flag off: whole");
    let mut kernel = partial_kernel();
    submit(&mut kernel, OrderSide::Buy, QtySpec::Units(100.0), TimeInForce::Gtc, "b");
    let events = kernel.step(0, &bar(0, 100.0), StepInput::default());
    assert_eq!(fills(&events), vec![100.0], "a bar fills whole even with the flag on");
}
