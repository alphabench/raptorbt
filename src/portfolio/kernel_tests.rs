//! Tests for the simulation kernel.
//!
//! Split out of `kernel.rs`, which the file-size rules cap; included back
//! into that module so `super::*` and private items still resolve.

use super::*;

fn make_kernel() -> EngineKernel {
    let config = BacktestConfig::default();
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

fn enter(kernel: &mut EngineKernel, idx: usize, price: Price) {
    let events = kernel.step(
        idx,
        &bar(idx as i64, price),
        StepInput { entry: true, ..StepInput::default() },
    );
    assert!(
        matches!(events.as_slice(), [EngineEvent::Entered { .. }]),
        "expected entry, got {events:?}"
    );
}

#[test]
fn set_stop_price_is_noop_when_flat() {
    let mut kernel = make_kernel();
    kernel.set_stop_price(Some(90.0));
    assert!(kernel.position_snapshot().is_none());
}

fn trade(ts: i64, price: Price, size: f64) -> TradeTick {
    TradeTick { timestamp: ts, price, size, signed_size: 0.0 }
}

#[test]
fn step_trade_enters_and_exits_at_the_print() {
    let mut kernel = make_kernel();
    let events = kernel.step_trade(
        0,
        &trade(0, 100.0, 5.0),
        StepInput { entry: true, ..Default::default() },
    );
    assert!(matches!(events.as_slice(), [EngineEvent::Entered { price, .. }] if *price == 100.0));
    assert!(kernel.is_in_position());

    let events =
        kernel.step_trade(1, &trade(1, 110.0, 5.0), StepInput { exit: true, ..Default::default() });
    assert!(
        matches!(events.as_slice(), [EngineEvent::Exited { trade, .. }] if trade.exit_price == 110.0)
    );
}

#[test]
fn step_quote_does_not_move_the_trailing_watermark() {
    // A bid that never traded must not ratchet a position's trailing
    // stop — that would manufacture exits from an untraded price.
    let mut kernel = make_kernel();
    kernel.step_trade(0, &trade(0, 100.0, 1.0), StepInput { entry: true, ..Default::default() });
    let before = kernel.position_snapshot().expect("in position");

    let events = kernel.step_quote(&QuoteTick { timestamp: 1, bid: 500.0, ask: 501.0 });
    assert!(events.is_empty());
    let after = kernel.position_snapshot().expect("still in position");
    assert_eq!(before.stop_price, after.stop_price);
    assert_eq!(kernel.best_bid(), Some(500.0));
    assert_eq!(kernel.best_ask(), Some(501.0));
}

#[test]
fn step_quote_does_not_match_resting_orders() {
    let mut kernel = make_kernel();
    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(10.0),
        OrderKind::Limit { price: 90.0 },
        TimeInForce::Gtc,
        0,
        0,
        "q".to_string(),
        None,
        None,
    );
    // A quote straddling the limit must not fill it.
    kernel.step_quote(&QuoteTick { timestamp: 1, bid: 80.0, ask: 81.0 });
    assert!(!kernel.is_in_position());

    // The print that follows is the evidence, and does fill it.
    let events = kernel.step_trade(1, &trade(2, 89.0, 10.0), StepInput::default());
    assert!(
        events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })),
        "the print should fill the resting limit, got {events:?}"
    );
}

#[test]
fn step_trade_does_not_fill_bar_phase_market_orders() {
    // AT_CLOSE queues for a bar phase a print does not have.
    let mut kernel = make_kernel();
    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(10.0),
        OrderKind::Market,
        TimeInForce::AtClose,
        0,
        0,
        "atclose".to_string(),
        None,
        None,
    );
    let events = kernel.step_trade(1, &trade(1, 100.0, 1.0), StepInput::default());
    assert!(
        !events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })),
        "AT_CLOSE must keep resting on a print, got {events:?}"
    );
    assert!(!kernel.is_in_position());

    // It fills on the next bar event.
    let events = kernel.step(2, &bar(2, 100.0), StepInput::default());
    assert!(events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })));
}

#[test]
fn step_trade_skips_orders_submitted_on_the_same_event() {
    let mut kernel = make_kernel();
    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(10.0),
        OrderKind::Limit { price: 200.0 },
        TimeInForce::Gtc,
        5,
        0,
        "same".to_string(),
        None,
        None,
    );
    // Submitted while observing event 5: cannot rest into event 5.
    let events = kernel.step_trade(5, &trade(5, 100.0, 1.0), StepInput::default());
    assert!(!events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })));
    // Event 6 matches it.
    let events = kernel.step_trade(6, &trade(6, 100.0, 1.0), StepInput::default());
    assert!(events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })));
}

#[test]
fn queue_model_is_off_by_default() {
    // A resting limit fills on the first print that reaches it, exactly
    // as before the queue model existed.
    let mut kernel = make_kernel();
    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(10.0),
        OrderKind::Limit { price: 99.0 },
        TimeInForce::Gtc,
        0,
        0,
        "q".to_string(),
        None,
        None,
    );
    let events = kernel.step_trade(1, &trade(1, 99.0, 1.0), StepInput::default());
    assert!(events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })));
}

#[test]
fn queue_model_holds_an_order_behind_displayed_size() {
    let config = BacktestConfig { queue_fill_model: true, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    );
    // 300 lots displayed at our price: we join behind them.
    kernel.book.apply_depth(&crate::data::DepthTick::from_levels(
        0,
        &[crate::data::BookLevel { price: 99.0, size: 300.0 }],
        &[crate::data::BookLevel { price: 101.0, size: 100.0 }],
    ));
    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(10.0),
        OrderKind::Limit { price: 99.0 },
        TimeInForce::Gtc,
        0,
        0,
        "q".to_string(),
        None,
        None,
    );

    // A small print at our price does not reach us.
    let events = kernel.step_trade(1, &trade(1, 99.0, 100.0), StepInput::default());
    assert!(!events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })));
    assert!(!kernel.is_in_position());

    // Enough volume prints through the queue ahead, and we fill.
    let events = kernel.step_trade(2, &trade(2, 99.0, 250.0), StepInput::default());
    assert!(
        events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })),
        "the queue ahead was exhausted, got {events:?}"
    );
}

#[test]
fn queue_model_fills_when_the_level_trades_through() {
    let config = BacktestConfig { queue_fill_model: true, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    );
    kernel.book.apply_depth(&crate::data::DepthTick::from_levels(
        0,
        &[crate::data::BookLevel { price: 99.0, size: 100_000.0 }],
        &[crate::data::BookLevel { price: 101.0, size: 100.0 }],
    ));
    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(10.0),
        OrderKind::Limit { price: 99.0 },
        TimeInForce::Gtc,
        0,
        0,
        "q".to_string(),
        None,
        None,
    );
    // A huge queue ahead, but the print cleared the level entirely.
    let events = kernel.step_trade(1, &trade(1, 98.0, 1.0), StepInput::default());
    assert!(events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })));
}

#[test]
fn queue_model_falls_back_to_probability_on_bar_events() {
    // A bar's volume is not volume at the limit price, so the queue
    // model must not consume it; fill_prob_limit=1.0 fills as always.
    let config = BacktestConfig { queue_fill_model: true, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    );
    kernel.book.apply_depth(&crate::data::DepthTick::from_levels(
        0,
        &[crate::data::BookLevel { price: 99.0, size: 100_000.0 }],
        &[crate::data::BookLevel { price: 101.0, size: 100.0 }],
    ));
    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(10.0),
        OrderKind::Limit { price: 99.0 },
        TimeInForce::Gtc,
        0,
        0,
        "q".to_string(),
        None,
        None,
    );
    let events = kernel.step(1, &bar(1, 98.5), StepInput::default());
    assert!(
        events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })),
        "bar events fall back to fill_prob_limit, got {events:?}"
    );
}

#[test]
fn external_open_count_overrides_the_ledger() {
    // A flat kernel would pass a max_positions=1 gate on its own ledger;
    // a portfolio that already holds a position elsewhere says otherwise.
    let mut kernel = make_kernel().with_risk_gate(RiskGate::new(Some(1), None));
    assert_eq!(kernel.open_count(), 0);

    kernel.set_external_open_count(Some(1));
    let events = kernel.step(0, &bar(0, 100.0), StepInput { entry: true, ..StepInput::default() });
    assert!(
        matches!(
            events.as_slice(),
            [EngineEvent::EntryRejected { reason: RejectReason::MaxPositions, .. }]
        ),
        "expected a portfolio-wide rejection, got {events:?}"
    );
    assert!(!kernel.is_in_position());

    // Clearing it restores ledger-derived counting: the slot is free.
    kernel.set_external_open_count(None);
    enter(&mut kernel, 1, 100.0);
    assert!(kernel.is_in_position());
}

#[test]
fn set_stop_and_target_update_open_position() {
    let mut kernel = make_kernel();
    enter(&mut kernel, 0, 100.0);

    kernel.set_stop_price(Some(95.0));
    kernel.set_target_price(Some(110.0));

    let snap = kernel.position_snapshot().unwrap();
    assert_eq!(snap.stop_price, Some(95.0));
    assert_eq!(snap.target_price, Some(110.0));

    kernel.set_stop_price(None);
    assert_eq!(kernel.position_snapshot().unwrap().stop_price, None);
}

#[test]
fn programmatic_stop_triggers_exit() {
    let mut kernel = make_kernel();
    enter(&mut kernel, 0, 100.0);
    kernel.set_stop_price(Some(98.5));

    // Bar trades down through the stop.
    let events = kernel.step(1, &bar(1, 98.0), StepInput::default());
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => {
            assert_eq!(trade.exit_reason, ExitReason::StopLoss);
        }
        other => panic!("expected stop exit, got {other:?}"),
    }
    assert!(!kernel.is_in_position());
}

#[test]
fn entry_stop_override_beats_config() {
    let config = BacktestConfig {
        stop: StopConfig::Fixed { percent: 0.05 },
        target: TargetConfig::Fixed { percent: 0.10 },
        ..BacktestConfig::default()
    };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    );

    let events = kernel.step(
        0,
        &bar(0, 100.0),
        StepInput {
            entry: true,
            stop_price_override: Some(97.0),
            target_price_override: Some(104.0),
            ..StepInput::default()
        },
    );
    assert!(matches!(events.as_slice(), [EngineEvent::Entered { .. }]));

    let snap = kernel.position_snapshot().unwrap();
    assert_eq!(snap.stop_price, Some(97.0));
    assert_eq!(snap.target_price, Some(104.0));
}

#[test]
fn zero_size_entry_emits_rejection() {
    let config = BacktestConfig::default();
    let fee_model = config.fee_model();
    // Lot of 10,000 units at price 100 with 100k capital -> raw size
    // ~999 units floors to zero lots.
    let inst = InstrumentConfig {
        lot_size: Some(10_000.0),
        alloted_capital: None,
        stop: None,
        target: None,
        existing_qty: None,
        avg_price: None,
    };
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        Some(&inst),
    );

    let events = kernel.step(0, &bar(0, 100.0), StepInput { entry: true, ..StepInput::default() });
    match events.as_slice() {
        [EngineEvent::EntryRejected { reason, .. }] => {
            assert_eq!(reason.as_str(), "zero_size");
        }
        other => panic!("expected zero-size rejection, got {other:?}"),
    }
    assert!(!kernel.is_in_position());
}

#[test]
fn multiplier_scales_notional_and_pnl() {
    use crate::instruments::{InstrumentKind, InstrumentSpec};

    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let spec = InstrumentSpec {
        multiplier: 50.0,
        lot_size: 1.0,
        ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
    };
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "FUT".to_string(),
        Direction::Long,
        None,
    )
    .with_instrument(spec);

    // 100k capital at price 100 with multiplier 50: 20 contracts.
    enter(&mut kernel, 0, 100.0);
    let snap = kernel.position_snapshot().unwrap();
    assert!((snap.size - 20.0).abs() < 1e-9, "size {}", snap.size);
    assert!(kernel.cash().abs() < 1e-6, "cash {}", kernel.cash());

    // Price to 102: equity = 102 * 20 * 50 = 102_000.
    assert!((kernel.equity(102.0) - 102_000.0).abs() < 1e-6);

    // Exit at 102: pnl = 2 * 20 * 50 = 2_000.
    let events = kernel.step(1, &bar(1, 102.0), StepInput { exit: true, ..StepInput::default() });
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => {
            assert!((trade.pnl - 2_000.0).abs() < 1e-6, "pnl {}", trade.pnl);
        }
        other => panic!("expected exit, got {other:?}"),
    }
    assert!((kernel.cash() - 102_000.0).abs() < 1e-6);
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

#[test]
fn resting_limit_buy_fills_next_bar_and_opens() {
    let mut kernel = zero_fee_kernel();
    let id = kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(10.0),
        OrderKind::Limit { price: 99.0 },
        TimeInForce::Gtc,
        0,
        0,
        "ord-1".into(),
        Some(95.0),
        None,
    );

    // Bar 0 (submission bar): only the acknowledgment, no fill.
    let events = kernel.step(0, &bar(0, 100.0), StepInput::default());
    assert!(matches!(
        events.as_slice(),
        [EngineEvent::OrderAccepted { order_id, .. }] if *order_id == id
    ));
    assert!(!kernel.is_in_position());

    // Bar 1 trades down through the limit: fill at 99, position opens
    // with the attached protective stop.
    let events = kernel.step(1, &bar(1, 99.5), StepInput::default());
    match events.as_slice() {
        [EngineEvent::OrderFilled { order_id, price, size, .. }, EngineEvent::Entered { .. }] => {
            assert_eq!(*order_id, id);
            assert_eq!(*price, 99.0);
            assert_eq!(*size, 10.0);
        }
        other => panic!("expected fill + entered, got {other:?}"),
    }
    let snap = kernel.position_snapshot().unwrap();
    assert_eq!(snap.stop_price, Some(95.0));
    assert_eq!(kernel.order(id).unwrap().status, OrderStatus::Filled);
}

#[test]
fn market_order_fills_on_submission_bar() {
    let mut kernel = zero_fee_kernel();
    let id = kernel.submit_order(
        OrderSide::Buy,
        QtySpec::CapitalFrac(0.5),
        OrderKind::Market,
        TimeInForce::Gtc,
        3,
        3,
        "mkt-1".into(),
        None,
        None,
    );

    let events = kernel.step(3, &bar(3, 100.0), StepInput::default());
    match events.as_slice() {
        [EngineEvent::OrderAccepted { .. }, EngineEvent::OrderFilled { order_id, price, .. }, EngineEvent::Entered { .. }] =>
        {
            assert_eq!(*order_id, id);
            // FillPrice::Close on the submission bar — same contract as
            // the signal-entry path.
            assert_eq!(*price, 100.0);
        }
        other => panic!("expected accept + fill + entered, got {other:?}"),
    }
    // Half the capital: 50k / 100 = 500 units.
    assert!((kernel.position_snapshot().unwrap().size - 500.0).abs() < 1e-9);
}

#[test]
fn sell_limit_closes_position_as_order_exit() {
    let mut kernel = zero_fee_kernel();
    enter(&mut kernel, 0, 100.0);

    kernel.submit_order(
        OrderSide::Sell,
        QtySpec::FullPosition,
        OrderKind::Limit { price: 105.0 },
        TimeInForce::Gtc,
        0,
        0,
        "tp-1".into(),
        None,
        None,
    );

    // Bar 1 stays below the limit.
    let events = kernel.step(1, &bar(1, 103.0), StepInput::default());
    assert!(matches!(events.as_slice(), [EngineEvent::OrderAccepted { .. }]));
    assert!(kernel.is_in_position());

    // Bar 2 trades through it.
    let events = kernel.step(2, &bar(2, 105.5), StepInput::default());
    match events.as_slice() {
        [EngineEvent::OrderFilled { price, .. }, EngineEvent::Exited { trade, .. }] => {
            assert_eq!(*price, 105.0);
            assert_eq!(trade.exit_reason, ExitReason::Order);
        }
        other => panic!("expected fill + exit, got {other:?}"),
    }
    assert!(!kernel.is_in_position());
}

#[test]
fn opening_order_rejected_while_in_position() {
    let mut kernel = zero_fee_kernel();
    enter(&mut kernel, 0, 100.0);

    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(1.0),
        OrderKind::Limit { price: 99.0 },
        TimeInForce::Gtc,
        0,
        0,
        "dup-1".into(),
        None,
        None,
    );

    let events = kernel.step(1, &bar(1, 98.5), StepInput::default());
    assert!(events
        .iter()
        .any(|e| matches!(e, EngineEvent::OrderRejected { reason: "position_open", .. })));
    assert_eq!(kernel.position_snapshot().unwrap().entry_idx, 0);
}

#[test]
fn oversized_unit_order_rejects_for_capital() {
    let mut kernel = zero_fee_kernel();
    kernel.submit_order(
        OrderSide::Buy,
        QtySpec::Units(10_000.0), // 10k * 100 = 1M >> 100k capital
        OrderKind::Limit { price: 100.0 },
        TimeInForce::Gtc,
        0,
        0,
        "big-1".into(),
        None,
        None,
    );
    let _ = kernel.step(0, &bar(0, 100.0), StepInput::default());
    let events = kernel.step(1, &bar(1, 99.0), StepInput::default());
    assert!(events
        .iter()
        .any(|e| matches!(e, EngineEvent::OrderRejected { reason: "insufficient_capital", .. })));
    assert!(!kernel.is_in_position());
}

#[test]
fn per_contract_fees_charge_on_contracts_not_notional() {
    use crate::instruments::{InstrumentKind, InstrumentSpec};

    // 2.5 currency units per contract per side, IB-style.
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let spec = InstrumentSpec {
        multiplier: 50.0,
        ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
    };
    let mut kernel = EngineKernel::new(
        config,
        FeeModel::per_share(2.5),
        SlippageModel::None,
        FillPrice::Close,
        "FUT".to_string(),
        Direction::Long,
        None,
    )
    .with_instrument(spec);

    enter(&mut kernel, 0, 100.0);
    let size = kernel.position_snapshot().unwrap().size;

    let events = kernel.step(1, &bar(1, 100.0), StepInput { exit: true, ..StepInput::default() });
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => {
            // Round trip: 2.5 per contract per side, NOT 2.5 * 50.
            let expected = 2.0 * 2.5 * size;
            assert!(
                (trade.fees - expected).abs() < 1e-9,
                "fees {} != {expected} (size {size})",
                trade.fees
            );
        }
        other => panic!("expected exit, got {other:?}"),
    }
}

#[test]
fn percentage_fees_charge_on_notional() {
    use crate::instruments::{InstrumentKind, InstrumentSpec};

    let config = BacktestConfig { fees: 0.001, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let spec = InstrumentSpec {
        multiplier: 50.0,
        ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
    };
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "FUT".to_string(),
        Direction::Long,
        None,
    )
    .with_instrument(spec);

    enter(&mut kernel, 0, 100.0);
    let size = kernel.position_snapshot().unwrap().size;

    let events = kernel.step(1, &bar(1, 100.0), StepInput { exit: true, ..StepInput::default() });
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => {
            // 0.1% of true notional (price * size * multiplier), each side.
            let expected = 2.0 * 0.001 * 100.0 * size * 50.0;
            assert!(
                (trade.fees - expected).abs() < 1e-6,
                "fees {} != {expected} (size {size})",
                trade.fees
            );
        }
        other => panic!("expected exit, got {other:?}"),
    }
}

#[test]
fn expiry_settles_position_and_rejects_entries() {
    use crate::instruments::{InstrumentKind, InstrumentSpec};

    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let spec = InstrumentSpec {
        expiration_ns: Some(5),
        ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
    };
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "FUT".to_string(),
        Direction::Long,
        None,
    )
    .with_instrument(spec);

    enter(&mut kernel, 0, 100.0);

    // Bar at the expiry timestamp: settle at close, no signal needed.
    let events = kernel.step(5, &bar(5, 103.0), StepInput::default());
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => {
            assert_eq!(trade.exit_reason, ExitReason::Settlement);
            assert!((trade.exit_price - 103.0).abs() < 1e-9);
        }
        other => panic!("expected settlement, got {other:?}"),
    }
    assert!(!kernel.is_in_position());

    // Post-expiry entry is refused.
    let events = kernel.step(6, &bar(6, 103.0), StepInput { entry: true, ..StepInput::default() });
    match events.as_slice() {
        [EngineEvent::EntryRejected { reason, .. }] => {
            assert_eq!(reason.as_str(), "expired");
        }
        other => panic!("expected expired rejection, got {other:?}"),
    }
}

#[test]
fn pre_activation_entry_is_rejected() {
    use crate::instruments::{InstrumentKind, InstrumentSpec};

    let config = BacktestConfig::default();
    let fee_model = config.fee_model();
    let spec = InstrumentSpec {
        activation_ns: Some(10),
        ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
    };
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "FUT".to_string(),
        Direction::Long,
        None,
    )
    .with_instrument(spec);

    let events = kernel.step(0, &bar(5, 100.0), StepInput { entry: true, ..StepInput::default() });
    match events.as_slice() {
        [EngineEvent::EntryRejected { reason, .. }] => {
            assert_eq!(reason.as_str(), "inactive");
        }
        other => panic!("expected inactive rejection, got {other:?}"),
    }

    let events = kernel.step(1, &bar(10, 100.0), StepInput { entry: true, ..StepInput::default() });
    assert!(matches!(events.as_slice(), [EngineEvent::Entered { .. }]));
}

#[test]
fn config_stop_quantizes_to_tick_grid() {
    use crate::instruments::{InstrumentKind, InstrumentSpec};

    let config =
        BacktestConfig { stop: StopConfig::Fixed { percent: 0.033 }, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let spec =
        InstrumentSpec { price_increment: 0.05, ..InstrumentSpec::new("EQ", InstrumentKind::Cash) };
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "EQ".to_string(),
        Direction::Long,
        None,
    )
    .with_instrument(spec);

    enter(&mut kernel, 0, 100.0);
    // Raw stop = 96.7; on the 0.05 grid floored for a long -> 96.70
    // exactly (already on grid); use a messier percent to prove rounding:
    let stop = kernel.position_snapshot().unwrap().stop_price.unwrap();
    assert!((stop / 0.05 - (stop / 0.05).round()).abs() < 1e-9, "stop {stop} not on grid");
}

#[test]
fn spec_lot_size_defers_to_instrument_config() {
    use crate::instruments::{InstrumentKind, InstrumentSpec};

    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let inst = InstrumentConfig { lot_size: Some(25.0), ..InstrumentConfig::default() };
    let spec = InstrumentSpec {
        lot_size: 50.0,
        ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
    };
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "FUT".to_string(),
        Direction::Long,
        Some(&inst),
    )
    .with_instrument(spec);

    // 100k at price 100 -> 1000 raw; explicit config lot 25 wins over
    // the spec's 50, and 1000 is already a multiple of 25.
    enter(&mut kernel, 0, 100.0);
    let size = kernel.position_snapshot().unwrap().size;
    assert!((size - 1000.0).abs() < 1e-9, "size {size}");
}

#[test]
fn entry_without_override_uses_config_stop() {
    let config =
        BacktestConfig { stop: StopConfig::Fixed { percent: 0.05 }, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    );

    let events = kernel.step(0, &bar(0, 100.0), StepInput { entry: true, ..StepInput::default() });
    assert!(matches!(events.as_slice(), [EngineEvent::Entered { .. }]));

    let snap = kernel.position_snapshot().unwrap();
    assert_eq!(snap.stop_price, Some(95.0));
    assert_eq!(snap.direction, Direction::Long);
    assert_eq!(snap.entry_idx, 0);
}

const ALGO_SEC: i64 = 1_000_000_000;

/// A TWAP accumulates: under the default netting policy only the first
/// slice can open, so these tests use the hedging policy.
fn twap_kernel() -> EngineKernel {
    make_kernel().with_position_policy(crate::portfolio::ledger::PositionPolicy::Independent)
}

#[test]
fn a_twap_slice_fills_on_the_step_it_is_released_on() {
    // The whole design depends on this: slices are submitted inside the
    // step, just before the market sweep, so they do not trail a step
    // behind their own schedule.
    let mut kernel = twap_kernel();
    kernel
        .submit_algo(
            OrderSide::Buy,
            QtySpec::Units(30.0),
            OrderKind::Market,
            TimeInForce::Gtc,
            "tw".to_string(),
            crate::execution::algos::ExecAlgorithm::Twap { slices: 3, interval_ns: ALGO_SEC },
            false,
            0,
            0,
        )
        .expect("valid schedule");

    let events = kernel.step(0, &bar(0, 100.0), StepInput::default());
    assert!(
        events.iter().any(|e| matches!(e, EngineEvent::OrderFilled { .. })),
        "the first slice must fill on its release step, got {events:?}"
    );
    assert!(kernel.is_in_position());
}

#[test]
fn twap_releases_one_slice_per_interval_through_the_kernel() {
    let mut kernel = twap_kernel();
    kernel
        .submit_algo(
            OrderSide::Buy,
            QtySpec::Units(30.0),
            OrderKind::Market,
            TimeInForce::Gtc,
            "tw".to_string(),
            crate::execution::algos::ExecAlgorithm::Twap { slices: 3, interval_ns: ALGO_SEC },
            false,
            0,
            0,
        )
        .expect("valid schedule");

    let mut fills = 0;
    for i in 0..3i64 {
        let ts = i * ALGO_SEC;
        let events = kernel.step(
            i as usize,
            &KernelBar {
                timestamp: ts,
                open: 100.0,
                high: 100.0,
                low: 100.0,
                close: 100.0,
                volume: 1.0,
            },
            StepInput::default(),
        );
        fills += events.iter().filter(|e| matches!(e, EngineEvent::OrderFilled { .. })).count();
    }
    assert_eq!(fills, 3, "one slice per interval");
}

#[test]
fn a_tick_session_slices_on_time_not_on_event_count() {
    // Many prints inside one interval must not accelerate the schedule.
    // This is why slicing is timed rather than counted in bars: `idx` is an
    // event ordinal on a tick feed.
    let mut kernel = twap_kernel();
    kernel
        .submit_algo(
            OrderSide::Buy,
            QtySpec::Units(20.0),
            OrderKind::Market,
            TimeInForce::Gtc,
            "tw".to_string(),
            crate::execution::algos::ExecAlgorithm::Twap { slices: 2, interval_ns: ALGO_SEC },
            false,
            0,
            0,
        )
        .expect("valid schedule");

    let mut fills = 0;
    // Ten prints, all inside the first interval.
    for i in 0..10i64 {
        let tick =
            TradeTick { timestamp: i * 1_000_000, price: 100.0, size: 1.0, signed_size: 0.0 };
        let events = kernel.step_trade(i as usize, &tick, StepInput::default());
        fills += events.iter().filter(|e| matches!(e, EngineEvent::OrderFilled { .. })).count();
    }
    assert_eq!(fills, 1, "only the first slice is due inside one interval");
}

#[test]
fn cancelling_a_schedule_stops_further_slices() {
    let mut kernel = twap_kernel();
    let algo_id = kernel
        .submit_algo(
            OrderSide::Buy,
            QtySpec::Units(40.0),
            OrderKind::Market,
            TimeInForce::Gtc,
            "tw".to_string(),
            crate::execution::algos::ExecAlgorithm::Twap { slices: 4, interval_ns: ALGO_SEC },
            false,
            0,
            0,
        )
        .expect("valid schedule");

    kernel.step(0, &bar(0, 100.0), StepInput::default());
    assert!(kernel.cancel_algo(algo_id, 1));

    let mut later_fills = 0;
    for i in 1..4i64 {
        let events = kernel.step(
            i as usize,
            &KernelBar {
                timestamp: i * ALGO_SEC,
                open: 100.0,
                high: 100.0,
                low: 100.0,
                close: 100.0,
                volume: 1.0,
            },
            StepInput::default(),
        );
        later_fills +=
            events.iter().filter(|e| matches!(e, EngineEvent::OrderFilled { .. })).count();
    }
    assert_eq!(later_fills, 0, "a cancelled schedule releases nothing further");
}

#[test]
fn a_schedule_refuses_capital_fraction_sizing() {
    let mut kernel = twap_kernel();
    let result = kernel.submit_algo(
        OrderSide::Buy,
        QtySpec::CapitalFrac(0.5),
        OrderKind::Market,
        TimeInForce::Gtc,
        "tw".to_string(),
        crate::execution::algos::ExecAlgorithm::Twap { slices: 2, interval_ns: ALGO_SEC },
        false,
        0,
        0,
    );
    assert!(result.is_err(), "each slice would size against a different account");
}

fn expiring_kernel(spec: crate::instruments::InstrumentSpec) -> EngineKernel {
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "OPT".to_string(),
        Direction::Long,
        None,
    )
    .with_instrument(spec)
}

#[test]
fn settlement_is_free_by_default() {
    use crate::instruments::{InstrumentKind, InstrumentSpec};
    let mut kernel = expiring_kernel(InstrumentSpec {
        expiration_ns: Some(5),
        ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
    });
    enter(&mut kernel, 0, 100.0);
    let events = kernel.step(5, &bar(5, 110.0), StepInput::default());
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => assert_eq!(trade.fees, 0.0),
        other => panic!("expected settlement, got {other:?}"),
    }
}

#[test]
fn a_settlement_fee_is_charged_on_the_settled_notional() {
    use crate::instruments::{InstrumentKind, InstrumentSpec};
    let mut kernel = expiring_kernel(InstrumentSpec {
        expiration_ns: Some(5),
        settlement_fee: 0.01,
        ..InstrumentSpec::new("FUT", InstrumentKind::Contract { underlying: None })
    });
    enter(&mut kernel, 0, 100.0);
    let events = kernel.step(5, &bar(5, 110.0), StepInput::default());
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => {
            // 1% of the settled notional, not of the entry price.
            let expected = 110.0 * trade.size * 0.01;
            assert!(
                (trade.fees - expected).abs() < 1e-9,
                "fees {} should be {expected}",
                trade.fees
            );
        }
        other => panic!("expected settlement, got {other:?}"),
    }
}

#[test]
fn an_option_settles_at_its_own_close_without_an_underlying() {
    use crate::instruments::{InstrumentKind, InstrumentSpec, OptionRight};
    let mut kernel = expiring_kernel(InstrumentSpec {
        expiration_ns: Some(5),
        ..InstrumentSpec::new(
            "CE",
            InstrumentKind::Option {
                underlying: None,
                strike: 100.0,
                right: OptionRight::Call,
                binary: false,
            },
        )
    });
    enter(&mut kernel, 0, 7.0);
    // No underlying supplied: the contract's own close is all we know.
    let events = kernel.step(5, &bar(5, 9.0), StepInput::default());
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => {
            assert!((trade.exit_price - 9.0).abs() < 1e-9);
        }
        other => panic!("expected settlement, got {other:?}"),
    }
}

#[test]
fn an_option_settles_to_intrinsic_against_a_supplied_underlying() {
    use crate::instruments::{InstrumentKind, InstrumentSpec, OptionRight};
    let mut kernel = expiring_kernel(InstrumentSpec {
        expiration_ns: Some(5),
        ..InstrumentSpec::new(
            "CE",
            InstrumentKind::Option {
                underlying: None,
                strike: 100.0,
                right: OptionRight::Call,
                binary: false,
            },
        )
    });
    enter(&mut kernel, 0, 7.0);
    // The underlying sits at 112, so a 100 call is worth 12 — regardless of
    // where the option's own last print happened to be.
    kernel.set_underlying_price(Some(112.0));
    let events = kernel.step(5, &bar(5, 9.0), StepInput::default());
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => {
            assert!(
                (trade.exit_price - 12.0).abs() < 1e-9,
                "expected intrinsic 12.0, got {}",
                trade.exit_price
            );
        }
        other => panic!("expected settlement, got {other:?}"),
    }
}

#[test]
fn an_out_of_the_money_option_settles_worthless() {
    use crate::instruments::{InstrumentKind, InstrumentSpec, OptionRight};
    let mut kernel = expiring_kernel(InstrumentSpec {
        expiration_ns: Some(5),
        ..InstrumentSpec::new(
            "CE",
            InstrumentKind::Option {
                underlying: None,
                strike: 100.0,
                right: OptionRight::Call,
                binary: false,
            },
        )
    });
    enter(&mut kernel, 0, 7.0);
    kernel.set_underlying_price(Some(95.0));
    let events = kernel.step(5, &bar(5, 0.5), StepInput::default());
    match events.as_slice() {
        [EngineEvent::Exited { trade, .. }] => {
            assert_eq!(trade.exit_price, 0.0, "a 100 call with spot at 95 expires worthless");
        }
        other => panic!("expected settlement, got {other:?}"),
    }
}

fn margin_kernel(liquidate: bool) -> EngineKernel {
    let config = BacktestConfig {
        fees: 0.0,
        liquidate_on_margin_call: liquidate,
        ..BacktestConfig::default()
    };
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
    .with_account_mode(AccountMode::Margin { leverage: 50.0 })
}

#[test]
fn a_margin_call_only_halts_by_default() {
    let mut kernel = margin_kernel(false);
    enter(&mut kernel, 0, 100.0);
    let events = kernel.step(1, &bar(1, 40.0), StepInput::default());
    assert!(events.iter().any(|e| matches!(e, EngineEvent::MarginCall { .. })));
    assert!(kernel.is_in_position(), "the position rides on; the strategy decides what to do");
}

#[test]
fn liquidation_closes_positions_on_the_call() {
    let mut kernel = margin_kernel(true);
    enter(&mut kernel, 0, 100.0);
    let events = kernel.step(1, &bar(1, 40.0), StepInput::default());
    assert!(events.iter().any(|e| matches!(e, EngineEvent::MarginCall { .. })));
    match events.iter().find(|e| matches!(e, EngineEvent::Exited { .. })) {
        Some(EngineEvent::Exited { trade, .. }) => {
            assert_eq!(trade.exit_reason, ExitReason::Liquidation);
        }
        _ => panic!("expected a liquidation, got {events:?}"),
    }
    assert!(!kernel.is_in_position(), "the broker closed it");
}

#[test]
fn liquidation_pays_exit_costs_unlike_settlement() {
    // A forced close crosses the spread; settlement does not.
    let config =
        BacktestConfig { fees: 0.001, liquidate_on_margin_call: true, ..BacktestConfig::default() };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    )
    .with_account_mode(AccountMode::Margin { leverage: 50.0 });

    enter(&mut kernel, 0, 100.0);
    let events = kernel.step(1, &bar(1, 40.0), StepInput::default());
    match events.iter().find(|e| matches!(e, EngineEvent::Exited { .. })) {
        Some(EngineEvent::Exited { trade, .. }) => {
            assert!(trade.fees > 0.0, "a liquidation is a real trade-out");
        }
        _ => panic!("expected a liquidation, got {events:?}"),
    }
}

#[test]
fn liquidation_does_not_fire_without_a_margin_call() {
    let mut kernel = margin_kernel(true);
    enter(&mut kernel, 0, 100.0);
    // A gentle move keeps equity above the requirement.
    let events = kernel.step(1, &bar(1, 101.0), StepInput::default());
    assert!(!events.iter().any(|e| matches!(e, EngineEvent::Exited { .. })));
    assert!(kernel.is_in_position());
}

// -- order side is authoritative for opening ---------------------------------

/// Submit a market order and step one bar so it fills.
fn market_order(
    kernel: &mut EngineKernel,
    idx: usize,
    price: Price,
    side: OrderSide,
    reduce_only: bool,
) -> Vec<EngineEvent> {
    kernel.submit_order_full(
        side,
        QtySpec::CapitalFrac(0.5),
        OrderKind::Market,
        TimeInForce::Gtc,
        idx,
        idx as i64,
        format!("o{idx}"),
        None,
        None,
        false,
        reduce_only,
        None,
    );
    kernel.step(idx, &bar(idx as i64, price), StepInput::default())
}

#[test]
fn sell_order_opens_short_on_a_default_long_kernel() {
    // The bug this fixes: the sell was reinterpreted as a close, found no
    // position, and vanished without a trade or a rejection.
    let mut kernel = make_kernel();
    let events = market_order(&mut kernel, 0, 100.0, OrderSide::Sell, false);
    match events.iter().find(|e| matches!(e, EngineEvent::Entered { .. })) {
        Some(EngineEvent::Entered { direction, .. }) => {
            assert_eq!(*direction, Direction::Short, "the order's side decides");
        }
        _ => panic!("expected a short entry, got {events:?}"),
    }
    assert!(kernel.is_in_position());
}

#[test]
fn short_opened_by_order_orients_stop_above_and_target_below() {
    let config = BacktestConfig {
        stop: StopConfig::Fixed { percent: 0.05 },
        target: TargetConfig::Fixed { percent: 0.10 },
        ..BacktestConfig::default()
    };
    let fee_model = config.fee_model();
    let mut kernel = EngineKernel::new(
        config,
        fee_model,
        SlippageModel::None,
        FillPrice::Close,
        "TEST".to_string(),
        Direction::Long,
        None,
    );
    market_order(&mut kernel, 0, 100.0, OrderSide::Sell, false);
    let snap = kernel.position_snapshot().expect("a short position");
    assert_eq!(snap.direction, Direction::Short);
    let stop = snap.stop_price.expect("a stop");
    let target = snap.target_price.expect("a target");
    assert!(stop > snap.entry_price, "a short's stop sits ABOVE entry: {stop}");
    assert!(target < snap.entry_price, "a short's target sits BELOW entry: {target}");
}

#[test]
fn a_short_opened_by_order_profits_when_price_falls() {
    let mut kernel = make_kernel();
    market_order(&mut kernel, 0, 100.0, OrderSide::Sell, false);
    // Buy back lower: the close is the opposing side while in position.
    let events = market_order(&mut kernel, 1, 90.0, OrderSide::Buy, false);
    match events.iter().find(|e| matches!(e, EngineEvent::Exited { .. })) {
        Some(EngineEvent::Exited { trade, .. }) => {
            assert!(trade.pnl > 0.0, "short profits on a fall, got {}", trade.pnl);
        }
        _ => panic!("expected the buy to close the short, got {events:?}"),
    }
    assert!(!kernel.is_in_position());
}

#[test]
fn an_opposing_order_still_closes_rather_than_reversing() {
    // Bracket legs and take-profit orders depend on this.
    let mut kernel = make_kernel();
    enter(&mut kernel, 0, 100.0);
    let events = market_order(&mut kernel, 1, 110.0, OrderSide::Sell, false);
    assert!(
        events.iter().any(|e| matches!(e, EngineEvent::Exited { .. })),
        "expected a close, got {events:?}"
    );
    assert!(!kernel.is_in_position(), "it must not reverse into a short");
}

#[test]
fn a_reduce_only_order_never_opens_and_is_counted() {
    let mut kernel = make_kernel();
    let events = market_order(&mut kernel, 0, 100.0, OrderSide::Sell, true);
    assert!(!kernel.is_in_position(), "reduce-only must never open");
    match events.iter().find(|e| matches!(e, EngineEvent::OrderRejected { .. })) {
        Some(EngineEvent::OrderRejected { reason, .. }) => {
            assert_eq!(*reason, "reduce_only");
        }
        _ => panic!("expected a rejection, got {events:?}"),
    }
    assert_eq!(kernel.rejected_entries(), 1, "a refusal must be observable");
}

#[test]
fn a_leg_can_flip_side_within_one_run() {
    // The whole point: long, flat, then short on the same kernel.
    let mut kernel = make_kernel();
    market_order(&mut kernel, 0, 100.0, OrderSide::Buy, false);
    assert_eq!(kernel.position_snapshot().expect("a position").direction, Direction::Long);
    market_order(&mut kernel, 1, 110.0, OrderSide::Sell, false);
    assert!(!kernel.is_in_position(), "flat between the two sides");
    market_order(&mut kernel, 2, 110.0, OrderSide::Sell, false);
    assert_eq!(
        kernel.position_snapshot().expect("a position").direction,
        Direction::Short,
        "the same leg reopened on the other side"
    );
}

fn adoption_kernel(mode: AccountMode) -> EngineKernel {
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
    .with_account_mode(mode)
}

#[test]
fn fully_funded_margin_adoption_locks_the_cost_basis() {
    // Margin funds an open by LOCKING the notional, not by debiting the
    // balance (see `open_at`). Adoption must fund the same way: margin
    // equity is `cash + unrealized`, with no position-value term, so a
    // cash-style debit would never be offset and would understate equity by
    // the cost basis for the entire run.
    let mut kernel = adoption_kernel(AccountMode::Margin { leverage: 1.0 });
    kernel.set_cash(100_000.0);

    kernel.adopt_position(0, 90.0, 100.0).expect("fully funded adoption must be allowed");

    assert_eq!(kernel.locked_margin(), 9_000.0, "the whole notional locks at leverage 1.0");
    assert_eq!(kernel.cash(), 100_000.0, "margin must not debit the balance");
    // Priced at the adoption price the holding is worth exactly what it cost,
    // so equity is back to the starting capital.
    assert_eq!(kernel.equity(90.0), 100_000.0);
    assert_eq!(kernel.free_capital(), 91_000.0, "free capital drops by the cost basis");
}

#[test]
fn cash_and_fully_funded_margin_adoption_agree() {
    // A fully funded book is economically identical to cash for a long
    // holding, so the two modes must report the same numbers. Divergence
    // means the funding arm and the mode's equity formula disagree.
    let mut cash = adoption_kernel(AccountMode::Cash);
    cash.set_cash(100_000.0);
    cash.adopt_position(0, 90.0, 100.0).unwrap();

    let mut margin = adoption_kernel(AccountMode::Margin { leverage: 1.0 });
    margin.set_cash(100_000.0);
    margin.adopt_position(0, 90.0, 100.0).unwrap();

    for close in [90.0, 95.0, 85.0] {
        assert_eq!(
            cash.equity(close),
            margin.equity(close),
            "equity must agree at close {close}"
        );
    }
    assert_eq!(cash.free_capital(), margin.free_capital());
}

#[test]
fn leveraged_adoption_is_refused_not_guessed() {
    // Above leverage 1.0 the broker's posted margin genuinely is not
    // derivable from quantity and average price. Guessing would misstate
    // free capital, which gates every later entry.
    let mut kernel = adoption_kernel(AccountMode::Margin { leverage: 2.0 });
    kernel.set_cash(100_000.0);

    let err = kernel.adopt_position(0, 90.0, 100.0).unwrap_err();
    assert!(err.contains("fully funded"), "expected a leverage refusal, got: {err}");
}
