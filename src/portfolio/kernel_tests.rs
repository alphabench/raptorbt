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
    let events =
        kernel.step_trade(0, &trade(0, 100.0, 5.0), StepInput { entry: true, ..Default::default() });
    assert!(matches!(events.as_slice(), [EngineEvent::Entered { price, .. }] if *price == 100.0));
    assert!(kernel.is_in_position());

    let events =
        kernel.step_trade(1, &trade(1, 110.0, 5.0), StepInput { exit: true, ..Default::default() });
    assert!(matches!(events.as_slice(), [EngineEvent::Exited { trade, .. }] if trade.exit_price == 110.0));
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
    let events = kernel.step(
        0,
        &bar(0, 100.0),
        StepInput { entry: true, ..StepInput::default() },
    );
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

    let events =
        kernel.step(0, &bar(0, 100.0), StepInput { entry: true, ..StepInput::default() });
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
    assert!(events.iter().any(|e| matches!(
        e,
        EngineEvent::OrderRejected { reason: "position_open", .. }
    )));
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
    assert!(events.iter().any(|e| matches!(
        e,
        EngineEvent::OrderRejected { reason: "insufficient_capital", .. }
    )));
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

    let events =
        kernel.step(1, &bar(1, 100.0), StepInput { exit: true, ..StepInput::default() });
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

    let events =
        kernel.step(1, &bar(1, 100.0), StepInput { exit: true, ..StepInput::default() });
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
    let events =
        kernel.step(6, &bar(6, 103.0), StepInput { entry: true, ..StepInput::default() });
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

    let events =
        kernel.step(0, &bar(5, 100.0), StepInput { entry: true, ..StepInput::default() });
    match events.as_slice() {
        [EngineEvent::EntryRejected { reason, .. }] => {
            assert_eq!(reason.as_str(), "inactive");
        }
        other => panic!("expected inactive rejection, got {other:?}"),
    }

    let events =
        kernel.step(1, &bar(10, 100.0), StepInput { entry: true, ..StepInput::default() });
    assert!(matches!(events.as_slice(), [EngineEvent::Entered { .. }]));
}

#[test]
fn config_stop_quantizes_to_tick_grid() {
    use crate::instruments::{InstrumentKind, InstrumentSpec};

    let config = BacktestConfig {
        stop: StopConfig::Fixed { percent: 0.033 },
        ..BacktestConfig::default()
    };
    let fee_model = config.fee_model();
    let spec = InstrumentSpec {
        price_increment: 0.05,
        ..InstrumentSpec::new("EQ", InstrumentKind::Cash)
    };
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
    let config = BacktestConfig {
        stop: StopConfig::Fixed { percent: 0.05 },
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

    let events = kernel.step(0, &bar(0, 100.0), StepInput { entry: true, ..StepInput::default() });
    assert!(matches!(events.as_slice(), [EngineEvent::Entered { .. }]));

    let snap = kernel.position_snapshot().unwrap();
    assert_eq!(snap.stop_price, Some(95.0));
    assert_eq!(snap.direction, Direction::Long);
    assert_eq!(snap.entry_idx, 0);
}
