//! Fixed-point accounting: cash, the cost basis of the open positions and
//! realised P&L reconcile to the minor unit, as integers.
//!
//! The kernel books every amount that moves cash (the notional of a fill, a
//! fee, a P&L) as a count of minor units. These tests drive it with random
//! fill sequences (partial fills, fees, shorts, flips, margin accounts) and
//! compare integers with `==`: no tolerance anywhere.
//!
//! Cash account:  cash = initial + sum(realised pnl) - (open cost basis + open entry fees)
//! Margin account: cash = initial + sum(realised pnl) - open entry fees
//!   (a lock holds margin back from use but does not debit cash)

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};

use raptorbt::accounts::{AccountMode, Money};
use raptorbt::core::types::{BacktestConfig, Direction, ExitReason, Trade};
use raptorbt::execution::orders::{OrderKind, OrderSide, QtySpec, TimeInForce};
use raptorbt::execution::{FeeModel, FillPrice, SlippageModel};
use raptorbt::instruments::{InstrumentKind, InstrumentSpec, SettlementPrice};
use raptorbt::portfolio::{EngineEvent, EngineKernel, KernelBar, StepInput};

const INITIAL: f64 = 1_000_000.0;
const INITIAL_MINOR: i128 = 100_000_000;

fn config(cases: u32, seed: u64) -> Config {
    Config {
        cases,
        failure_persistence: None,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(seed),
        max_shrink_iters: 4096,
        ..Config::default()
    }
}

#[derive(Debug, Clone)]
enum Fees {
    None,
    Pct(f64),
    Fixed(f64),
    PerShare(f64),
}

impl Fees {
    fn model(&self) -> FeeModel {
        match self {
            Fees::None => FeeModel::None,
            Fees::Pct(r) => FeeModel::Percentage(*r),
            Fees::Fixed(a) => FeeModel::Fixed(*a),
            Fees::PerShare(r) => FeeModel::PerShare(*r),
        }
    }
}

#[derive(Debug, Clone)]
struct Setup {
    fees: Fees,
    leverage: Option<f64>,
    partial: bool,
    mult: f64,
    decimals: u32,
}

fn setup() -> impl Strategy<Value = Setup> {
    (
        prop_oneof![
            Just(Fees::None),
            (0.0001f64..0.02).prop_map(Fees::Pct),
            (0.0f64..40.0).prop_map(Fees::Fixed),
            (0.0001f64..0.5).prop_map(Fees::PerShare),
        ],
        prop_oneof![Just(None), (1.5f64..8.0).prop_map(Some)],
        any::<bool>(),
        prop_oneof![Just(1.0), Just(5.0), Just(50.0)],
        prop_oneof![Just(2u32), Just(2), Just(0), Just(4)],
    )
        .prop_map(|(fees, leverage, partial, mult, decimals)| Setup {
            fees,
            leverage,
            partial,
            mult,
            decimals,
        })
}

fn kernel(s: &Setup) -> EngineKernel {
    let cfg = BacktestConfig {
        initial_capital: INITIAL,
        partial_fills: s.partial,
        money_decimals: s.decimals,
        ..BacktestConfig::default()
    };
    let mut spec = InstrumentSpec::new("T", InstrumentKind::Cash);
    spec.multiplier = s.mult;
    let k = EngineKernel::new(
        cfg,
        s.fees.model(),
        SlippageModel::Percentage(0.001),
        FillPrice::Close,
        "T".to_string(),
        Direction::Long,
        None,
    )
    .with_instrument(spec);
    match s.leverage {
        Some(leverage) => k.with_account_mode(AccountMode::Margin { leverage }),
        None => k,
    }
}

/// One bar's draw: close move, and per-bar orders (buy?, units, reduce_only).
type Step = (f64, Vec<(bool, f64, bool)>);

fn steps() -> impl Strategy<Value = Vec<Step>> {
    prop::collection::vec(
        (
            -0.06f64..0.06,
            prop::collection::vec((any::<bool>(), 1.0f64..400.0, prop::bool::weighted(0.25)), 0..3),
        ),
        4..60,
    )
}

fn check_trade(t: &Trade, decimals: u32) -> Result<(), TestCaseError> {
    let m = t.money.expect("kernel trades carry booked amounts");
    prop_assert_eq!(u32::from(m.places), decimals);
    let gross = match t.direction {
        Direction::Long => m.exit_notional - m.entry_notional,
        Direction::Short => m.entry_notional - m.exit_notional,
    };
    prop_assert_eq!(m.pnl, gross - m.entry_fees - m.exit_fees);
    let scale = 10f64.powi(decimals as i32);
    prop_assert_eq!(t.pnl, m.pnl as f64 / scale);
    prop_assert_eq!(t.fees, (m.entry_fees + m.exit_fees) as f64 / scale);
    Ok(())
}

fn reconcile(
    k: &EngineKernel,
    margin: bool,
    initial: Money,
    realised: i128,
) -> Result<(), TestCaseError> {
    let (cost, fees) = k.open_booked();
    let open = if margin { fees } else { cost + fees };
    prop_assert_eq!(
        k.cash_money().minor(),
        initial.minor() + realised - open.minor(),
        "cash {} realised {} open cost {} open fees {}",
        k.cash_money().minor(),
        realised,
        cost.minor(),
        fees.minor()
    );
    Ok(())
}

proptest! {
    #![proptest_config(config(1500, 0xA2A2))]

    /// After every bar: cash, open cost basis and realised P&L reconcile
    /// exactly; every closed trade's P&L is its booked notionals less its
    /// booked fees; once flat, cash is initial capital plus realised P&L.
    #[test]
    fn cash_cost_basis_and_realised_pnl_reconcile_exactly(s in setup(), steps in steps()) {
        let mut k = kernel(&s);
        let initial = k.cash_money();
        let margin = s.leverage.is_some();
        let mut realised = 0i128;
        let mut close = 100.0f64;
        let mut last_bar = None;
        for (i, (ret, orders)) in steps.iter().enumerate() {
            let open = close;
            close *= 1.0 + ret;
            let bar = KernelBar {
                timestamp: i as i64 * 60_000_000_000,
                open,
                high: open.max(close) * 1.002,
                low: open.min(close) * 0.998,
                close,
                volume: 1_000.0,
            };
            for (n, (buy, units, reduce_only)) in orders.iter().enumerate() {
                k.submit_order_full(
                    if *buy { OrderSide::Buy } else { OrderSide::Sell },
                    QtySpec::Units(*units),
                    OrderKind::Market,
                    TimeInForce::Gtc,
                    i,
                    bar.timestamp,
                    format!("o{i}-{n}"),
                    None,
                    None,
                    false,
                    *reduce_only,
                    None,
                );
            }
            for event in k.step(i, &bar, StepInput::default()) {
                if let EngineEvent::Exited { trade, .. } = event {
                    check_trade(&trade, s.decimals)?;
                    realised += trade.money.unwrap().pnl;
                }
            }
            reconcile(&k, margin, initial, realised)?;
            last_bar = Some((i, bar));
        }
        let (i, bar) = last_bar.unwrap();
        for trade in k.finalize_all(i, &bar) {
            check_trade(&trade, s.decimals)?;
            realised += trade.money.unwrap().pnl;
        }
        reconcile(&k, margin, initial, realised)?;
        prop_assert_eq!(k.open_count(), 0);
        prop_assert_eq!(k.cash_money().minor(), initial.minor() + realised);
    }
}

proptest! {
    #![proptest_config(config(600, 0xA2B3))]

    /// The same identity across a contract's expiry: positions held to the
    /// expiry settle at the bar's close (a declared price in half the cases),
    /// pay a settlement fee booked in minor units, and cash, cost basis and
    /// realised P&L still agree to the minor unit; later orders are refused.
    #[test]
    fn settlement_at_expiry_keeps_the_books_exact(
        s in setup(),
        steps in steps(),
        fee in prop_oneof![Just(0.0), Just(0.0005), Just(0.003)],
        declared in prop::option::of(50.0f64..150.0),
    ) {
        let mut k = kernel(&s);
        let mut spec = InstrumentSpec::new("T", InstrumentKind::Contract { underlying: None });
        spec.multiplier = s.mult;
        spec.expiration_ns = Some((steps.len() as i64 / 2).max(1) * 60_000_000_000);
        spec.settlement_fee = fee;
        if let Some(p) = declared {
            spec.settlement.price = SettlementPrice::Fixed(p);
        }
        k.set_instrument(spec);
        let initial = k.cash_money();
        let margin = s.leverage.is_some();
        let mut realised = 0i128;
        let mut close = 100.0f64;
        let mut settled = false;
        for (i, (ret, orders)) in steps.iter().enumerate() {
            let open = close;
            close *= 1.0 + ret;
            let bar = KernelBar {
                timestamp: i as i64 * 60_000_000_000,
                open,
                high: open.max(close) * 1.002,
                low: open.min(close) * 0.998,
                close,
                volume: 1_000.0,
            };
            for (n, (buy, units, reduce_only)) in orders.iter().enumerate() {
                k.submit_order_full(
                    if *buy { OrderSide::Buy } else { OrderSide::Sell },
                    QtySpec::Units(*units),
                    OrderKind::Market,
                    TimeInForce::Gtc,
                    i,
                    bar.timestamp,
                    format!("o{i}-{n}"),
                    None,
                    None,
                    false,
                    *reduce_only,
                    None,
                );
            }
            for event in k.step(i, &bar, StepInput::default()) {
                match event {
                    EngineEvent::Exited { trade, .. } => {
                        check_trade(&trade, s.decimals)?;
                        settled |= trade.exit_reason == ExitReason::Settlement;
                        realised += trade.money.unwrap().pnl;
                    }
                    EngineEvent::Entered { .. } => {
                        prop_assert!(!settled, "a position opened after the contract settled");
                    }
                    _ => {}
                }
            }
            reconcile(&k, margin, initial, realised)?;
        }
        if settled {
            prop_assert_eq!(k.open_count(), 0);
        }
    }
}

/// The property above proves nothing unless its cases close shorts and
/// partial slices; count what a slice of them does.
#[test]
fn the_property_closes_trades_of_both_directions() {
    use proptest::strategy::ValueTree;
    let mut runner = proptest::test_runner::TestRunner::new(config(1, 7));
    let (mut trades, mut shorts, mut partial_slices) = (0usize, 0usize, 0usize);
    for _ in 0..300 {
        let (s, steps) = (setup(), steps()).new_tree(&mut runner).unwrap().current();
        let mut k = kernel(&s);
        let mut close = 100.0f64;
        for (i, (ret, orders)) in steps.iter().enumerate() {
            let open = close;
            close *= 1.0 + ret;
            let bar = KernelBar {
                timestamp: i as i64 * 60_000_000_000,
                open,
                high: open.max(close) * 1.002,
                low: open.min(close) * 0.998,
                close,
                volume: 1_000.0,
            };
            for (n, (buy, units, ro)) in orders.iter().enumerate() {
                k.submit_order_full(
                    if *buy { OrderSide::Buy } else { OrderSide::Sell },
                    QtySpec::Units(*units),
                    OrderKind::Market,
                    TimeInForce::Gtc,
                    i,
                    bar.timestamp,
                    format!("o{i}-{n}"),
                    None,
                    None,
                    false,
                    *ro,
                    None,
                );
            }
            for e in k.step(i, &bar, StepInput::default()) {
                if let EngineEvent::Exited { trade, .. } = e {
                    trades += 1;
                    if trade.direction == Direction::Short {
                        shorts += 1;
                    }
                    if k.open_count() > 0 {
                        partial_slices += 1;
                    }
                }
            }
        }
    }
    eprintln!("COVERAGE trades {trades} shorts {shorts} partial {partial_slices}");
    assert!(trades > 500 && shorts > 100 && partial_slices > 50);
}

/// A million fills, 500,000 round trips: cash is still exactly the initial
/// capital plus the sum of the booked P&L, where a float account is off by
/// accumulated rounding.
#[test]
fn a_million_fills_add_no_drift() {
    let cfg =
        BacktestConfig { initial_capital: INITIAL, fees: 0.0003, ..BacktestConfig::default() };
    let mut k = EngineKernel::new(
        cfg.clone(),
        cfg.fee_model(),
        SlippageModel::None,
        FillPrice::Close,
        "T".to_string(),
        Direction::Long,
        None,
    );
    let mut realised = 0i128;
    let mut float_cash = INITIAL; // the float account this replaces
    let mut price = 100.0f64;
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for i in 0..1_000_000usize {
        // xorshift, so the path is fixed
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        price *= 1.0 + ((state % 2001) as f64 - 1000.0) / 2_000_000.0;
        let bar = KernelBar {
            timestamp: i as i64 * 1_000_000_000,
            open: price,
            high: price * 1.001,
            low: price * 0.999,
            close: price,
            volume: 1_000.0,
        };
        let input = if i % 2 == 0 {
            StepInput { entry: true, size_mult: Some(0.0005), ..StepInput::default() }
        } else {
            StepInput { exit: true, ..StepInput::default() }
        };
        for event in k.step(i, &bar, input) {
            if let EngineEvent::Exited { trade, .. } = event {
                realised += trade.money.unwrap().pnl;
                float_cash += trade.pnl;
            }
        }
    }
    assert_eq!(k.open_count(), 0);
    assert!(realised != 0, "the run must have traded");
    assert!(k.cash_money().minor() > INITIAL_MINOR / 2, "the run must not have been wiped out");
    assert_eq!(k.cash_money().minor(), INITIAL_MINOR + realised);
    // The f64 account of the same trades agrees to float noise, not exactly.
    let exact = (INITIAL_MINOR + realised) as f64 / 100.0;
    eprintln!(
        "float account {float_cash:.6} exact {exact:.2} difference {:.3e}",
        float_cash - exact
    );
    assert!((float_cash - exact).abs() < 1.0, "float {float_cash} exact {exact}");
}
