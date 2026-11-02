//! Expiry, settlement, delivery and corporate actions on a book: the numbers,
//! and the integer reconciliation across all of them.
//!
//! The reconciliation compares integers with `==`, as `accounting_money.rs`
//! does: the account's balance is the initial cash plus every booked P&L
//! (trades, settlements, deliveries, cash in lieu) plus every dividend, less
//! what the open positions still hold (their cost basis and entry fees in a
//! cash account, their entry fees in a margin one).

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};

use raptorbt::accounts::AccountMode;
use raptorbt::core::types::{BacktestConfig, Direction, ExitReason, Trade};
use raptorbt::execution::orders::{OrderKind, OrderSide, QtySpec, TimeInForce};
use raptorbt::instruments::{
    AdjustmentPolicy, CorporateAction, InstrumentKind, InstrumentSpec, OptionRight,
    SettlementPrice, SettlementSource, SettlementStyle,
};
use raptorbt::portfolio::session::{ScheduleData, ScheduleEntry};
use raptorbt::portfolio::{EngineEvent, KernelBar, PortfolioBook, PositionPolicy, StepInput};

const INITIAL: f64 = 1_000_000.0;
const MIN: i64 = 60_000_000_000;

fn bar(ts: i64, close: f64) -> KernelBar {
    KernelBar {
        timestamp: ts,
        open: close,
        high: close + 0.5,
        low: close - 0.5,
        close,
        volume: 1_000.0,
    }
}

struct Rig {
    book: PortfolioBook,
    next_idx: Vec<usize>,
    trades: Vec<Trade>,
    events: Vec<EngineEvent>,
}

impl Rig {
    fn new(mode: AccountMode, decimals: u32) -> Self {
        let config = BacktestConfig {
            initial_capital: INITIAL,
            fees: 0.0,
            money_decimals: decimals,
            ..BacktestConfig::default()
        };
        Self {
            book: PortfolioBook::new(config, mode),
            next_idx: Vec::new(),
            trades: Vec::new(),
            events: Vec::new(),
        }
    }

    fn add(&mut self, symbol: &str, spec: InstrumentSpec) -> usize {
        self.next_idx.push(0);
        self.book.add_instrument(
            symbol.to_string(),
            Direction::Long,
            Some(spec),
            None,
            PositionPolicy::Net,
        )
    }

    fn order(&mut self, instrument: usize, ts: i64, buy: bool, units: f64) {
        let idx = self.next_idx[instrument];
        self.book.kernel_mut(instrument).submit_order_full(
            if buy { OrderSide::Buy } else { OrderSide::Sell },
            QtySpec::Units(units),
            OrderKind::Market,
            TimeInForce::Gtc,
            idx,
            ts,
            format!("o{instrument}-{idx}-{buy}"),
            None,
            None,
            false,
            false,
            None,
        );
    }

    fn step(&mut self, instrument: usize, b: KernelBar) -> Vec<EngineEvent> {
        let idx = self.next_idx[instrument];
        self.next_idx[instrument] += 1;
        let entry = ScheduleEntry { instrument, local_idx: idx, data: ScheduleData::Bar(b) };
        let events = self.book.apply(&entry, None, StepInput::default());
        for e in &events {
            if let EngineEvent::Exited { trade, .. } = e {
                self.trades.push((**trade).clone());
            }
        }
        self.events.extend(events.iter().cloned());
        events
    }

    fn realised(&self) -> i128 {
        self.trades.iter().map(|t| t.money.expect("booked").pnl).sum()
    }

    /// The integer identity; `Ok` when the account's books balance.
    fn reconcile(&self, margin: bool, initial_minor: i128) -> Result<(), String> {
        let (mut cost, mut fees) = (0i128, 0i128);
        for i in 0..self.next_idx.len() {
            let (c, f) = self.book.kernel(i).open_booked();
            cost += c.minor();
            fees += f.minor();
        }
        let open = if margin { fees } else { cost + fees };
        let expected = initial_minor + self.realised() + self.book.dividends_money().minor() - open;
        let got = self.book.pool().0.minor();
        if got == expected {
            Ok(())
        } else {
            Err(format!(
                "balance {got} != initial {initial_minor} + realised {} + dividends {} - open {open}",
                self.realised(),
                self.book.dividends_money().minor()
            ))
        }
    }
}

fn future(symbol: &str, expiry: i64) -> InstrumentSpec {
    InstrumentSpec {
        expiration_ns: Some(expiry),
        multiplier: 5.0,
        ..InstrumentSpec::new(symbol, InstrumentKind::Contract { underlying: Some("A".into()) })
    }
}

fn share(symbol: &str) -> InstrumentSpec {
    InstrumentSpec { size_increment: 1.0, ..InstrumentSpec::new(symbol, InstrumentKind::Cash) }
}

fn call(symbol: &str, strike: f64, expiry: i64) -> InstrumentSpec {
    InstrumentSpec {
        expiration_ns: Some(expiry),
        multiplier: 10.0,
        ..InstrumentSpec::new(
            symbol,
            InstrumentKind::Option {
                strike,
                right: OptionRight::Call,
                underlying: Some("A".into()),
                binary: false,
            },
        )
    }
}

fn settled(events: &[EngineEvent]) -> Vec<&EngineEvent> {
    events.iter().filter(|e| matches!(e, EngineEvent::Settled { .. })).collect()
}

// -- numbers ---------------------------------------------------------------------

#[test]
fn a_future_settles_at_the_declared_price_and_the_account_is_exact() {
    let mut rig = Rig::new(AccountMode::Cash, 2);
    let mut spec = future("F", 3 * MIN);
    spec.settlement.price = SettlementPrice::Fixed(120.0);
    let f = rig.add("F", spec);
    rig.order(f, 0, true, 10.0);
    rig.step(f, bar(0, 100.0));
    rig.step(f, bar(MIN, 101.0));
    let events = rig.step(f, bar(3 * MIN, 130.0));
    // 10 contracts x multiplier 5: bought at 100, settled at the declared 120
    let t = rig.trades.last().unwrap();
    assert_eq!((t.exit_reason, t.exit_price), (ExitReason::Settlement, 120.0));
    assert_eq!(t.money.unwrap().pnl, (120 - 100) * 10 * 5 * 100);
    match settled(&events)[..] {
        [EngineEvent::Settled { source, style, price, .. }] => {
            assert_eq!(
                (*source, *style, *price),
                (SettlementSource::Declared, SettlementStyle::Cash, 120.0)
            );
        }
        _ => panic!("expected one settlement, got {events:?}"),
    }
    rig.reconcile(false, 100_000_000).unwrap();
    assert_eq!(rig.book.pool().0.minor(), 100_000_000 + 20 * 10 * 5 * 100);
}

#[test]
fn a_call_exercised_delivers_the_underlying_at_the_strike() {
    let mut rig = Rig::new(AccountMode::Cash, 2);
    let a = rig.add("A", share("A"));
    let mut spec = call("O", 100.0, 3 * MIN);
    spec.settlement.style = SettlementStyle::Physical;
    let o = rig.add("O", spec);
    rig.order(o, 0, true, 2.0);
    rig.step(o, bar(0, 5.0));
    rig.step(a, bar(0, 100.0));
    rig.step(a, bar(MIN, 104.0));
    rig.step(a, bar(2 * MIN, 110.0));
    // the option has no more bars; the underlying's print at the expiry notices it
    let events = rig.step(a, bar(3 * MIN, 111.0));
    let ends = settled(&events);
    assert_eq!(ends.len(), 1);
    match ends[0] {
        EngineEvent::Settled { outcome, deliver, delivery_price, source, price, .. } => {
            assert_eq!((*outcome, *deliver, *delivery_price), ("exercised", 20.0, 100.0));
            // the contract waited for the underlying's print at the expiry instant
            assert_eq!((*source, *price), (SettlementSource::Underlying, 0.0));
        }
        _ => unreachable!(),
    }
    // 20 shares taken in at 100 (a purchase of 2000), option premium 2 x 10 x 5 lost
    assert_eq!(rig.book.kernel(a).open_count(), 1);
    rig.reconcile(false, 100_000_000).unwrap();
    assert_eq!(rig.book.pool().0.minor(), 100_000_000 - 2 * 10 * 5 * 100 - 20 * 100 * 100);
}

#[test]
fn a_split_rescales_the_position_and_pays_the_fraction_in_cash() {
    let mut rig = Rig::new(AccountMode::Cash, 2);
    let a = rig.add("A", share("A"));
    rig.book
        .set_corporate_actions(&[CorporateAction::split("A", 2 * MIN, 1.5)], AdjustmentPolicy::Raw)
        .unwrap();
    rig.order(a, 0, true, 3.0);
    rig.step(a, bar(0, 100.0));
    rig.step(a, bar(MIN, 100.0));
    let events = rig.step(a, bar(2 * MIN, 66.0));
    // 3 shares x 1.5 = 4.5: 4 kept, half a share paid at 100 / 1.5
    let pos = rig.book.kernel(a).position_snapshot().unwrap();
    assert_eq!(pos.size, 4.0);
    assert!((pos.entry_price - 100.0 / 1.5).abs() < 1e-12);
    let lieu: Vec<_> =
        rig.trades.iter().filter(|t| t.exit_reason == ExitReason::CorporateAction).collect();
    assert_eq!(lieu.len(), 1);
    assert!((lieu[0].size - 0.5).abs() < 1e-12 && (lieu[0].exit_price - 100.0 / 1.5).abs() < 1e-12);
    assert!(events
        .iter()
        .any(|e| matches!(e, EngineEvent::CorporateAction { ratio, .. } if *ratio == 1.5)));
    // 3 shares bought for 300.00; half a share sold for 33.33 in cash
    rig.reconcile(false, 100_000_000).unwrap();
    assert_eq!(rig.book.pool().0.minor(), 100_000_000 - 30_000 + 3_333);
    // the remaining position still carries the rest of the booked cost basis
    assert_eq!(rig.book.kernel(a).open_booked().0.minor(), 30_000 - 3_333);
}

#[test]
fn a_dividend_is_credited_on_its_pay_date_to_the_holder_at_the_ex_date() {
    let mut rig = Rig::new(AccountMode::Cash, 2);
    let a = rig.add("A", share("A"));
    rig.book
        .set_corporate_actions(
            &[CorporateAction::dividend("A", MIN, 3.0, Some(3 * MIN))],
            AdjustmentPolicy::Raw,
        )
        .unwrap();
    rig.order(a, 0, true, 10.0);
    rig.step(a, bar(0, 100.0));
    rig.step(a, bar(MIN, 100.0));
    rig.step(a, bar(2 * MIN, 100.0));
    assert_eq!(rig.book.dividends_received(), 0.0);
    // sold before the pay date: the holder at the ex-date is still paid
    rig.order(a, 2 * MIN, false, 10.0);
    rig.step(a, bar(3 * MIN, 100.0));
    assert_eq!(rig.book.dividends_received(), 30.0);
    rig.reconcile(false, 100_000_000).unwrap();
}

#[test]
fn an_order_on_an_expired_contract_is_rejected() {
    let mut rig = Rig::new(AccountMode::Cash, 2);
    let f = rig.add("F", future("F", 2 * MIN));
    rig.step(f, bar(0, 100.0));
    rig.step(f, bar(2 * MIN, 100.0));
    rig.order(f, 2 * MIN, true, 1.0);
    let events = rig.step(f, bar(3 * MIN, 100.0));
    assert!(events.iter().any(|e| matches!(
        e,
        EngineEvent::OrderRejected { reason, .. } if *reason == "expired"
    )));
}

// -- the reconciliation, over random runs ---------------------------------------

/// One bar's move, and a (buy?, units) order for each of the three instruments.
type Step = (f64, [Option<(bool, f64)>; 3]);

#[derive(Debug, Clone)]
struct Scenario {
    margin: Option<f64>,
    decimals: u32,
    steps: Vec<Step>,
    fut_expiry: usize,
    opt_expiry: usize,
    fut_ends: usize,
    opt_ends: usize,
    settle_fut: u8,
    physical: bool,
    actions: Vec<(usize, u8, f64)>,
    fee: f64,
}

fn order() -> impl Strategy<Value = Option<(bool, f64)>> {
    prop_oneof![
        3 => Just(None),
        2 => (any::<bool>(), 1.0f64..30.0).prop_map(Some),
    ]
}

fn scenario() -> impl Strategy<Value = Scenario> {
    (
        prop_oneof![Just(None), (1.5f64..6.0).prop_map(Some)],
        prop_oneof![Just(2u32), Just(0), Just(4)],
        prop::collection::vec((-0.05f64..0.05, [order(), order(), order()]), 8..30),
        3usize..10,
        3usize..10,
        prop::collection::vec(
            (
                0usize..30,
                0u8..4,
                prop_oneof![Just(2.0), Just(3.0), Just(0.5), Just(1.5), Just(4.0 / 3.0), Just(2.5)],
            ),
            0..4,
        ),
        (0usize..30, 0usize..30, 0u8..4, any::<bool>(), prop_oneof![Just(0.0), Just(0.0005)]),
    )
        .prop_map(
            |(margin, decimals, steps, fe, oe, actions, (fend, oend, sf, physical, fee))| {
                Scenario {
                    margin,
                    decimals,
                    fut_expiry: fe,
                    opt_expiry: oe,
                    fut_ends: fend,
                    opt_ends: oend,
                    steps,
                    settle_fut: sf,
                    physical,
                    actions,
                    fee,
                }
            },
        )
}

fn run(s: &Scenario) -> Result<Rig, String> {
    let mode = match s.margin {
        Some(leverage) => AccountMode::Margin { leverage },
        None => AccountMode::Cash,
    };
    let margin = s.margin.is_some();
    let mut rig = Rig::new(mode, s.decimals);
    let a = rig.add("A", share("A"));
    let mut fspec = future("F", s.fut_expiry as i64 * MIN);
    fspec.settlement_fee = s.fee;
    fspec.settlement.price = match s.settle_fut {
        0 => SettlementPrice::Auto,
        1 => SettlementPrice::LastPrice,
        2 => SettlementPrice::Underlying,
        _ => SettlementPrice::Fixed(97.5),
    };
    let f = rig.add("F", fspec);
    let mut ospec = call("O", 100.0, s.opt_expiry as i64 * MIN);
    if s.physical {
        ospec.settlement.style = SettlementStyle::Physical;
    }
    let o = rig.add("O", ospec);
    let actions: Vec<CorporateAction> = s
        .actions
        .iter()
        .map(|(at, kind, x)| {
            let ts = *at as i64 * MIN;
            match kind {
                0 => CorporateAction::split("A", ts, *x),
                1 => CorporateAction::bonus("A", ts, *x),
                _ => CorporateAction::dividend("A", ts, *x, Some(ts + (*at as i64 % 3) * MIN)),
            }
        })
        .collect();
    rig.book.set_corporate_actions(&actions, AdjustmentPolicy::Raw).map_err(|e| e.to_string())?;
    let initial_minor = rig.book.pool().0.minor();
    let mut price = 100.0f64;
    for (i, (ret, orders)) in s.steps.iter().enumerate() {
        let ts = i as i64 * MIN;
        price *= 1.0 + ret;
        for (inst, order) in [a, f, o].into_iter().zip(orders) {
            if let Some((buy, units)) = order {
                rig.order(inst, ts, *buy, units.floor().max(1.0));
            }
        }
        rig.step(a, bar(ts, price));
        if i < s.fut_ends {
            rig.step(f, bar(ts, price * 1.01));
        }
        if i < s.opt_ends {
            rig.step(o, bar(ts, (price - 95.0).max(1.0)));
        }
        rig.reconcile(margin, initial_minor).map_err(|e| format!("step {i}: {e}"))?;
    }
    Ok(rig)
}

proptest! {
    #![proptest_config(Config {
        cases: 600,
        failure_persistence: None,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(0xE17E),
        max_shrink_iters: 4096,
        ..Config::default()
    })]

    /// Across expiry, settlement, delivery, splits, bonuses and dividends the
    /// account's balance, the open cost basis and the booked P&L still agree
    /// to the minor unit.
    #[test]
    fn the_books_balance_across_the_whole_lifecycle(s in scenario()) {
        if let Err(e) = run(&s) {
            prop_assert!(false, "{e}");
        }
    }
}

/// The property proves nothing unless its cases settle, deliver, split and pay
/// dividends; count what a slice of them does.
#[test]
fn the_property_exercises_every_part_of_the_lifecycle() {
    use proptest::strategy::ValueTree;
    let mut runner = proptest::test_runner::TestRunner::new(Config {
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(7),
        ..Config::default()
    });
    let (mut settled_n, mut delivered, mut lieu, mut dividends, mut refused) = (0, 0, 0, 0, 0);
    for _ in 0..400 {
        let s = scenario().new_tree(&mut runner).unwrap().current();
        let rig = run(&s).unwrap();
        for e in &rig.events {
            match e {
                EngineEvent::Settled { deliver, .. } => {
                    settled_n += 1;
                    if *deliver != 0.0 {
                        delivered += 1;
                    }
                }
                EngineEvent::CorporateAction { kind, cash, .. } => {
                    if matches!(kind, raptorbt::instruments::CorporateKind::Dividend) {
                        dividends += 1;
                    } else if *cash != 0.0 {
                        lieu += 1;
                    }
                }
                EngineEvent::LifecycleRefused { .. } => refused += 1,
                _ => {}
            }
        }
    }
    eprintln!("COVERAGE settled {settled_n} delivered {delivered} lieu {lieu} dividends {dividends} refused {refused}");
    assert!(settled_n > 100 && delivered > 20 && lieu > 10 && dividends > 30);
}
