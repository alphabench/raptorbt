//! Property-based invariants for the simulation kernel and its order book.
//!
//! Every property drives `EngineKernel` through its public API with random
//! bar paths, signals and typed order sequences, then checks a statement that
//! must hold for any input:
//!
//! * accounting: equity equals cash plus the marked value of positions, bar
//!   by bar, and final equity equals initial capital plus the trades' PnL;
//! * a cash account never goes negative at 100% sizing;
//! * netting never flips a position inside one fill;
//! * fill timing: nothing fills on the bar that decided it, and resting
//!   orders match only from the bar after their submission;
//! * order lifecycles stay legal and agree with their fills;
//! * fills land on the price grid and limits are never crossed;
//! * identical inputs give bit-identical output.
//!
//! Case counts are fixed and the RNG seed is pinned, so a run is repeatable
//! and writes no regression files.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};

use raptorbt::accounts::AccountMode;
use raptorbt::core::types::{
    BacktestConfig, Direction, ExitReason, FillTiming, StopConfig, TargetConfig, Trade,
};
use raptorbt::execution::indian_costs::Segment;
use raptorbt::execution::orders::{
    OrderKind, OrderSide, OrderStatus, QtySpec, TimeInForce, TrailOffset,
};
use raptorbt::execution::{FeeModel, FillPrice, SlippageModel};
use raptorbt::instruments::{InstrumentKind, InstrumentSpec};
use raptorbt::portfolio::{EngineEvent, EngineKernel, KernelBar, StepInput};

const INITIAL: f64 = 100_000.0;
/// Relative tolerance for money comparisons: far above accumulated float
/// error on a few dozen fills (about 1e-14, see numerics.md), far below any
/// real accounting mistake.
const MONEY_TOL: f64 = 1e-10;

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

// ---------------------------------------------------------------------------
// Scenario generation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum FeeKind {
    None,
    Pct(f64),
    Fixed(f64),
    Indian(usize),
}

const SEGMENTS: [Segment; 8] = [
    Segment::EquityIntraday,
    Segment::EquityDelivery,
    Segment::FuturesNfo,
    Segment::OptionsNfo,
    Segment::FuturesMcx,
    Segment::OptionsMcx,
    Segment::FuturesCds,
    Segment::OptionsCds,
];

impl FeeKind {
    fn model(&self) -> FeeModel {
        match self {
            FeeKind::None => FeeModel::None,
            FeeKind::Pct(r) => FeeModel::Percentage(*r),
            FeeKind::Fixed(a) => FeeModel::Fixed(*a),
            FeeKind::Indian(i) => FeeModel::Indian { segment: SEGMENTS[*i] },
        }
    }
}

fn fee_kind() -> impl Strategy<Value = FeeKind> {
    prop_oneof![
        Just(FeeKind::None),
        (0.0001f64..0.02).prop_map(FeeKind::Pct),
        (0.0f64..100.0).prop_map(FeeKind::Fixed),
        (0usize..SEGMENTS.len()).prop_map(FeeKind::Indian),
    ]
}

/// Market definition and account shared by every scenario.
#[derive(Debug, Clone)]
struct Env {
    long: bool,
    leverage: Option<f64>,
    fee: FeeKind,
    slip: bool,
    /// Tick size; `0.0` for none.
    incr: f64,
    mult: f64,
    timing: FillTiming,
    stop_pct: Option<f64>,
    target_pct: Option<f64>,
}

fn env() -> impl Strategy<Value = Env> {
    (
        any::<bool>(),
        prop_oneof![Just(None), (1.5f64..10.0).prop_map(Some)],
        fee_kind(),
        any::<bool>(),
        prop_oneof![Just(0.0), Just(0.01), Just(0.05), Just(0.25), Just(0.5)],
        prop_oneof![Just(1.0), Just(1.0), Just(5.0), Just(50.0)],
        prop_oneof![Just(FillTiming::SameBarClose), Just(FillTiming::NextBarOpen)],
        prop_oneof![Just(None), (0.005f64..0.08).prop_map(Some)],
        prop_oneof![Just(None), (0.005f64..0.08).prop_map(Some)],
    )
        .prop_map(|(long, leverage, fee, slip, incr, mult, timing, stop_pct, target_pct)| {
            Env { long, leverage, fee, slip, incr, mult, timing, stop_pct, target_pct }
        })
}

impl Env {
    fn kernel(&self) -> EngineKernel {
        self.kernel_with(BacktestConfig::default())
    }

    fn kernel_with(&self, base: BacktestConfig) -> EngineKernel {
        let cfg = BacktestConfig {
            initial_capital: INITIAL,
            fill_timing: Some(self.timing),
            stop: self.stop_pct.map_or(StopConfig::None, |percent| StopConfig::Fixed { percent }),
            target: self
                .target_pct
                .map_or(TargetConfig::None, |percent| TargetConfig::Fixed { percent }),
            ..base
        };
        let slippage =
            if self.slip { SlippageModel::Percentage(0.002) } else { SlippageModel::None };
        let direction = if self.long { Direction::Long } else { Direction::Short };
        let mut spec = InstrumentSpec::new("T", InstrumentKind::Cash);
        spec.price_increment = self.incr;
        spec.multiplier = self.mult;
        let kernel = EngineKernel::new(
            cfg,
            self.fee.model(),
            slippage,
            FillPrice::for_timing(self.timing),
            "T".to_string(),
            direction,
            None,
        )
        .with_instrument(spec);
        match self.leverage {
            Some(leverage) => kernel.with_account_mode(AccountMode::Margin { leverage }),
            None => kernel,
        }
    }

    fn is_cash(&self) -> bool {
        self.leverage.is_none()
    }
}

/// One bar's raw draw: gap, return, upper wick, lower wick.
type RawBar = (f64, f64, f64, f64);

fn raw_bar() -> impl Strategy<Value = RawBar> {
    (-0.03f64..0.03, -0.08f64..0.08, 0.0f64..0.03, 0.0f64..0.03)
}

fn make_bars(raw: &[RawBar]) -> Vec<KernelBar> {
    let mut prev_close = 100.0;
    raw.iter()
        .enumerate()
        .map(|(i, &(gap, ret, up, dn))| {
            let open = prev_close * (1.0 + gap);
            let close = open * (1.0 + ret);
            let bar = KernelBar {
                timestamp: i as i64 * 60_000_000_000,
                open,
                high: open.max(close) * (1.0 + up),
                low: open.min(close) * (1.0 - dn),
                close,
                volume: 1_000.0,
            };
            prev_close = close;
            bar
        })
        .collect()
}

fn signal_bar() -> impl Strategy<Value = (RawBar, bool, bool)> {
    (raw_bar(), prop::bool::weighted(0.3), prop::bool::weighted(0.3))
}

// ---------------------------------------------------------------------------
// Signal-driven runs
// ---------------------------------------------------------------------------

struct SignalRun {
    bars: Vec<KernelBar>,
    entries: Vec<bool>,
    exits: Vec<bool>,
    /// Trades closed during the run, in event order.
    closed: Vec<Trade>,
    /// Trades force-closed at the end of data.
    finalized: Vec<Trade>,
    /// Kernel equity and cash after each bar's step.
    equity: Vec<f64>,
    cash: Vec<f64>,
    /// Kernel equity and cash after the end-of-data close.
    final_equity: f64,
    final_cash: f64,
}

fn run_signals(env: &Env, scn: &[(RawBar, bool, bool)]) -> SignalRun {
    let raw: Vec<RawBar> = scn.iter().map(|s| s.0).collect();
    let bars = make_bars(&raw);
    let entries: Vec<bool> = scn.iter().map(|s| s.1).collect();
    let exits: Vec<bool> = scn.iter().map(|s| s.2).collect();
    let mut kernel = env.kernel();
    let mut closed = Vec::new();
    let mut equity = Vec::new();
    let mut cash = Vec::new();
    for (i, bar) in bars.iter().enumerate() {
        let input = StepInput { entry: entries[i], exit: exits[i], ..StepInput::default() };
        for event in kernel.step(i, bar, input) {
            if let EngineEvent::Exited { trade, .. } = event {
                closed.push(*trade);
            }
        }
        equity.push(kernel.equity(bar.close));
        cash.push(kernel.cash());
    }
    let last = bars.len() - 1;
    let finalized = kernel.finalize_all(last, &bars[last]);
    SignalRun {
        final_equity: kernel.equity(bars[last].close),
        final_cash: kernel.cash(),
        bars,
        entries,
        exits,
        closed,
        finalized,
        equity,
        cash,
    }
}

fn dir_sign(d: Direction) -> f64 {
    match d {
        Direction::Long => 1.0,
        Direction::Short => -1.0,
    }
}

/// What booking leaves between kernel equity and the independent model while a
/// position is open: the position's cost was debited rounded to the paisa
/// (half a paisa at most; up to three half-paisas for a short, whose cost
/// enters its marked value as collateral and again as gain, and whose booked
/// close rounds the exit too), the model marks it at the exact price. Closed
/// trades carry no such gap, because their P&L is the booked P&L.
const BOOK_GRAIN: f64 = 0.015;

/// `open` is the number of trade records that can be open at once (a position
/// closed in slices books each slice's cost separately).
fn near_booked(a: f64, b: f64, scale: f64, open: usize) -> bool {
    (a - b).abs() <= BOOK_GRAIN * open.max(1) as f64 + MONEY_TOL * scale.max(1.0)
}

fn near(a: f64, b: f64, scale: f64) -> bool {
    (a - b).abs() <= MONEY_TOL * scale.max(1.0)
}

/// Independent model of equity: capital, plus the PnL of every closed trade,
/// plus each open position's unrealised gain less the entry costs it has
/// already paid. Built only from trade records, never from kernel state.
fn model_equity(
    closed: &[Trade],
    finalized: &[Trade],
    bars: &[KernelBar],
    mult: f64,
    i: usize,
) -> f64 {
    let mut eq = INITIAL;
    for t in closed {
        if t.exit_idx <= i {
            eq += t.pnl;
        } else if t.entry_idx <= i {
            eq += open_value(t, bars[i].close, mult);
        }
    }
    for t in finalized {
        if t.entry_idx <= i {
            eq += open_value(t, bars[i].close, mult);
        }
    }
    eq
}

fn open_value(t: &Trade, close: f64, mult: f64) -> f64 {
    dir_sign(t.direction) * (close - t.entry_price) * t.size * mult - t.entry_fees
}

proptest! {
    #![proptest_config(config(4000, 0x1001))]

    /// Equity is the independent model of cash plus marked positions on every
    /// bar, and the end state is initial capital plus the sum of trade PnL.
    #[test]
    fn accounting_identity_holds_on_every_bar(
        env in env(),
        scn in prop::collection::vec(signal_bar(), 4..40),
    ) {
        let run = run_signals(&env, &scn);
        for i in 0..run.bars.len() {
            let want = model_equity(&run.closed, &run.finalized, &run.bars, env.mult, i);
            prop_assert!(
                near_booked(run.equity[i], want, INITIAL, run.closed.len() + run.finalized.len()),
                "bar {}: kernel equity {} vs model {} (env {:?})",
                i, run.equity[i], want, env
            );
        }
        let total: f64 = run.closed.iter().chain(&run.finalized).map(|t| t.pnl).sum();
        prop_assert!(
            near(run.final_equity, INITIAL + total, INITIAL),
            "final equity {} vs initial + sum(pnl) {}", run.final_equity, INITIAL + total
        );
        // Flat at the end, so equity is cash alone.
        prop_assert!(near(run.final_cash, run.final_equity, INITIAL));
        // Fees are booked once and in full on every trade.
        for t in run.closed.iter().chain(&run.finalized) {
            prop_assert!(near(t.fees, t.entry_fees + t.exit_fees, t.fees));
        }
    }

    /// A fully funded account never spends money it does not have, whatever
    /// the fee model, direction, tick size or multiplier.
    #[test]
    fn cash_account_never_goes_negative(
        mut env in env(),
        scn in prop::collection::vec(signal_bar(), 4..40),
    ) {
        env.leverage = None;
        let run = run_signals(&env, &scn);
        for (i, c) in run.cash.iter().enumerate() {
            prop_assert!(*c >= -MONEY_TOL * INITIAL, "bar {}: cash {} (env {:?})", i, c, env);
        }
        prop_assert!(run.final_cash >= -MONEY_TOL * INITIAL);
    }

    /// Under `next_bar_open` a signal never trades on the bar that raised it:
    /// each entry or exit follows a signal on the previous bar, at that bar's
    /// open when nothing perturbs the price.
    #[test]
    fn next_bar_open_never_fills_on_the_decision_bar(
        mut env in env(),
        scn in prop::collection::vec(signal_bar(), 4..40),
    ) {
        env.timing = FillTiming::NextBarOpen;
        let run = run_signals(&env, &scn);
        let exact = !env.slip && env.incr == 0.0;
        for t in run.closed.iter().chain(&run.finalized) {
            prop_assert!(t.entry_idx >= 1, "entry on the first bar: {:?}", t);
            prop_assert!(run.entries[t.entry_idx - 1], "entry {} without a prior signal", t.entry_idx);
            prop_assert_eq!(t.entry_time, run.bars[t.entry_idx].timestamp);
            if exact {
                prop_assert_eq!(t.entry_price, run.bars[t.entry_idx].open);
            }
            if t.exit_reason == ExitReason::Signal {
                prop_assert!(t.exit_idx >= 1);
                prop_assert!(run.exits[t.exit_idx - 1], "exit {} without a prior signal", t.exit_idx);
                if exact {
                    prop_assert_eq!(t.exit_price, run.bars[t.exit_idx].open);
                }
            }
        }
    }

    /// With a tick size every executed price is a whole number of ticks.
    /// End-of-data closes are valuation marks at the last close, not fills.
    #[test]
    fn signal_fills_land_on_the_price_grid(
        mut env in env(),
        scn in prop::collection::vec(signal_bar(), 4..40),
    ) {
        prop_assume!(env.incr > 0.0);
        env.target_pct = None;
        let run = run_signals(&env, &scn);
        for t in &run.closed {
            prop_assert!(on_grid(t.entry_price, env.incr), "entry {} off grid {}", t.entry_price, env.incr);
            if t.exit_reason == ExitReason::Signal {
                prop_assert!(on_grid(t.exit_price, env.incr), "exit {} off grid {}", t.exit_price, env.incr);
            }
        }
        for t in &run.finalized {
            prop_assert!(on_grid(t.entry_price, env.incr));
        }
    }
}

fn on_grid(price: f64, incr: f64) -> bool {
    let ticks = price / incr;
    (ticks - ticks.round()).abs() <= 1e-9 * ticks.round().abs().max(1.0)
}

// ---------------------------------------------------------------------------
// Typed order sequences
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum KindSpec {
    Market,
    Limit(f64),
    StopMarket(f64),
    StopLimit(f64, f64),
    MarketIfTouched(f64),
    LimitIfTouched(f64, f64),
    MarketToLimit,
    TrailingStop(f64),
    TrailingStopLimit(f64, f64),
}

#[derive(Debug, Clone)]
enum QtyKind {
    Units(f64),
    Frac(f64),
    Full,
}

#[derive(Debug, Clone)]
enum TifKind {
    Gtc,
    Day,
    Gtd(i64),
    Ioc,
    Fok,
    AtOpen,
    AtClose,
}

#[derive(Debug, Clone)]
enum Cmd {
    Submit {
        buy: bool,
        qty: QtyKind,
        kind: KindSpec,
        tif: TifKind,
        reduce_only: bool,
        post_only: bool,
        attach_stop: Option<f64>,
        attach_target: Option<f64>,
    },
    Bracket {
        buy: bool,
        units: f64,
        entry_limit: Option<f64>,
        stop: f64,
        target: f64,
    },
    Cancel(usize),
    CancelAll,
    Modify {
        target: usize,
        units: Option<f64>,
        limit: Option<f64>,
        trigger: Option<f64>,
    },
    Oco(usize, usize),
}

/// Offsets are fractions of the bar close at submission time.
fn offset() -> impl Strategy<Value = f64> {
    -0.06f64..0.06
}

fn kind_spec(limit_if_touched: bool) -> BoxedStrategy<KindSpec> {
    let lit_weight = if limit_if_touched { 1 } else { 0 };
    prop_oneof![
        4 => Just(KindSpec::Market),
        4 => offset().prop_map(KindSpec::Limit),
        2 => offset().prop_map(KindSpec::StopMarket),
        2 => (offset(), offset()).prop_map(|(a, b)| KindSpec::StopLimit(a, b)),
        1 => offset().prop_map(KindSpec::MarketIfTouched),
        lit_weight => (offset(), offset()).prop_map(|(a, b)| KindSpec::LimitIfTouched(a, b)),
        1 => Just(KindSpec::MarketToLimit),
        1 => (0.005f64..0.05).prop_map(KindSpec::TrailingStop),
        1 => (0.005f64..0.05, 0.0f64..0.02).prop_map(|(a, b)| KindSpec::TrailingStopLimit(a, b)),
    ]
    .boxed()
}

fn qty_kind() -> impl Strategy<Value = QtyKind> {
    prop_oneof![
        4 => (1.0f64..200.0).prop_map(|u| QtyKind::Units(u.round())),
        3 => (0.1f64..1.0).prop_map(QtyKind::Frac),
        2 => Just(QtyKind::Full),
    ]
}

fn tif_kind() -> impl Strategy<Value = TifKind> {
    prop_oneof![
        6 => Just(TifKind::Gtc),
        2 => Just(TifKind::Day),
        2 => (1i64..12).prop_map(|k| TifKind::Gtd(k * 60_000_000_000)),
        1 => Just(TifKind::Ioc),
        1 => Just(TifKind::Fok),
        1 => Just(TifKind::AtOpen),
        1 => Just(TifKind::AtClose),
    ]
}

/// Submissions, cancels and modifications; with `contingent`, also brackets
/// and one-cancels-other links.
fn cmd(contingent: bool, limit_if_touched: bool) -> impl Strategy<Value = Cmd> {
    let submit = (
        any::<bool>(),
        qty_kind(),
        kind_spec(limit_if_touched),
        tif_kind(),
        prop::bool::weighted(0.3),
        prop::bool::weighted(0.15),
        prop::option::weighted(0.15, offset()),
        prop::option::weighted(0.15, offset()),
    )
        .prop_map(
            |(buy, qty, kind, tif, reduce_only, post_only, attach_stop, attach_target)| {
                Cmd::Submit {
                    buy,
                    qty,
                    kind,
                    tif,
                    reduce_only,
                    post_only,
                    attach_stop,
                    attach_target,
                }
            },
        );
    let bracket = (
        any::<bool>(),
        (1.0f64..100.0).prop_map(f64::round),
        prop::option::of(offset()),
        0.005f64..0.06,
        0.005f64..0.06,
    )
        .prop_map(|(buy, units, entry_limit, stop, target)| Cmd::Bracket {
            buy,
            units,
            entry_limit,
            stop,
            target,
        });
    let modify = (
        0usize..64,
        prop::option::of((1.0f64..100.0).prop_map(f64::round)),
        prop::option::of(offset()),
        prop::option::of(offset()),
    )
        .prop_map(|(target, units, limit, trigger)| Cmd::Modify {
            target,
            units,
            limit,
            trigger,
        });
    let oco = (0usize..64, 0usize..64).prop_map(|(a, b)| Cmd::Oco(a, b));
    let cancel = (0usize..64).prop_map(Cmd::Cancel);
    if contingent {
        prop_oneof![
            8 => submit,
            3 => bracket,
            2 => cancel,
            1 => Just(Cmd::CancelAll),
            2 => modify,
            2 => oco,
        ]
        .boxed()
    } else {
        prop_oneof![
            8 => submit,
            2 => cancel,
            1 => Just(Cmd::CancelAll),
            2 => modify,
        ]
        .boxed()
    }
}

fn order_scenario(
    contingent: bool,
    limit_if_touched: bool,
) -> impl Strategy<Value = Vec<(RawBar, Vec<Cmd>)>> {
    prop::collection::vec(
        (raw_bar(), prop::collection::vec(cmd(contingent, limit_if_touched), 0..3)),
        4..30,
    )
}

#[derive(Debug, Clone, Copy)]
struct Fill {
    idx: usize,
    order_id: u64,
    price: f64,
    size: f64,
}

struct OrderRun {
    bars: Vec<KernelBar>,
    kernel: EngineKernel,
    fills: Vec<Fill>,
    closed: Vec<Trade>,
    finalized: Vec<Trade>,
    equity: Vec<f64>,
    cash: Vec<f64>,
    /// Per fill: (signed position before, after, whether the consequence
    /// events held both an exit and an entry).
    groups: Vec<(Fill, f64, f64, bool, bool)>,
    /// Signed position implied by events after each step, and what the
    /// kernel reports.
    tracked: Vec<(f64, f64)>,
    /// Bars during which the open position had been added to: a trade record keeps one
    /// average entry price for the whole position, so the trade-built model cannot mark it
    /// before it closes (netting adds, covered by the unit tests).
    grown: Vec<bool>,
}

fn run_orders(env: &Env, base: BacktestConfig, scn: &[(RawBar, Vec<Cmd>)]) -> OrderRun {
    let raw: Vec<RawBar> = scn.iter().map(|s| s.0).collect();
    let bars = make_bars(&raw);
    let mut kernel = env.kernel_with(base);
    let mut ids: Vec<u64> = Vec::new();
    let mut fills = Vec::new();
    let mut closed = Vec::new();
    let mut groups = Vec::new();
    let mut equity = Vec::new();
    let mut cash = Vec::new();
    let mut tracked = Vec::new();
    let mut grown = vec![false; bars.len()];
    let (mut start, mut grew) = (0usize, false);
    let mut signed = 0.0f64;

    let price_at = |close: f64, off: f64| -> f64 {
        let p = close * (1.0 + off);
        let p = if env.incr > 0.0 {
            (p / env.incr).round() * env.incr
        } else {
            (p * 100.0).round() / 100.0
        };
        p.max(if env.incr > 0.0 { env.incr } else { 0.01 })
    };

    for (i, bar) in bars.iter().enumerate() {
        let ts = bar.timestamp;
        for c in &scn[i].1 {
            match c {
                Cmd::Submit {
                    buy,
                    qty,
                    kind,
                    tif,
                    reduce_only,
                    post_only,
                    attach_stop,
                    attach_target,
                } => {
                    let side = if *buy { OrderSide::Buy } else { OrderSide::Sell };
                    let qty = match qty {
                        QtyKind::Units(u) => QtySpec::Units(*u),
                        QtyKind::Frac(f) => QtySpec::CapitalFrac(*f),
                        QtyKind::Full => QtySpec::FullPosition,
                    };
                    let p = |off: f64| price_at(bar.close, off);
                    let kind = match kind {
                        KindSpec::Market => OrderKind::Market,
                        KindSpec::Limit(o) => OrderKind::Limit { price: p(*o) },
                        KindSpec::StopMarket(o) => OrderKind::StopMarket { trigger: p(*o) },
                        KindSpec::StopLimit(a, b) => {
                            OrderKind::StopLimit { trigger: p(*a), price: p(*b) }
                        }
                        KindSpec::MarketIfTouched(o) => {
                            OrderKind::MarketIfTouched { trigger: p(*o) }
                        }
                        KindSpec::LimitIfTouched(a, b) => {
                            OrderKind::LimitIfTouched { trigger: p(*a), price: p(*b) }
                        }
                        KindSpec::MarketToLimit => OrderKind::MarketToLimit,
                        KindSpec::TrailingStop(f) => OrderKind::TrailingStopMarket {
                            offset: TrailOffset::Price(bar.close * f),
                        },
                        KindSpec::TrailingStopLimit(f, l) => OrderKind::TrailingStopLimit {
                            offset: TrailOffset::Price(bar.close * f),
                            limit_offset: bar.close * l,
                        },
                    };
                    let tif = match tif {
                        TifKind::Gtc => TimeInForce::Gtc,
                        TifKind::Day => TimeInForce::Day,
                        TifKind::Gtd(d) => TimeInForce::Gtd { expire_ns: ts + d },
                        TifKind::Ioc => TimeInForce::Ioc,
                        TifKind::Fok => TimeInForce::Fok,
                        TifKind::AtOpen => TimeInForce::AtOpen,
                        TifKind::AtClose => TimeInForce::AtClose,
                    };
                    // AtOpen/AtClose apply to market orders only.
                    let tif = if matches!(kind, OrderKind::Market)
                        || !matches!(tif, TimeInForce::AtOpen | TimeInForce::AtClose)
                    {
                        tif
                    } else {
                        TimeInForce::Gtc
                    };
                    let post_only = *post_only && matches!(kind, OrderKind::Limit { .. });
                    let id = kernel.submit_order_full(
                        side,
                        qty,
                        kind,
                        tif,
                        i,
                        ts,
                        format!("o{}", ids.len()),
                        attach_stop.map(&p),
                        attach_target.map(&p),
                        post_only,
                        *reduce_only,
                        None,
                    );
                    ids.push(id);
                }
                Cmd::Bracket { buy, units, entry_limit, stop, target } => {
                    let side = if *buy { OrderSide::Buy } else { OrderSide::Sell };
                    let exit_side = side.flip();
                    let sign = if *buy { 1.0 } else { -1.0 };
                    let entry_kind = match entry_limit {
                        Some(o) => OrderKind::Limit { price: price_at(bar.close, *o) },
                        None => OrderKind::Market,
                    };
                    let entry = kernel.submit_order_full(
                        side,
                        QtySpec::Units(*units),
                        entry_kind,
                        TimeInForce::Gtc,
                        i,
                        ts,
                        format!("o{}", ids.len()),
                        None,
                        None,
                        false,
                        false,
                        None,
                    );
                    ids.push(entry);
                    let stop_id = kernel.submit_order_full(
                        exit_side,
                        QtySpec::FullPosition,
                        OrderKind::StopMarket { trigger: price_at(bar.close, -sign * stop) },
                        TimeInForce::Gtc,
                        i,
                        ts,
                        format!("o{}", ids.len()),
                        None,
                        None,
                        false,
                        true,
                        Some(entry),
                    );
                    ids.push(stop_id);
                    let target_id = kernel.submit_order_full(
                        exit_side,
                        QtySpec::FullPosition,
                        OrderKind::Limit { price: price_at(bar.close, sign * target) },
                        TimeInForce::Gtc,
                        i,
                        ts,
                        format!("o{}", ids.len()),
                        None,
                        None,
                        false,
                        true,
                        Some(entry),
                    );
                    ids.push(target_id);
                    kernel.link_oco(&[stop_id, target_id]);
                }
                Cmd::Cancel(k) => {
                    if !ids.is_empty() {
                        kernel.cancel_order(i, ids[k % ids.len()]);
                    }
                }
                Cmd::CancelAll => {
                    kernel.cancel_all_orders(i);
                }
                Cmd::Modify { target, units, limit, trigger } => {
                    if !ids.is_empty() {
                        kernel.modify_order(
                            ids[target % ids.len()],
                            units.map(QtySpec::Units),
                            limit.map(|o| price_at(bar.close, o)),
                            trigger.map(|o| price_at(bar.close, o)),
                        );
                    }
                }
                Cmd::Oco(a, b) => {
                    if !ids.is_empty() {
                        let (a, b) = (ids[a % ids.len()], ids[b % ids.len()]);
                        if a != b {
                            kernel.link_oco(&[a, b]);
                        }
                    }
                }
            }
        }

        let events = kernel.step(i, bar, StepInput::default());
        let mut j = 0;
        while j < events.len() {
            match &events[j] {
                EngineEvent::OrderFilled { idx, order_id, price, size, .. } => {
                    let fill = Fill { idx: *idx, order_id: *order_id, price: *price, size: *size };
                    fills.push(fill);
                    let before = signed;
                    let (mut exited, mut entered) = (false, false);
                    let mut k = j + 1;
                    while k < events.len() {
                        match &events[k] {
                            EngineEvent::Entered { size, direction, .. } => {
                                if signed != 0.0 && signed.signum() == dir_sign(*direction).signum()
                                {
                                    grew = true;
                                } else if signed == 0.0 {
                                    start = i;
                                }
                                signed += dir_sign(*direction) * size;
                                entered = true;
                            }
                            // The position's own stop or target, hit on the bar its
                            // entry filled at the open of, is a separate event of that
                            // bar, not a consequence of the fill (the outer loop books it).
                            EngineEvent::Exited { trade, .. }
                                if entered
                                    && matches!(
                                        trade.exit_reason,
                                        ExitReason::StopLoss
                                            | ExitReason::TakeProfit
                                            | ExitReason::TrailingStop
                                    ) =>
                            {
                                break
                            }
                            EngineEvent::Exited { trade, .. } => {
                                signed -= dir_sign(trade.direction) * trade.size;
                                if signed.abs() < 1e-12 && grew {
                                    grown[start..=i].iter_mut().for_each(|g| *g = true);
                                    grew = false;
                                }
                                exited = true;
                                closed.push((**trade).clone());
                            }
                            _ => break,
                        }
                        k += 1;
                    }
                    groups.push((fill, before, signed, exited && entered, exited));
                    j = k;
                    continue;
                }
                EngineEvent::Entered { size, direction, .. } => {
                    if signed != 0.0 && signed.signum() == dir_sign(*direction).signum() {
                        grew = true;
                    } else if signed == 0.0 {
                        start = i;
                    }
                    signed += dir_sign(*direction) * size;
                }
                EngineEvent::Exited { trade, .. } => {
                    signed -= dir_sign(trade.direction) * trade.size;
                    if signed.abs() < 1e-12 && grew {
                        grown[start..=i].iter_mut().for_each(|g| *g = true);
                        grew = false;
                    }
                    closed.push((**trade).clone());
                }
                _ => {}
            }
            j += 1;
        }
        equity.push(kernel.equity(bar.close));
        cash.push(kernel.cash());
        let held: f64 =
            kernel.position_snapshots().iter().map(|p| dir_sign(p.direction) * p.size).sum();
        tracked.push((signed, held));
        if grew {
            grown[start..=i].iter_mut().for_each(|g| *g = true);
        }
        if signed.abs() < 1e-12 {
            grew = false;
        }
    }
    let last = bars.len() - 1;
    let finalized = kernel.finalize_all(last, &bars[last]);
    OrderRun { bars, kernel, fills, closed, finalized, equity, cash, groups, tracked, grown }
}

fn orders_env() -> impl Strategy<Value = Env> {
    env().prop_map(|mut e| {
        // Stops and targets from config would add exits that are not under
        // test here; attached levels on orders still exercise them.
        e.stop_pct = None;
        e.target_pct = None;
        e
    })
}

type Scenario = Vec<(RawBar, Vec<Cmd>)>;

fn check_netting(env: &Env, scn: &Scenario) -> Result<(), TestCaseError> {
    let run = run_orders(env, BacktestConfig::default(), scn);
    for (fill, before, after, both, _) in &run.groups {
        prop_assert!(!both, "fill {:?} exited and entered in one go", fill);
        let flipped = *before * *after < 0.0 && before.abs() > 1e-9 && after.abs() > 1e-9;
        prop_assert!(!flipped, "fill {:?} flipped position {} -> {}", fill, before, after);
        let order = run.kernel.order(fill.order_id).expect("filled order exists");
        if order.reduce_only {
            prop_assert!(
                after.abs() <= before.abs() + 1e-9,
                "reduce_only fill {:?} grew exposure {} -> {}",
                fill,
                before,
                after
            );
        }
    }
    // The event stream and the kernel agree on what is held.
    for (i, (tracked, held)) in run.tracked.iter().enumerate() {
        prop_assert!(
            (tracked - held).abs() <= 1e-6 * held.abs().max(1.0),
            "bar {}: events imply {} but kernel holds {}",
            i,
            tracked,
            held
        );
    }
    Ok(())
}

/// Orders only ever move through legal states (an illegal transition panics,
/// which fails the case), end in a state consistent with their fills, and
/// never fill past what was requested.
fn check_lifecycles(env: &Env, scn: &Scenario) -> Result<(), TestCaseError> {
    let run = run_orders(env, BacktestConfig::default(), scn);
    for order in run.kernel.order_book() {
        let filled_events: f64 =
            run.fills.iter().filter(|f| f.order_id == order.id).map(|f| f.size).sum();
        prop_assert!(
            (order.filled_qty - filled_events).abs() <= 1e-9 * filled_events.max(1.0),
            "order {} filled_qty {} vs events {}",
            order.id,
            order.filled_qty,
            filled_events
        );
        // An exit larger than the position fills what is held and completes;
        // the surplus is dropped by design (numerics: "never reverses").
        let closed_position =
            run.groups.iter().any(|(fill, _, _, _, exited)| fill.order_id == order.id && *exited);
        if let QtySpec::Units(requested) = order.qty {
            prop_assert!(
                order.filled_qty <= requested + 1e-9,
                "order {} overfilled: {} of {}",
                order.id,
                order.filled_qty,
                requested
            );
            let complete = order.filled_qty + 1e-9 >= requested;
            if complete {
                prop_assert_eq!(order.status, OrderStatus::Filled, "order {}", order.id);
            }
            if order.status == OrderStatus::Filled && !closed_position {
                prop_assert!(
                    complete,
                    "order {} is Filled with {} of {}",
                    order.id,
                    order.filled_qty,
                    requested
                );
            }
        }
        prop_assert_eq!(
            order.status == OrderStatus::Filled,
            order.filled_qty > 0.0,
            "order {} status {:?} with {} filled (no partial fills configured)",
            order.id,
            order.status,
            order.filled_qty
        );
        if !order.status.is_terminal() {
            prop_assert!(order.filled_qty == 0.0, "working order {} has fills", order.id);
        }
    }
    Ok(())
}

/// Resting orders match from the bar after their submission. Under
/// `next_bar_open` nothing fills on its submission bar at all.
fn check_submission_bar(env: &Env, scn: &Scenario) -> Result<(), TestCaseError> {
    let run = run_orders(env, BacktestConfig::default(), scn);
    for fill in &run.fills {
        let order = run.kernel.order(fill.order_id).expect("order exists");
        let resting = !matches!(order.kind, OrderKind::Market)
            || matches!(order.tif, TimeInForce::AtOpen | TimeInForce::AtClose);
        // The one exception: a plain limit strictly through the decision bar's
        // close trades there under same_bar_close, like a market order placed
        // on that bar. A limit at the price itself rests.
        let marketable_on_decision = env.timing == FillTiming::SameBarClose
            && fill.idx == order.submitted_idx
            && order.parent_id.is_none()
            && !order.post_only
            && !matches!(order.tif, TimeInForce::AtOpen | TimeInForce::AtClose)
            && match (order.kind, order.side) {
                (OrderKind::Limit { price }, OrderSide::Buy) => price > run.bars[fill.idx].close,
                (OrderKind::Limit { price }, OrderSide::Sell) => price < run.bars[fill.idx].close,
                _ => false,
            };
        if marketable_on_decision {
            continue;
        }
        if resting || env.timing == FillTiming::NextBarOpen {
            prop_assert!(
                fill.idx > order.submitted_idx,
                "{:?} order {} submitted on bar {} filled on bar {}",
                order.kind,
                order.id,
                order.submitted_idx,
                fill.idx
            );
        } else {
            prop_assert_eq!(fill.idx, order.submitted_idx);
        }
    }
    Ok(())
}

/// With a tick size every order fill sits on the grid, and a limit is never
/// filled through its price.
fn check_grid_and_limits(env: &Env, scn: &Scenario) -> Result<(), TestCaseError> {
    let run = run_orders(env, BacktestConfig::default(), scn);
    for fill in &run.fills {
        let order = run.kernel.order(fill.order_id).expect("order exists");
        if env.incr > 0.0 {
            prop_assert!(
                on_grid(fill.price, env.incr),
                "{:?} fill at {} off the {} grid",
                order.kind,
                fill.price,
                env.incr
            );
        }
        if let Some(limit) = order.kind.limit_price() {
            let slack = 1e-9 * limit.abs();
            match order.side {
                OrderSide::Buy => prop_assert!(
                    fill.price <= limit + slack,
                    "buy {:?} filled at {}",
                    order.kind,
                    fill.price
                ),
                OrderSide::Sell => prop_assert!(
                    fill.price >= limit - slack,
                    "sell {:?} filled at {}",
                    order.kind,
                    fill.price
                ),
            }
        }
    }
    Ok(())
}

/// The accounting identity and the cash floor also hold on the order path.
fn check_order_path_accounting(env: &Env, scn: &Scenario) -> Result<(), TestCaseError> {
    let run = run_orders(env, BacktestConfig::default(), scn);
    for i in 0..run.bars.len() {
        if run.grown[i] {
            continue;
        }
        let want = model_equity(&run.closed, &run.finalized, &run.bars, env.mult, i);
        prop_assert!(
            near_booked(run.equity[i], want, INITIAL, run.closed.len() + run.finalized.len()),
            "bar {}: kernel equity {} vs model {}",
            i,
            run.equity[i],
            want
        );
    }
    // Whatever the path, once everything is closed the account is capital plus the PnL of
    // every trade (adds, reductions and flips included).
    let pnl: f64 = run.closed.iter().chain(run.finalized.iter()).map(|t| t.pnl).sum();
    prop_assert!(
        near_booked(
            run.kernel.cash(),
            INITIAL + pnl,
            INITIAL,
            run.closed.len() + run.finalized.len()
        ),
        "end: cash {} vs capital plus trade PnL {}",
        run.kernel.cash(),
        INITIAL + pnl
    );
    if env.is_cash() {
        for c in &run.cash {
            prop_assert!(*c >= -MONEY_TOL * INITIAL, "cash {}", c);
        }
    }
    Ok(())
}

/// A reduce-only order trades against the position it reduces: a sell
/// against a long, a buy against a short.
fn check_reduce_only_opposes(env: &Env, scn: &Scenario) -> Result<(), TestCaseError> {
    let run = run_orders(env, BacktestConfig::default(), scn);
    for (fill, before, after, _, _) in &run.groups {
        let order = run.kernel.order(fill.order_id).expect("filled order exists");
        if order.reduce_only {
            let reduces = match order.side {
                OrderSide::Sell => *before > 0.0,
                OrderSide::Buy => *before < 0.0,
            };
            prop_assert!(
                reduces,
                "reduce_only {:?} fill {:?} moved position {} -> {}",
                order.side,
                fill,
                before,
                after
            );
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(config(2500, 0x2002))]

    /// Under the netting policy one fill can close a position or open one
    /// from flat, never both: no order takes a long to short or back.
    #[test]
    fn netting_never_flips_a_position_in_one_fill(
        env in orders_env(),
        scn in order_scenario(true, true),
    ) {
        check_netting(&env, &scn)?;
    }

    #[test]
    fn order_lifecycles_stay_legal_and_match_their_fills(
        env in orders_env(),
        scn in order_scenario(true, true),
    ) {
        check_lifecycles(&env, &scn)?;
    }

    #[test]
    fn orders_never_fill_on_their_submission_bar(
        env in orders_env(),
        scn in order_scenario(true, true),
    ) {
        check_submission_bar(&env, &scn)?;
    }

    #[test]
    fn order_fills_sit_on_the_grid_and_respect_limits(
        env in orders_env(),
        scn in order_scenario(true, true),
    ) {
        check_grid_and_limits(&env, &scn)?;
    }

    #[test]
    fn order_path_accounting_identity(
        env in orders_env(),
        scn in order_scenario(true, true),
    ) {
        check_order_path_accounting(&env, &scn)?;
    }
}

// ---------------------------------------------------------------------------
// Order edge cases, each with a minimal reproduction below
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(config(200, 0x4004))]

    /// A reduce-only order that would add to the position is refused.
    #[test]
    fn reduce_only_orders_only_oppose_the_position(
        env in orders_env(),
        scn in order_scenario(true, true),
    ) {
        check_reduce_only_opposes(&env, &scn)?;
    }
}

fn flat_bars(n: usize) -> Vec<(RawBar, Vec<Cmd>)> {
    (0..n).map(|_| ((0.0, 0.0, 0.0, 0.0), Vec::new())).collect()
}

fn submit(buy: bool, qty: f64, kind: KindSpec, reduce_only: bool) -> Cmd {
    Cmd::Submit {
        buy,
        qty: QtyKind::Units(qty),
        kind,
        tif: TifKind::Gtc,
        reduce_only,
        post_only: false,
        attach_stop: None,
        attach_target: None,
    }
}

fn plain_env() -> Env {
    Env {
        long: true,
        leverage: None,
        fee: FeeKind::None,
        slip: false,
        incr: 0.0,
        mult: 1.0,
        timing: FillTiming::SameBarClose,
        stop_pct: None,
        target_pct: None,
    }
}

/// Minimal reproduction: a bracket whose stop and target are both inside the
/// bar after the entry.
#[test]
fn bracket_with_both_legs_touched_in_one_bar() {
    let mut scn = flat_bars(2);
    scn[0].1.push(Cmd::Bracket {
        buy: true,
        units: 1.0,
        entry_limit: None,
        stop: 0.005,
        target: 0.005,
    });
    scn[1].0 = (0.0, 0.0, 0.02, 0.02); // high 102, low 98: stop 99.5 and target 100.5 both inside
    let run = run_orders(&plain_env(), BacktestConfig::default(), &scn);
    for order in run.kernel.order_book() {
        assert!(order.status.is_terminal() || order.filled_qty == 0.0);
    }
}

/// With no position a reduce-only order has nothing to reduce and is refused.
#[test]
fn reduce_only_with_no_position_is_refused() {
    let mut scn = flat_bars(3);
    scn[0].1.push(submit(true, 5.0, KindSpec::Market, true));
    let run = run_orders(&plain_env(), BacktestConfig::default(), &scn);
    assert!(run.fills.is_empty(), "a reduce-only order opened a position");
    assert_eq!(run.kernel.order(0).expect("order 0").status, OrderStatus::Rejected);
}

/// Minimal reproduction: a reduce-only sell while short closes the short.
#[test]
fn reduce_only_sell_while_short_is_refused() {
    let mut scn = flat_bars(3);
    scn[0].1.push(submit(false, 5.0, KindSpec::Market, false));
    scn[1].1.push(submit(false, 5.0, KindSpec::Market, true));
    let run = run_orders(&plain_env(), BacktestConfig::default(), &scn);
    assert!(run.closed.is_empty(), "a sell closed a short: {:?}", run.closed.len());
    assert_eq!(run.kernel.order(1).expect("order 1").status, OrderStatus::Rejected);
}

/// Minimal reproduction: a short is covered by a buy limit-if-touched at
/// 100 and pays 100.2 under 0.2% slippage.
#[test]
fn limit_if_touched_fill_applies_slippage() {
    let env = Env { slip: true, ..plain_env() };
    let mut scn = flat_bars(4);
    scn[0].1.push(submit(false, 1.0, KindSpec::Market, false));
    scn[1].1.push(submit(true, 1.0, KindSpec::LimitIfTouched(0.0, 0.0), false));
    let run = run_orders(&env, BacktestConfig::default(), &scn);
    let cover = run.fills.last().expect("the limit-if-touched filled");
    assert!(cover.price <= 100.0 + 1e-9, "buy limit 100 filled at {}", cover.price);
}

/// A triggered trailing-stop-limit rests as a limit: a short covered through
/// it pays no slippage and never more than the limit.
#[test]
fn trailing_stop_limit_fill_respects_its_limit_under_slippage() {
    let env = Env { slip: true, ..plain_env() };
    let mut scn = flat_bars(5);
    scn[0].1.push(submit(false, 1.0, KindSpec::Market, false));
    scn[1].1.push(submit(true, 1.0, KindSpec::TrailingStopLimit(0.01, 0.0), false));
    scn[2].0 = (0.0, 0.0, 0.02, 0.0); // low 100, high 102: trails to 100, triggers at 101
    scn[3].0 = (0.01, 0.0, 0.0, 0.0); // opens at 101, the limit
    let run = run_orders(&env, BacktestConfig::default(), &scn);
    let cover = run.fills.last().expect("the trailing-stop-limit filled");
    assert_eq!(cover.order_id, 1);
    assert!(cover.price <= 101.0 + 1e-9, "buy limit 101 filled at {}", cover.price);
}

// ---------------------------------------------------------------------------
// The generators reach what the properties talk about
// ---------------------------------------------------------------------------

/// A property over inputs that never trade passes for the wrong reason, so
/// pin that the generators produce trades, fills and every terminal order
/// state, in both directions and both account types.
#[test]
fn generators_reach_trades_fills_and_every_terminal_state() {
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;
    use std::collections::BTreeMap;

    let mut runner = TestRunner::new(config(0, 0x5005));
    let mut seen: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut bump = |k: &'static str, n: usize| *seen.entry(k).or_default() += n;

    let signals = (env(), prop::collection::vec(signal_bar(), 4..40));
    for _ in 0..200 {
        let (e, scn) = signals.new_tree(&mut runner).expect("tree").current();
        let run = run_signals(&e, &scn);
        let n = run.closed.len() + run.finalized.len();
        bump("signal trades", n);
        bump(if e.long { "long trades" } else { "short trades" }, n);
        bump(if e.is_cash() { "cash trades" } else { "margin trades" }, n);
        bump(
            "stop or target exits",
            run.closed
                .iter()
                .filter(|t| matches!(t.exit_reason, ExitReason::StopLoss | ExitReason::TakeProfit))
                .count(),
        );
        bump("gridded trades", if e.incr > 0.0 { n } else { 0 });
    }

    let orders = (orders_env(), order_scenario(true, true));
    for _ in 0..200 {
        let (e, scn) = orders.new_tree(&mut runner).expect("tree").current();
        let run = run_orders(&e, BacktestConfig::default(), &scn);
        bump("order fills", run.fills.len());
        bump("order trades", run.closed.len() + run.finalized.len());
        for o in run.kernel.order_book() {
            bump(
                match o.status {
                    OrderStatus::Filled => "orders filled",
                    OrderStatus::Canceled => "orders canceled",
                    OrderStatus::Expired => "orders expired",
                    OrderStatus::Rejected => "orders rejected",
                    OrderStatus::Accepted | OrderStatus::Triggered => "orders working",
                    _ => "orders other",
                },
                1,
            );
        }
    }

    let floor = [
        ("signal trades", 400),
        ("long trades", 100),
        ("short trades", 100),
        ("cash trades", 100),
        ("margin trades", 100),
        ("stop or target exits", 20),
        ("gridded trades", 100),
        ("order fills", 300),
        ("order trades", 100),
        ("orders filled", 300),
        ("orders canceled", 50),
        ("orders expired", 20),
        ("orders rejected", 20),
        ("orders working", 100),
    ];
    for (key, min) in floor {
        let got = seen.get(key).copied().unwrap_or(0);
        assert!(got >= min, "{key}: {got} < {min}; saw {seen:?}");
    }
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn trade_bits(trades: &[Trade]) -> Vec<(usize, usize, u64, u64, u64, u64, u64)> {
    trades
        .iter()
        .map(|t| {
            (
                t.entry_idx,
                t.exit_idx,
                t.entry_price.to_bits(),
                t.exit_price.to_bits(),
                t.size.to_bits(),
                t.pnl.to_bits(),
                t.fees.to_bits(),
            )
        })
        .collect()
}

proptest! {
    #![proptest_config(config(600, 0x3003))]

    /// The same inputs, including a stochastic-fill seed, give identical
    /// trades, orders and equity on every repeat.
    #[test]
    fn repeated_runs_are_bit_identical(
        env in orders_env(),
        scn in order_scenario(true, true),
        fill_seed in any::<u64>(),
    ) {
        let base = BacktestConfig {
            fill_prob_limit: 0.5,
            fill_prob_slippage: 0.3,
            fill_seed,
            ..BacktestConfig::default()
        };
        let first = run_orders(&env, base.clone(), &scn);
        for _ in 0..2 {
            let again = run_orders(&env, base.clone(), &scn);
            prop_assert_eq!(bits(&first.equity), bits(&again.equity));
            prop_assert_eq!(bits(&first.cash), bits(&again.cash));
            prop_assert_eq!(trade_bits(&first.closed), trade_bits(&again.closed));
            prop_assert_eq!(trade_bits(&first.finalized), trade_bits(&again.finalized));
            let a: Vec<_> = first
                .kernel
                .order_book()
                .iter()
                .map(|o| (o.id, o.status, o.filled_qty.to_bits()))
                .collect();
            let b: Vec<_> = again
                .kernel
                .order_book()
                .iter()
                .map(|o| (o.id, o.status, o.filled_qty.to_bits()))
                .collect();
            prop_assert_eq!(a, b);
            let fa: Vec<_> =
                first.fills.iter().map(|f| (f.idx, f.order_id, f.price.to_bits())).collect();
            let fb: Vec<_> =
                again.fills.iter().map(|f| (f.idx, f.order_id, f.price.to_bits())).collect();
            prop_assert_eq!(fa, fb);
        }
    }

    /// The signal path repeats exactly too.
    #[test]
    fn repeated_signal_runs_are_bit_identical(
        env in env(),
        scn in prop::collection::vec(signal_bar(), 4..40),
    ) {
        let first = run_signals(&env, &scn);
        for _ in 0..2 {
            let again = run_signals(&env, &scn);
            prop_assert_eq!(bits(&first.equity), bits(&again.equity));
            prop_assert_eq!(trade_bits(&first.closed), trade_bits(&again.closed));
            prop_assert_eq!(trade_bits(&first.finalized), trade_bits(&again.finalized));
        }
    }
}
