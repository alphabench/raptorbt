//! Guard tests for execution-timing causality.
//!
//! A decision made from bar i's data may trade no earlier than bar i's own
//! close. Under `FillTiming::NextBarOpen` it trades at bar i+1's open.
//! Through 0.10, `upon_bar_close=false` filled a bar-i signal at bar i's
//! OWN open — a price that traded before the information the signal used
//! existed. These tests pin the corrected semantics; if the shortcut ever
//! comes back, they fail.

use raptorbt::core::types::{
    BacktestConfig, CompiledSignals, Direction, ExitReason, FillTiming, OhlcvData, StopConfig,
    TargetConfig,
};
use raptorbt::portfolio::engine::PortfolioEngine;

fn config(fill_timing: FillTiming) -> BacktestConfig {
    BacktestConfig {
        initial_capital: 100_000.0,
        fees: 0.0,
        slippage: 0.0,
        stop: StopConfig::None,
        target: TargetConfig::None,
        fill_timing: Some(fill_timing),
        ..Default::default()
    }
}

fn flat_ohlcv(n: usize, price: f64) -> OhlcvData {
    OhlcvData {
        timestamps: (0..n as i64).map(|i| i * 1_000_000_000).collect(),
        open: vec![price; n],
        high: vec![price; n],
        low: vec![price; n],
        close: vec![price; n],
        volume: vec![1000.0; n],
    }
}

fn signals(n: usize, entry_at: usize, exit_at: Option<usize>) -> CompiledSignals {
    let mut entries = vec![false; n];
    let mut exits = vec![false; n];
    entries[entry_at] = true;
    if let Some(i) = exit_at {
        exits[i] = true;
    }
    CompiledSignals {
        symbol: "GUARD".to_string(),
        entries,
        exits,
        position_sizes: None,
        direction: Direction::Long,
        weight: 1.0,
    }
}

/// One bar rallies 100 -> 200 intrabar; every later price is 150 forever.
///
/// A causal trader learns "bar 5 closed at 200" only when it closes; the
/// cheapest price available from that moment on is 150, and the run ends at
/// 150 — no causal strategy acting on this signal can make money.
fn rally_fixture() -> OhlcvData {
    let mut ohlcv = flat_ohlcv(12, 150.0);
    ohlcv.open[5] = 100.0;
    ohlcv.low[5] = 100.0;
    ohlcv.high[5] = 200.0;
    ohlcv.close[5] = 200.0;
    // Bars before the signal sit at 150 too.
    for i in 0..5 {
        ohlcv.open[i] = 150.0;
        ohlcv.high[i] = 150.0;
        ohlcv.low[i] = 150.0;
        ohlcv.close[i] = 150.0;
    }
    ohlcv
}

#[test]
fn next_bar_open_fills_the_bar_after_the_signal() {
    let engine = PortfolioEngine::new(config(FillTiming::NextBarOpen));
    let result = engine.run_single(&rally_fixture(), &signals(12, 5, Some(10)));

    assert_eq!(result.trades.len(), 1);
    let trade = &result.trades[0];

    // The signal is a function of bar 5's close (200); the earliest
    // tradeable price after that is bar 6's open.
    assert_eq!(trade.entry_idx, 6, "entry fills on the bar after the signal");
    assert_eq!(trade.entry_price, 150.0, "entry fills at the next bar's open");
    // The exit signal at bar 10 fills at bar 11's open.
    assert_eq!(trade.exit_idx, 11);
    assert_eq!(trade.exit_price, 150.0);

    // The causally-impossible ~+50% is gone: buy 150, sell 150.
    assert!(
        result.metrics.total_return_pct.abs() < 1e-9,
        "no causal trader earns anything here; reported {}%",
        result.metrics.total_return_pct
    );
}

#[test]
fn next_bar_open_books_the_crash_the_exit_reacted_to() {
    // Crash bar: opens 150, closes 50; price stays at 50 afterwards. The
    // exit signal exists *because* of the 50 close, so a causal trader
    // cannot get out above 50.
    let mut ohlcv = flat_ohlcv(12, 150.0);
    ohlcv.low[8] = 50.0;
    ohlcv.close[8] = 50.0;
    for i in 9..12 {
        ohlcv.open[i] = 50.0;
        ohlcv.high[i] = 50.0;
        ohlcv.low[i] = 50.0;
        ohlcv.close[i] = 50.0;
    }

    let engine = PortfolioEngine::new(config(FillTiming::NextBarOpen));
    let result = engine.run_single(&ohlcv, &signals(12, 2, Some(8)));

    let trade = &result.trades[0];
    assert_eq!(trade.entry_idx, 3, "entered at the bar after the entry signal");
    assert_eq!(trade.entry_price, 150.0);
    assert_eq!(trade.exit_idx, 9, "exit fills the bar after the crash was seen");
    assert_eq!(trade.exit_price, 50.0, "the crash is not dodged");
    assert!(trade.pnl < 0.0, "the loss is real and must be booked");
}

/// The explicitly-named legacy mode reproduces the pre-0.11 numbers, and
/// nothing else does.
#[test]
fn lookahead_mode_reproduces_pre_0_11_results_by_name_only() {
    let engine = PortfolioEngine::new(config(FillTiming::SameBarOpenLookahead));
    let result = engine.run_single(&rally_fixture(), &signals(12, 5, Some(10)));

    let trade = &result.trades[0];
    // Pre-0.11 `upon_bar_close=false`: entry at the signal bar's own open.
    assert_eq!(trade.entry_idx, 5);
    assert_eq!(trade.entry_price, 100.0);
    assert_eq!(trade.exit_price, 150.0);
    assert!(
        result.metrics.total_return_pct > 45.0,
        "the legacy mode must replay the old (unearnable) result exactly"
    );
}

/// The deprecated bool maps onto the corrected semantics: `false` is
/// next-bar-open, never the old same-bar-open look-ahead.
#[test]
fn deprecated_bool_resolves_to_next_bar_open() {
    let cfg = BacktestConfig {
        initial_capital: 100_000.0,
        fees: 0.0,
        slippage: 0.0,
        stop: StopConfig::None,
        target: TargetConfig::None,
        upon_bar_close: false,
        fill_timing: None,
        ..Default::default()
    };
    assert_eq!(cfg.resolved_fill_timing(), FillTiming::NextBarOpen);

    let result = PortfolioEngine::new(cfg).run_single(&rally_fixture(), &signals(12, 5, Some(10)));
    assert_eq!(result.trades[0].entry_price, 150.0, "bool=false must not reach bar 5's open");
}

/// A signal on the final bar has no next bar to fill on: it does not trade.
#[test]
fn final_bar_signal_never_fills() {
    let engine = PortfolioEngine::new(config(FillTiming::NextBarOpen));
    let result = engine.run_single(&flat_ohlcv(12, 150.0), &signals(12, 11, None));
    assert!(result.trades.is_empty(), "a last-bar decision has nothing to trade against");
}

/// A position opened at bar i+1's open lives through bar i+1: its stop can
/// fire within the same bar it was filled on.
#[test]
fn deferred_entry_can_stop_out_within_its_fill_bar() {
    let mut cfg = config(FillTiming::NextBarOpen);
    cfg.stop = StopConfig::Fixed { percent: 0.02 };

    // Signal at bar 4; bar 5 opens at 150 and collapses through the stop.
    let mut ohlcv = flat_ohlcv(12, 150.0);
    ohlcv.low[5] = 100.0;
    ohlcv.close[5] = 110.0;
    for i in 6..12 {
        ohlcv.open[i] = 110.0;
        ohlcv.high[i] = 110.0;
        ohlcv.low[i] = 110.0;
        ohlcv.close[i] = 110.0;
    }

    let result = PortfolioEngine::new(cfg).run_single(&ohlcv, &signals(12, 4, None));
    let trade = &result.trades[0];
    assert_eq!(trade.entry_idx, 5);
    assert_eq!(trade.entry_price, 150.0);
    assert_eq!(trade.exit_reason, ExitReason::StopLoss);
    assert_eq!(trade.exit_idx, 5, "the fill bar's own range applies to the new position");
    assert_eq!(trade.exit_price, 147.0, "2% below the 150.0 entry");
}

/// Same-bar-close mode is untouched by the fix: a signal at bar i still
/// fills at bar i's close, the price the decision coincides with.
#[test]
fn same_bar_close_semantics_are_unchanged() {
    let engine = PortfolioEngine::new(config(FillTiming::SameBarClose));
    let result = engine.run_single(&rally_fixture(), &signals(12, 5, Some(10)));

    let trade = &result.trades[0];
    assert_eq!(trade.entry_idx, 5);
    assert_eq!(trade.entry_price, 200.0, "close-mode fills at the decision bar's close");
}

// ---- Multi-leg runners -------------------------------------------------
//
// Basket and pairs legs carry full OHLC, so NextBarOpen fills at each
// leg's own open on the bar after the decision. Their default (and
// historical) behavior is same-bar-close, which stays byte-identical.

fn stepped_ohlcv(n: usize) -> OhlcvData {
    // Distinct open vs close on every bar so a wrong price source is
    // always visible: bar i opens at 100+i and closes at 200+i.
    OhlcvData {
        timestamps: (0..n as i64).map(|i| i * 1_000_000_000).collect(),
        open: (0..n).map(|i| 100.0 + i as f64).collect(),
        high: (0..n).map(|i| 300.0 + i as f64).collect(),
        low: (0..n).map(|i| 90.0 + i as f64).collect(),
        close: (0..n).map(|i| 200.0 + i as f64).collect(),
        volume: vec![1000.0; n],
    }
}

#[test]
fn basket_next_bar_open_fills_every_leg_at_its_next_open() {
    use raptorbt::strategies::basket::{BasketBacktest, BasketConfig};

    let cfg = BasketConfig { base: config(FillTiming::NextBarOpen), ..BasketConfig::default() };
    let instruments = vec![
        (stepped_ohlcv(10), signals(10, 3, Some(6))),
        (stepped_ohlcv(10), signals(10, 3, Some(6))),
    ];
    let result = BasketBacktest::new(cfg).run(&instruments);

    assert_eq!(result.trades.len(), 2, "one trade per leg");
    for trade in &result.trades {
        assert_eq!(trade.entry_idx, 4, "decision at 3 fills on 4");
        assert_eq!(trade.entry_price, 104.0, "at bar 4's open");
        assert_eq!(trade.exit_idx, 7, "decision at 6 fills on 7");
        assert_eq!(trade.exit_price, 107.0, "at bar 7's open");
    }
}

#[test]
fn basket_default_close_fills_are_unchanged() {
    use raptorbt::strategies::basket::{BasketBacktest, BasketConfig};

    let cfg = BasketConfig { base: config(FillTiming::SameBarClose), ..BasketConfig::default() };
    let instruments = vec![(stepped_ohlcv(10), signals(10, 3, Some(6)))];
    let result = BasketBacktest::new(cfg).run(&instruments);

    let trade = &result.trades[0];
    assert_eq!((trade.entry_idx, trade.entry_price), (3, 203.0));
    assert_eq!((trade.exit_idx, trade.exit_price), (6, 206.0));
}

#[test]
fn basket_last_bar_signal_never_fills() {
    use raptorbt::strategies::basket::{BasketBacktest, BasketConfig};

    let cfg = BasketConfig { base: config(FillTiming::NextBarOpen), ..BasketConfig::default() };
    let instruments = vec![(stepped_ohlcv(10), signals(10, 9, None))];
    let result = BasketBacktest::new(cfg).run(&instruments);
    assert!(result.trades.is_empty(), "a last-bar decision has nothing to trade against");
}

#[test]
fn pairs_next_bar_open_fills_both_legs_at_their_next_open() {
    use raptorbt::strategies::pairs::{PairsBacktest, PairsConfig};

    let cfg = PairsConfig { base: config(FillTiming::NextBarOpen), ..PairsConfig::default() };
    let leg1 = stepped_ohlcv(10);
    let mut leg2 = stepped_ohlcv(10);
    // Different open levels per leg so each leg's own series is provably used.
    for v in leg2.open.iter_mut() {
        *v += 50.0;
    }
    let result = PairsBacktest::new(cfg).run(&leg1, &leg2, &signals(10, 3, Some(6)));

    assert_eq!(result.trades.len(), 2);
    let l1 = &result.trades[0];
    let l2 = &result.trades[1];
    assert_eq!((l1.entry_idx, l1.entry_price), (4, 104.0), "leg1 at its bar-4 open");
    assert_eq!((l2.entry_idx, l2.entry_price), (4, 154.0), "leg2 at its bar-4 open");
    assert_eq!(l1.exit_price, 107.0);
    assert_eq!(l2.exit_price, 157.0);
}

#[test]
fn pairs_default_close_fills_are_unchanged() {
    use raptorbt::strategies::pairs::{PairsBacktest, PairsConfig};

    let cfg = PairsConfig { base: config(FillTiming::SameBarClose), ..PairsConfig::default() };
    let leg = stepped_ohlcv(10);
    let result = PairsBacktest::new(cfg).run(&leg, &leg.clone(), &signals(10, 3, Some(6)));
    assert_eq!(result.trades[0].entry_price, 203.0);
    assert_eq!(result.trades[0].exit_price, 206.0);
}

// Options and spread legs are premium-only series — no open exists — so
// NextBarOpen fills at the bar AFTER the decision, at that bar's premium
// (the honest causal fill). Strike selection stays on the decision bar.

#[test]
fn options_next_bar_open_fills_at_the_next_bars_premium() {
    use raptorbt::strategies::options::{OptionsBacktest, OptionsConfig};

    let spot = stepped_ohlcv(10);
    // Premium series: distinct per bar (10 + i).
    let premiums: Vec<f64> = (0..10).map(|i| 10.0 + i as f64).collect();

    let cfg = OptionsConfig { base: config(FillTiming::NextBarOpen), ..OptionsConfig::default() };
    let result = OptionsBacktest::new(cfg).run(&spot, &premiums, &signals(10, 3, Some(6)));

    let trade = &result.trades[0];
    assert_eq!(trade.entry_idx, 4, "decision at 3 fills on 4");
    assert_eq!(trade.entry_price, 14.0, "at bar 4's premium");
    assert_eq!(trade.exit_idx, 7);
    assert_eq!(trade.exit_price, 17.0, "at bar 7's premium");

    // Default mode unchanged: same-bar premium.
    let cfg = OptionsConfig { base: config(FillTiming::SameBarClose), ..OptionsConfig::default() };
    let result = OptionsBacktest::new(cfg).run(&spot, &premiums, &signals(10, 3, Some(6)));
    assert_eq!(result.trades[0].entry_price, 13.0);
    assert_eq!(result.trades[0].exit_price, 16.0);
}

#[test]
fn spreads_next_bar_open_prices_legs_off_the_next_bar() {
    use raptorbt::strategies::{
        LegConfig, SpreadBacktest, SpreadConfig, SpreadOptionType, SpreadType,
    };

    let n = 10;
    let timestamps: Vec<i64> = (0..n as i64).map(|i| i * 1_000_000_000).collect();
    let underlying: Vec<f64> = vec![100.0; n];
    // Two legs with distinct per-bar premiums.
    let legs_premiums = vec![
        (0..n).map(|i| 10.0 + i as f64).collect::<Vec<f64>>(),
        (0..n).map(|i| 20.0 + i as f64).collect::<Vec<f64>>(),
    ];
    let mut entries = vec![false; n];
    let mut exits = vec![false; n];
    entries[3] = true;
    exits[6] = true;

    let make = |timing: FillTiming| SpreadConfig {
        base: config(timing),
        spread_type: SpreadType::Straddle,
        leg_configs: vec![
            LegConfig::new(SpreadOptionType::Call, 100.0, 1, 1),
            LegConfig::new(SpreadOptionType::Put, 100.0, -1, 1),
        ],
        max_loss: None,
        target_profit: None,
        leg_expiry_timestamps: None,
    };

    let result = SpreadBacktest::new(make(FillTiming::NextBarOpen)).run(
        &timestamps,
        &underlying,
        &legs_premiums,
        &entries,
        &exits,
    );
    assert_eq!(result.trades.len(), 1);
    let trade = &result.trades[0];
    assert_eq!(trade.entry_idx, 4, "decision at 3 opens on 4, at bar 4's premiums");
    assert_eq!(trade.exit_idx, 7, "decision at 6 closes on 7");

    // Default mode unchanged: same-bar entry/exit indices.
    let result = SpreadBacktest::new(make(FillTiming::SameBarClose)).run(
        &timestamps,
        &underlying,
        &legs_premiums,
        &entries,
        &exits,
    );
    assert_eq!(result.trades[0].entry_idx, 3);
    assert_eq!(result.trades[0].exit_idx, 6);
}
