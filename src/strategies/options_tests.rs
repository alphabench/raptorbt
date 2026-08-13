//! Tests for the options strategy path.
//!
//! Split out of `options.rs` when the cost defects below were fixed: the file
//! was at its length limit, and the accounting needed far more coverage than
//! the three helper tests it carried.

use super::*;
use crate::core::types::{BacktestConfig, CompiledSignals, Direction, ExitReason, OhlcvData};

// ---------------------------------------------------------------------------
// Helper unit tests (moved from options.rs unchanged)
// ---------------------------------------------------------------------------

#[test]
fn test_strike_selection_atm() {
    let config = OptionsConfig {
        strike_interval: 50.0,
        strike_selection: StrikeSelection::Atm,
        ..Default::default()
    };
    let backtest = OptionsBacktest::new(config);

    // Spot at 17834, ATM should be 17850
    let strike = backtest.select_strike(17834.0);
    assert!((strike - 17850.0).abs() < 1e-10);
}

#[test]
fn test_strike_selection_otm() {
    let config = OptionsConfig {
        strike_interval: 50.0,
        strike_selection: StrikeSelection::Otm(2),
        option_type: OptionType::Call,
        ..Default::default()
    };
    let backtest = OptionsBacktest::new(config);

    // Spot at 17834, ATM=17850, OTM 2 strikes = 17950
    let strike = backtest.select_strike(17834.0);
    assert!((strike - 17950.0).abs() < 1e-10);
}

#[test]
fn test_position_sizing_percent() {
    let config =
        OptionsConfig { size_type: SizeType::Percent(0.5), lot_size: 50, ..Default::default() };
    let backtest = OptionsBacktest::new(config);

    // 50% of 100000 = 50000, option at 100 * lot 50 = 5000 per contract
    let contracts = backtest.calculate_contracts(100.0, 100_000.0);
    assert_eq!(contracts, 10);
}

// ---------------------------------------------------------------------------
// Costs and accounting
//
// Six defects lived here, and no test in this file ever called `run()` -- the
// whole accounting path was unpinned, which is how they accumulated. Fees
// ignored the lot multiplier while P&L honored it; the entry charge was
// dropped from both the trade record and the P&L; exits were priced with the
// buy-side schedule; `total_fees_paid` read zero; `size` reported lots; and
// the end-of-data close was never paid for out of the equity curve.
//
// One lot of 50 at a premium of 100 is a 5000 notional, so a 0.1% rate is
// 5.00 per side and 10.00 for the round trip. Every figure below is derived
// from that, deliberately absolute rather than self-consistent.
// ---------------------------------------------------------------------------

/// One round trip on a flat premium, so fee arithmetic is the only thing
/// moving. Enters at bar 1 and exits on signal at bar 4 unless `hold` is set,
/// in which case the position is left open and closes at end of data.
fn costed_run(
    premium: f64,
    fees: f64,
    segment: Option<&str>,
    lot_size: usize,
    hold: bool,
) -> BacktestResult {
    let n = 6;
    let spot = vec![17_800.0; n];
    let ohlcv = OhlcvData {
        timestamps: (0..n as i64).map(|i| i * 60_000_000_000).collect(),
        open: spot.clone(),
        high: spot.clone(),
        low: spot.clone(),
        close: spot.clone(),
        volume: vec![0.0; n],
    };

    let mut entries = vec![false; n];
    entries[1] = true;
    let mut exits = vec![false; n];
    if !hold {
        exits[4] = true;
    }

    let signals = CompiledSignals::new("OPT".to_string(), entries, exits, Direction::Long, 1.0);

    let config = OptionsConfig {
        base: BacktestConfig {
            initial_capital: 500_000.0,
            fees,
            slippage: 0.0,
            fee_segment: segment.map(|s| s.to_string()),
            ..Default::default()
        },
        // One lot exactly, so `lot_size` is the only size lever under test.
        size_type: SizeType::Contracts(1),
        lot_size,
        ..Default::default()
    };

    OptionsBacktest::new(config).run(&ohlcv, &vec![premium; n], &signals)
}

#[test]
fn costs_scale_with_the_lot_size() {
    // The headline defect: the fee path passed lots where every P&L line
    // passed contracts, so a 50-lot position was charged as a single one.
    let one = costed_run(100.0, 0.001, None, 1, false);
    let fifty = costed_run(100.0, 0.001, None, 50, false);

    assert!(
        (fifty.trades[0].fees - one.trades[0].fees * 50.0).abs() < 1e-9,
        "a lot of 50 trades 50 contracts and owes 50x the cost: {} vs {}",
        fifty.trades[0].fees,
        one.trades[0].fees
    );
}

#[test]
fn a_round_trip_is_charged_on_both_sides() {
    // `entry_fees` was hard-coded to 0.0 and `fees` held the exit half alone,
    // while cash had been debited for both.
    let result = costed_run(100.0, 0.001, None, 50, false);

    let trade = &result.trades[0];
    assert_eq!(trade.entry_fees, 5.00, "entry is one side of a 5000 notional at 0.1%");
    assert_eq!(trade.exit_fees, 5.00, "exit is the other side");
    assert_eq!(trade.fees, 10.00, "a round trip is two sides, not one");
    assert_eq!(trade.fees, trade.entry_fees + trade.exit_fees, "Trade documents this invariant");
}

#[test]
fn reported_costs_equal_what_the_equity_curve_paid() {
    // Deliberately an absolute assertion. Checking only that
    // `fees == entry_fees + exit_fees` passes even when both halves are wrong
    // -- the figures agree with each other and are simply not the money that
    // left the account. That is the shape of defect that survives review.
    let result = costed_run(100.0, 0.001, None, 50, false);

    let trade = &result.trades[0];
    let start = result.equity_curve[0];
    let end = *result.equity_curve.last().unwrap();

    assert!(
        (start - end - 10.00).abs() < 1e-9,
        "premium never moves, so the whole fall in equity is cost: {}",
        start - end
    );
    assert!(
        (start - end - trade.fees).abs() < 1e-9,
        "curve fell {} but the trade reports {}",
        start - end,
        trade.fees
    );
}

#[test]
fn the_pnl_is_net_of_both_sides() {
    // The entry charge was missing from `pnl` too, not just from the reported
    // fields, so every trade looked one entry charge more profitable.
    let result = costed_run(100.0, 0.001, None, 50, false);

    let trade = &result.trades[0];
    assert!(
        (trade.pnl + 10.00).abs() < 1e-9,
        "a flat round trip loses exactly its round-trip cost: {}",
        trade.pnl
    );
}

#[test]
fn trade_size_is_contracts_not_lots() {
    // `size` reported the lot count, disagreeing with every other path.
    let result = costed_run(100.0, 0.001, None, 50, false);

    assert_eq!(result.trades[0].size, 50.0, "one lot of 50 is 50 contracts");
}

#[test]
fn the_metrics_summary_reports_the_costs_charged() {
    // `total_fees_paid` fell to Default and read zero however much was billed
    // -- the same defect the spread path carried before 0.7.3.
    let result = costed_run(100.0, 0.001, None, 50, false);

    let charged: f64 = result.trades.iter().map(|t| t.fees).sum();
    assert!(charged > 0.0);
    assert!(
        (result.metrics.total_fees_paid - charged).abs() < 1e-9,
        "summary says {} but the trades total {}",
        result.metrics.total_fees_paid,
        charged
    );
}

#[test]
fn an_itemized_segment_reports_its_components() {
    // Brokerage is a flat Rs 20 per order, which no percentage rate can
    // express -- and `fee_breakdown` was hard-coded None, so a configured
    // segment reached the model and reported nothing.
    let result = costed_run(100.0, 0.001, Some("NFO-OPT"), 50, false);

    let trade = &result.trades[0];
    let breakdown = trade.fee_breakdown.expect("an itemized segment reports its components");

    assert_eq!(breakdown.brokerage, 40.0, "2 orders (entry, exit) x Rs 20");
    assert!(
        (breakdown.total() - trade.fees).abs() < 1e-9,
        "the itemized total is the money actually charged"
    );
    assert_eq!(trade.fees, trade.entry_fees + trade.exit_fees);
    assert!(trade.fees > 40.0, "real costs dwarf the 10.00 flat rate here: {}", trade.fees);
}

#[test]
fn the_itemized_schedule_puts_each_charge_on_the_side_that_owes_it() {
    // Regression guard: this fails if anyone routes back through
    // `FeeModel::calculate`, which hard-codes is_entry = true and would charge
    // the buy schedule -- stamp duty, no transaction tax -- on both sides.
    let result = costed_run(100.0, 0.001, Some("NFO-OPT"), 50, false);
    let trade = &result.trades[0];

    assert!(
        trade.entry_fees != trade.exit_fees,
        "stamp duty falls on the buy and transaction tax on the sell, so the \
         two sides cannot be equal: {} vs {}",
        trade.entry_fees,
        trade.exit_fees
    );
    assert!(trade.exit_fees > trade.entry_fees, "the sell side carries STT");
}

#[test]
fn an_end_of_data_close_is_paid_for_out_of_the_equity_curve() {
    // The end-of-data close computed fees and pushed a trade but never touched
    // cash, and it runs after the loop had already written the last equity
    // point from the position marked to market. The exit charge appeared in
    // the trade list and nowhere else.
    let result = costed_run(100.0, 0.001, None, 50, true);

    let trade = &result.trades[0];
    assert_eq!(trade.exit_reason, ExitReason::EndOfData);

    let start = result.equity_curve[0];
    let end = *result.equity_curve.last().unwrap();
    assert!(
        (start - end - trade.fees).abs() < 1e-9,
        "curve fell {} but the trade reports {}",
        start - end,
        trade.fees
    );
    assert!(
        (result.metrics.end_value - end).abs() < 1e-9,
        "the reported end value is the corrected curve, not the marked one"
    );
}

#[test]
fn the_two_exit_paths_charge_the_same_costs() {
    // A signal exit and an end-of-data exit at the same premium are the same
    // round trip and must cost the same. Cross-checks the end-of-data path
    // against the one that was already crediting cash.
    let signal = costed_run(100.0, 0.001, Some("NFO-OPT"), 50, false);
    let end_of_data = costed_run(100.0, 0.001, Some("NFO-OPT"), 50, true);

    let a = &signal.trades[0];
    let b = &end_of_data.trades[0];

    assert_eq!(a.exit_reason, ExitReason::Signal);
    assert_eq!(b.exit_reason, ExitReason::EndOfData);
    assert!((a.fees - b.fees).abs() < 1e-9, "same round trip, same cost: {} vs {}", a.fees, b.fees);
    assert!((a.entry_fees - b.entry_fees).abs() < 1e-9);
    assert!((a.exit_fees - b.exit_fees).abs() < 1e-9);
}
