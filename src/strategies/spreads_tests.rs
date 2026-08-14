//! Tests for the multi-leg spread backtest.
//!
//! Split out of `spreads.rs`, which the file-size rules cap; included back
//! into that module so `super::*` and private items still resolve.

use super::*;
use crate::core::types::BacktestConfig;
use crate::core::types::StopConfig;
use crate::core::types::TargetConfig;

/// `(timestamps, underlying, per-leg premiums, entries, exits)`.
type SampleData = (Vec<i64>, Vec<f64>, Vec<Vec<f64>>, Vec<bool>, Vec<bool>);

fn sample_data() -> SampleData {
    let n = 20;
    let timestamps: Vec<i64> = (0..n as i64).collect();
    let underlying: Vec<f64> = (100..120).map(|x| x as f64).collect();

    // Call and Put premiums
    let call_premiums: Vec<f64> = (0..n).map(|i| 5.0 + (i as f64 * 0.2)).collect();
    let put_premiums: Vec<f64> = (0..n).map(|i| 5.0 - (i as f64 * 0.1)).collect();

    let legs_premiums = vec![call_premiums, put_premiums];

    let entries = vec![
        false, true, false, false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false,
    ];
    let exits = vec![
        false, false, false, false, false, false, false, false, false, true, false, false, false,
        false, false, false, false, false, false, false,
    ];

    (timestamps, underlying, legs_premiums, entries, exits)
}

#[test]
fn test_straddle_backtest() {
    let base_config = BacktestConfig {
        initial_capital: 100_000.0,
        fees: 0.001,
        slippage: 0.0,
        stop: StopConfig::None,
        target: TargetConfig::None,
        upon_bar_close: true,
        ..Default::default()
    };

    let config = create_straddle_config(base_config, 100.0, 50, true);
    let backtest = SpreadBacktest::new(config);

    let (timestamps, underlying, legs_premiums, entries, exits) = sample_data();

    let result = backtest.run(&timestamps, &underlying, &legs_premiums, &entries, &exits);

    assert_eq!(result.trades.len(), 1);
    assert!(result.equity_curve.len() == timestamps.len());
}

#[test]
fn test_iron_condor_backtest() {
    let base_config = BacktestConfig::default();

    let config = create_iron_condor_config(
        base_config,
        95.0,  // short put
        90.0,  // long put
        105.0, // short call
        110.0, // long call
        50,
    );

    let backtest = SpreadBacktest::new(config);

    let n = 20;
    let timestamps: Vec<i64> = (0..n as i64).collect();
    let underlying: Vec<f64> = vec![100.0; n];

    // Four legs: short put, long put, short call, long call
    let legs_premiums = vec![
        vec![3.0; n], // short put
        vec![1.5; n], // long put
        vec![3.0; n], // short call
        vec![1.5; n], // long call
    ];

    let mut entries = vec![false; n];
    entries[1] = true;

    let mut exits = vec![false; n];
    exits[15] = true;

    let result = backtest.run(&timestamps, &underlying, &legs_premiums, &entries, &exits);

    assert_eq!(result.trades.len(), 1);
    // Every leg holds a flat premium for the whole run, so the structure
    // neither gains nor loses and the four legs must sum to exactly zero
    // before costs. This pins the multi-leg summation path in
    // `total_unrealized_pnl`, which the single-leg tests below cannot
    // reach: a per-leg sign error that happened to cancel across a
    // symmetric condor would still surface here as a non-zero gross.
    let gross = result.trades[0].pnl + result.trades[0].fees;
    assert_eq!(gross, 0.0);
    // Costs are real, though, so the booked P&L is a small loss.
    assert!(result.trades[0].fees > 0.0);
    assert!(result.trades[0].pnl < 0.0);
    // Both sides are charged and both are reported. Premiums are flat
    // here, so the two halves are equal -- which is exactly why this
    // assertion has to name them separately: a run that charged one side
    // and reported the other would still satisfy the `gross == 0` check
    // above.
    let trade = &result.trades[0];
    assert_eq!(trade.fees, trade.entry_fees + trade.exit_fees);
    assert!(trade.entry_fees > 0.0);
    assert_eq!(trade.entry_fees, trade.exit_fees);
}

// ---------------------------------------------------------------------
// Signed P&L regression pins.
//
// Through 0.6.3 `LegPosition::unrealized_pnl` carried a stray leading
// minus that negated every spread result. These assert the correct signed
// answer for all four short/long x win/lose cases, plus the two exit
// triggers that read the same figure.
//
// Fees are zeroed so the numbers are exact integers. The design -- assert
// precise values on a full deterministic engine run instead of reaching
// for a snapshot library -- follows nautilus's acceptance tests
// (tests/acceptance_tests/test_backtest.py). Read, not copied.
// ---------------------------------------------------------------------

/// One CE leg, lot 75, entered on bar 1 and exited on bar 4.
///
/// Premiums move thirty points across a 75 lot, so every correct answer
/// below is 2250.
fn single_leg(premiums: &[f64], quantity: i32) -> BacktestResult {
    single_leg_with(premiums, quantity, None, None, true)
}

/// As `single_leg`, but lets a test set the max-loss / target-profit
/// thresholds and drop the exit signal so a trigger is the only way out.
fn single_leg_with(
    premiums: &[f64],
    quantity: i32,
    max_loss: Option<f64>,
    target_profit: Option<f64>,
    exit_on_bar_4: bool,
) -> BacktestResult {
    let n = premiums.len();
    let timestamps: Vec<i64> = (0..n as i64).map(|i| i * 300_000_000_000).collect();
    let underlying = vec![24_550.0; n];

    let mut entries = vec![false; n];
    entries[1] = true;
    let mut exits = vec![false; n];
    if exit_on_bar_4 {
        exits[4] = true;
    }

    let config = SpreadConfig {
        base: BacktestConfig {
            initial_capital: 500_000.0,
            // Zeroed so assertions are exact: with the default rate the
            // true 2250 arrives as 2259, which hides sign-adjacent slips.
            fees: 0.0,
            slippage: 0.0,
            ..Default::default()
        },
        spread_type: SpreadType::Custom,
        leg_configs: vec![LegConfig::new(OptionType::Call, 24_800.0, quantity, 75)],
        max_loss,
        target_profit,
        ..Default::default()
    };

    SpreadBacktest::new(config).run(
        &timestamps,
        &underlying,
        &[premiums.to_vec()],
        &entries,
        &exits,
    )
}

/// Premium falls 100 -> 60. Entered at bar 1 (90), exited at bar 4 (60).
const FALLING: [f64; 6] = [100.0, 90.0, 80.0, 70.0, 60.0, 60.0];
/// The mirror: premium rises, entered at 70, exited at 100.
const RISING: [f64; 6] = [60.0, 70.0, 80.0, 90.0, 100.0, 100.0];

#[test]
fn a_short_leg_that_gained_reports_a_profit() {
    let result = single_leg(&FALLING, -1);

    // Sold at 90, bought back at 60: a 2250 gain, reported as a gain.
    assert_eq!(result.trades[0].pnl, 2250.0);
}

#[test]
fn a_short_leg_that_lost_reports_a_loss() {
    let result = single_leg(&RISING, -1);

    // Sold at 70, bought back at 100: a 2250 loss, reported as a loss.
    assert_eq!(result.trades[0].pnl, -2250.0);
}

#[test]
fn a_long_leg_that_gained_reports_a_profit() {
    let result = single_leg(&RISING, 1);

    // Bought at 70, sold at 100: a 2250 gain, reported as a gain.
    assert_eq!(result.trades[0].pnl, 2250.0);
}

#[test]
fn a_long_leg_that_lost_reports_a_loss() {
    let result = single_leg(&FALLING, 1);

    // Bought at 90, sold at 60: a 2250 loss, reported as a loss.
    assert_eq!(result.trades[0].pnl, -2250.0);
}

#[test]
fn net_premium_signs_are_already_correct() {
    let result = single_leg(&FALLING, -1);
    let trade = &result.trades[0];

    // Not a bug, and deliberately pinned: `entry_price` / `exit_price` are
    // computed independently of `unrealized_pnl` and are already right, so
    // the trade record carries the correct answer next to the wrong one.
    // The sign fix must not disturb these -- that is what makes it a pure
    // sign flip rather than a repricing.
    assert_eq!(trade.entry_price, -6750.0); // -90 * 75, a credit
    assert_eq!(trade.exit_price, -4500.0); // -60 * 75
    assert_eq!(trade.exit_price - trade.entry_price, 2250.0);
    // And the leg is sized off its stated 75 lot, not some other number.
    assert_eq!(trade.entry_price / -90.0, 75.0);
}

// The four trigger tests below are the reason the sign fix needed its own
// pins rather than riding along on the P&L assertions. `check_max_loss`
// and `check_target_profit` both read `total_unrealized_pnl`, so inverting
// it did not merely misreport a number -- it decided when a position
// closed. Each test drops the exit signal entirely, leaving the threshold
// as the only way out, so `exit_reason` is the assertion.
//
// A non-firing threshold is asserted as `EndOfData`, not as an empty trade
// book. Through 0.7.1 the terminal open position was settled into `cash`
// without recording a Trade, so "the stop did not fire" and "the position
// was never reported" were indistinguishable. 0.7.2 records that close, so
// the distinction is now expressible -- and these tests assert the thing
// they were always about: which exit reason closed the position.

#[test]
fn max_loss_does_not_fire_on_a_winner() {
    let result = single_leg_with(&FALLING, -1, Some(1000.0), None, false);

    // This leg gained 2250. Through 0.6.3 the stop read it as -2250 and
    // closed the position: a max-loss stop firing on a winner, which is
    // the most damaging half of the sign bug.
    assert_eq!(result.trades.len(), 1);
    assert_eq!(result.trades[0].exit_reason, ExitReason::EndOfData);
}

#[test]
fn max_loss_fires_on_a_real_loser() {
    let result = single_leg_with(&RISING, -1, Some(1000.0), None, false);

    // Sold at 70, premium ran to 100: a 2250 loss, past the 1000
    // threshold. The stop must still work.
    assert_eq!(result.trades.len(), 1);
    assert_eq!(result.trades[0].exit_reason, ExitReason::StopLoss);
}

#[test]
fn target_profit_does_not_fire_on_a_loser() {
    let result = single_leg_with(&RISING, -1, None, Some(1000.0), false);

    // This leg lost 2250. Through 0.6.3 the target read it as +2250 and
    // booked a "profit" on it.
    assert_eq!(result.trades.len(), 1);
    assert_eq!(result.trades[0].exit_reason, ExitReason::EndOfData);
}

#[test]
fn target_profit_fires_on_a_real_winner() {
    let result = single_leg_with(&FALLING, -1, None, Some(1000.0), false);

    assert_eq!(result.trades.len(), 1);
    assert_eq!(result.trades[0].exit_reason, ExitReason::TakeProfit);
}

/// Squareoff closes the position before the overnight gap.
///
/// **In plain words: without this, a backtest books profit the trader
/// could never have earned, because their broker would have closed the
/// position before the market shut.**
///
/// Two sessions, entered each morning, with a large adverse premium jump
/// between them. With squareoff the position is flat overnight and the gap
/// never touches it. Without squareoff the gap is booked as a price move
/// the strategy "traded through". The gap is deliberately far larger than
/// the intraday drift so the two arms cannot be confused.
///
/// This is the defect that prompted 0.7.2: on a
/// real NIFTY option corpus, removing the overnight hold moved a short
/// straddle's P&L by -24% and a short strangle's by -42%, and it moved a
/// LONG straddle the other way (+16%) -- confirming the defect amplifies
/// whichever direction a position already points rather than adding a
/// constant bias.
#[test]
fn squareoff_flattens_before_the_overnight_gap() {
    // 09:15 and 15:29 IST on two consecutive days.
    const IST_NS: i64 = (5 * 3600 + 30 * 60) * 1_000_000_000;
    let ist =
        |day: i64, h: i64, m: i64| (day * 86_400 + h * 3600 + m * 60) * 1_000_000_000 - IST_NS;
    let timestamps = vec![ist(0, 9, 15), ist(0, 15, 29), ist(1, 9, 15), ist(1, 15, 29)];

    // Premium drifts down 100 -> 95 intraday (a gain for a short), then
    // gaps UP to 200 overnight (a large loss for a short).
    let premiums = vec![100.0, 95.0, 200.0, 195.0];
    let underlying = vec![24_550.0; 4];
    let entries = vec![true, false, true, false];
    let exits = vec![false; 4];

    let base = |squareoff: Option<u32>| BacktestConfig {
        initial_capital: 500_000.0,
        fees: 0.0,
        slippage: 0.0,
        squareoff_time_minutes: squareoff,
        // The squareoff time is a LOCAL time, so the offset that defines
        // "local" has to be set too. Leaving it at the 0 (UTC) default
        // reads 15:29 IST as 09:59 and the squareoff never fires -- a
        // mistake worth pinning, since it fails silently in exactly the
        // way the original defect did.
        session_tz_offset_ns: IST_NS,
        ..Default::default()
    };
    let spread = |squareoff: Option<u32>| SpreadConfig {
        base: base(squareoff),
        spread_type: SpreadType::Custom,
        leg_configs: vec![LegConfig::new(OptionType::Call, 24_800.0, -1, 75)],
        ..Default::default()
    };

    let without = SpreadBacktest::new(spread(None)).run(
        &timestamps,
        &underlying,
        std::slice::from_ref(&premiums),
        &entries,
        &exits,
    );
    // 15:25 IST.
    let with = SpreadBacktest::new(spread(Some(15 * 60 + 25))).run(
        &timestamps,
        &underlying,
        &[premiums],
        &entries,
        &exits,
    );

    // Without squareoff: one position entered on day 0 and held through
    // the gap. Sold at 100, marked at 195: a 95-point loss on a short.
    assert_eq!(without.trades.len(), 1);
    assert_eq!(without.trades[0].pnl, -95.0 * 75.0);

    // With squareoff: two separate day trades, each closed at 15:29 --
    // the first bar at or after 15:25 in its session.
    assert_eq!(with.trades.len(), 2);
    assert_eq!(with.trades[0].exit_reason, ExitReason::Squareoff);
    assert_eq!(with.trades[1].exit_reason, ExitReason::Squareoff);

    // Each captured only its own session's 5-point decay, and the
    // overnight gap reached neither of them.
    assert_eq!(with.trades[0].pnl, 5.0 * 75.0);
    assert_eq!(with.trades[1].pnl, 5.0 * 75.0);

    // The headline: the defect is worth more than the strategy earns.
    assert!(with.metrics.end_value > without.metrics.end_value);
}

/// Squareoff must not re-open the position it just closed.
///
/// The entry signal fires every bar here, including the squareoff bar. A
/// broker's square-off leaves the trader flat into the close; re-entering
/// on the same bar would put the position straight back on and carry it
/// overnight anyway, silently restoring the defect.
#[test]
fn squareoff_does_not_re_enter_on_the_same_bar() {
    const IST_NS: i64 = (5 * 3600 + 30 * 60) * 1_000_000_000;
    let ist =
        |day: i64, h: i64, m: i64| (day * 86_400 + h * 3600 + m * 60) * 1_000_000_000 - IST_NS;
    let timestamps = vec![ist(0, 9, 15), ist(0, 15, 29), ist(1, 9, 15)];
    let premiums = vec![100.0, 95.0, 200.0];

    let result = SpreadBacktest::new(SpreadConfig {
        base: BacktestConfig {
            initial_capital: 500_000.0,
            fees: 0.0,
            slippage: 0.0,
            squareoff_time_minutes: Some(15 * 60 + 25),
            session_tz_offset_ns: IST_NS,
            ..Default::default()
        },
        spread_type: SpreadType::Custom,
        leg_configs: vec![LegConfig::new(OptionType::Call, 24_800.0, -1, 75)],
        ..Default::default()
    })
    .run(&timestamps, &[24_550.0; 3], &[premiums], &[true; 3], &[false; 3]);

    // Day 0 squared off at 15:29, then re-entered fresh on day 1 and left
    // open at end of data. Never a position spanning the gap.
    assert_eq!(result.trades.len(), 2);
    assert_eq!(result.trades[0].exit_reason, ExitReason::Squareoff);
    assert_eq!(result.trades[1].exit_reason, ExitReason::EndOfData);
    assert_eq!(result.trades[0].pnl, 5.0 * 75.0);
}

/// A position still open at the end of the data must appear in `trades()`.
///
/// Through 0.7.1 it did not. The terminal close settled P&L into `cash`,
/// so it reached `end_value`, `total_return_pct` and the equity curve --
/// but pushed no `Trade` and never called `record_trade`. The result was a
/// run whose entire return came from a position that `trades()` did not
/// list and `total_closed_trades` counted as zero.
///
/// That is the most dangerous shape a reporting defect can take: every
/// trade-level audit passes, because there is nothing to audit. It was
/// found by a run measuring overnight holds, where a single position
/// opened on the first morning and never closed carried the whole P&L.
///
/// The P&L assertion matters as much as the count: the recorded trade must
/// carry the SAME number that reached equity, or this fix would trade a
/// silent omission for a visible inconsistency.
#[test]
fn a_position_open_at_the_end_is_still_recorded_as_a_trade() {
    // No exit signal and no threshold: the only way out is end-of-data.
    let result = single_leg_with(&FALLING, -1, None, None, false);

    assert_eq!(result.trades.len(), 1, "the open position must be reported");
    let trade = &result.trades[0];
    assert_eq!(trade.exit_reason, ExitReason::EndOfData);

    // Entered at 90, ran to the final bar at 60: a 2250 gain on a short.
    assert_eq!(trade.pnl, 2250.0);

    // And it is counted, not merely listed.
    assert_eq!(result.metrics.total_closed_trades, 1);

    // The recorded P&L is the P&L that moved equity.
    let equity_gain = result.metrics.end_value - 500_000.0;
    assert!(
        (trade.pnl - equity_gain).abs() < 1e-9,
        "trade pnl {} must match the equity change {}",
        trade.pnl,
        equity_gain
    );
}

// ---------------------------------------------------------------------
// Cost regression pins.
//
// Through 0.7.2 the spread path computed its own fees instead of using the
// engine's fee model, and got four things wrong at once: it charged the
// entry twice, reported only the exit half on the trade, ignored the
// itemized schedule entirely, and dropped each leg's quantity. Every one
// of them understated costs, so a losing structure could read as a winner.
//
// These assert exact figures on full deterministic runs. A flat rate makes
// the arithmetic checkable by hand: one leg, one lot of 75, premium 100,
// rate 0.001 gives 100 * 75 * 0.001 = 7.50 per side.
// ---------------------------------------------------------------------

/// One CE leg with costs live, so the fee arithmetic is the thing measured.
///
/// The P&L helpers above zero the rate deliberately; these tests need it
/// non-zero. `segment` selects the itemized schedule when set.
fn costed_leg(premiums: &[f64], quantity: i32, fees: f64, segment: Option<&str>) -> BacktestResult {
    costed_spread(&[(premiums.to_vec(), quantity)], fees, segment, None)
}

/// As `costed_leg`, but with one entry per leg, and an optional expiry.
///
/// Each tuple is `(premium series, signed quantity)`; every leg carries a
/// lot of 75. `expiry_at` force-closes via settlement at that bar index.
fn costed_spread(
    legs: &[(Vec<f64>, i32)],
    fees: f64,
    segment: Option<&str>,
    expiry_at: Option<usize>,
) -> BacktestResult {
    let n = legs[0].0.len();
    let timestamps: Vec<i64> = (0..n as i64).map(|i| i * 300_000_000_000).collect();
    let underlying = vec![24_550.0; n];

    let mut entries = vec![false; n];
    entries[1] = true;
    let mut exits = vec![false; n];
    if expiry_at.is_none() {
        exits[4] = true;
    }

    let leg_configs: Vec<LegConfig> =
        legs.iter().map(|(_, q)| LegConfig::new(OptionType::Call, 24_800.0, *q, 75)).collect();
    let premiums: Vec<Vec<f64>> = legs.iter().map(|(p, _)| p.clone()).collect();

    let config = SpreadConfig {
        base: BacktestConfig {
            initial_capital: 500_000.0,
            fees,
            slippage: 0.0,
            fee_segment: segment.map(|s| s.to_string()),
            ..Default::default()
        },
        spread_type: SpreadType::Custom,
        leg_configs,
        leg_expiry_timestamps: expiry_at.map(|i| vec![timestamps[i]]),
        ..Default::default()
    };

    SpreadBacktest::new(config).run(&timestamps, &underlying, &premiums, &entries, &exits)
}

/// One leg, flat premium, flat rate: entry and exit each cost 7.50.
///
/// Pins the double charge. Through 0.7.2 the exit function multiplied by
/// two on top of an entry that had already been charged, so a round trip
/// billed three sides -- 22.50 where 15.00 was owed.
#[test]
fn a_round_trip_is_charged_exactly_twice() {
    let premiums = vec![100.0; 6];
    let result = costed_leg(&premiums, 1, 0.001, None);

    let trade = &result.trades[0];
    assert_eq!(trade.entry_fees, 7.50, "entry is one side of a 7500 notional at 0.1%");
    assert_eq!(trade.exit_fees, 7.50, "exit is the other side, not two");
    assert_eq!(trade.fees, 15.00, "a round trip is two sides, never three");
}

/// The reported total is the money the equity curve actually lost.
///
/// Pins the reporting gap. The entry charge used to be a local that nothing
/// retained, so `trades()` disclosed only the exit half while the curve had
/// been debited for both -- a divergence no trade-level audit could find.
#[test]
fn reported_costs_equal_what_the_equity_curve_paid() {
    let premiums = vec![100.0; 6];
    let result = costed_leg(&premiums, 1, 0.001, None);

    let trade = &result.trades[0];
    assert_eq!(trade.fees, trade.entry_fees + trade.exit_fees);

    // Premium never moves, so the whole fall in equity is cost.
    let start = result.equity_curve[0];
    let end = *result.equity_curve.last().unwrap();
    assert!(
        (start - end - trade.fees).abs() < 1e-9,
        "curve fell {} but trade says {}",
        start - end,
        trade.fees
    );
}

/// A two-lot leg costs twice a one-lot leg.
///
/// Pins the dropped quantity. Both fee functions used `lot_size` alone and
/// never `|quantity|`, so every multi-lot leg was billed as a single lot
/// however large the position.
#[test]
fn costs_scale_with_leg_quantity() {
    let premiums = vec![100.0; 6];
    let one = costed_leg(&premiums, 1, 0.001, None);
    let two = costed_leg(&premiums, 2, 0.001, None);

    assert_eq!(two.trades[0].fees, one.trades[0].fees * 2.0);
}

/// Cost is charged on the premium traded, whichever way the leg points.
///
/// A short leg sells to open and buys to close; a long leg does the
/// reverse. Under a flat rate the two sides cost the same, so the totals
/// match -- the asymmetry only appears under an itemized schedule.
#[test]
fn a_short_leg_and_a_long_leg_pay_the_same_flat_rate() {
    let premiums = vec![100.0; 6];
    let long = costed_leg(&premiums, 1, 0.001, None);
    let short = costed_leg(&premiums, -1, 0.001, None);

    assert_eq!(long.trades[0].fees, short.trades[0].fees);
}

/// Setting a segment charges the real schedule, per leg, per side.
///
/// Pins the ignored segment. The spread path held no fee model at all, so
/// the itemized schedule was unreachable and only the flat proportional
/// rate ever applied. Brokerage is flat per order, so a purely
/// proportional model never charges it at any premium -- the error is
/// largest exactly where cheap options live.
///
/// Four legs, two sides, at a flat 20 per order is 160 of brokerage that
/// the old path could not bill however the rate was set.
#[test]
fn an_itemized_segment_charges_per_order_brokerage_on_every_leg() {
    let premiums = vec![100.0; 6];
    let legs = vec![
        (premiums.clone(), 1),
        (premiums.clone(), -1),
        (premiums.clone(), 1),
        (premiums.clone(), -1),
    ];
    let result = costed_spread(&legs, 0.001, Some("NFO-OPT"), None);

    let trade = &result.trades[0];
    let breakdown = trade.fee_breakdown.expect("an itemized segment reports its components");

    assert_eq!(breakdown.brokerage, 160.0, "4 legs x 2 sides x 20 per order");
    assert!((breakdown.total() - trade.fees).abs() < 1e-9, "itemized total is the money charged");
    assert_eq!(trade.fees, trade.entry_fees + trade.exit_fees);

    // The flat rate would have billed 4 legs x 7.50 x 2 sides = 60.00 and no
    // brokerage at all. The real schedule is multiples of that.
    assert!(trade.fees > 200.0, "real costs dwarf the flat rate here: {}", trade.fees);
}

/// A leg left to expire pays nothing on the way out.
///
/// An option that expires is never sold: no order is placed, so no
/// brokerage and no transaction tax are owed. Charging a full exit there
/// would overstate the cost of every structure held to expiry -- the
/// mirror image of the undercharge this release fixes. Entry still stands.
#[test]
fn an_expiring_leg_pays_entry_costs_only() {
    let premiums = vec![100.0; 6];
    let result = costed_spread(&[(premiums, 1)], 0.001, Some("NFO-OPT"), Some(3));

    let trade = &result.trades[0];
    assert_eq!(trade.exit_reason, ExitReason::Settlement, "the leg expired");
    assert_eq!(trade.exit_fees, 0.0, "an expiring option is never traded out");
    assert!(trade.entry_fees > 0.0, "opening it was a real order");
    assert_eq!(trade.fees, trade.entry_fees);
}

/// Under the real schedule, the two directions do not cost the same.
///
/// Transaction tax lands on the sell and stamp duty on the buy, so a short
/// leg owes the tax when it opens and a long leg when it closes. Both pay
/// it once over a round trip; the halves differ, and a model that split a
/// round-trip figure evenly across the two sides would report neither
/// correctly.
#[test]
fn the_itemized_schedule_puts_each_charge_on_the_side_that_owes_it() {
    let premiums = vec![100.0; 6];
    let long = costed_leg(&premiums, 1, 0.001, Some("NFO-OPT"));
    let short = costed_leg(&premiums, -1, 0.001, Some("NFO-OPT"));

    let long_trade = &long.trades[0];
    let short_trade = &short.trades[0];

    // A long buys to open, so its entry owes stamp duty and its exit the tax.
    assert!(long_trade.entry_fees != long_trade.exit_fees, "the two sides differ");
    // A short sells to open, so the tax falls on entry instead -- the exact
    // reverse split, for the same round-trip total on a flat premium.
    assert!(short_trade.entry_fees > short_trade.exit_fees);
    assert!(long_trade.exit_fees > long_trade.entry_fees);
    assert!((long_trade.fees - short_trade.fees).abs() < 1e-9, "same round trip either way");
}

/// `total_fees_paid` reports the costs the run actually charged.
///
/// The spread path never called `record_fees`, so the summary metric read
/// zero however much was billed -- the same shape of defect as the trade
/// list disagreeing with the curve, one level up.
#[test]
fn the_metrics_summary_reports_the_costs_charged() {
    let premiums = vec![100.0; 6];
    let result = costed_leg(&premiums, 1, 0.001, None);

    let charged: f64 = result.trades.iter().map(|t| t.fees).sum();
    assert!(charged > 0.0);
    assert!(
        (result.metrics.total_fees_paid - charged).abs() < 1e-9,
        "summary says {} but trades total {}",
        result.metrics.total_fees_paid,
        charged
    );
}

/// A leg holding nothing is charged nothing.
///
/// Quantity is signed, and zero means the leg trades no contracts. It
/// places no order, so it owes no per-order brokerage -- which only became
/// visible once a flat per-order charge existed on this path at all.
#[test]
fn a_zero_quantity_leg_costs_nothing() {
    let premiums = vec![100.0; 6];
    let result = costed_spread(&[(premiums, 0)], 0.001, Some("NFO-OPT"), None);

    let trade = &result.trades[0];
    assert_eq!(trade.fees, 0.0, "a leg that trades nothing owes nothing");
}
