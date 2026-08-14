//! Tests for legs that expire on different days.
//!
//! Split from `spreads_tests.rs`, which the file-size rules cap; included
//! back into `spreads.rs` so `super::*` and private items still resolve.
//!
//! The structure under test throughout is a **calendar spread**: sell an
//! option expiring soon, buy one expiring later at the same strike. The
//! whole trade is the gap between the two expiries -- the near leg dies and
//! the far leg keeps living. Through 0.7.4 the engine closed both the moment
//! the near one expired, so it never simulated the part of the trade that is
//! the trade.

use super::*;
use crate::core::types::BacktestConfig;

const LOT: usize = 75;
const STRIKE: f64 = 24_800.0;
const BARS: usize = 31;
const NEAR_EXPIRY_BAR: usize = 20;
const FAR_EXPIRY_BAR: usize = 30;

/// `(premium series, signed quantity, expiry bar)` for one leg.
type LegSpec = (Vec<f64>, i32, usize);

/// The near leg: sold for 50, expires worthless at bar 20.
///
/// The premium series carries the settlement value from the expiry bar
/// onward, because that is the engine's contract -- the caller substitutes
/// intrinsic, the engine never invents a price. `after` is what the series
/// reads once the leg is dead; a caller who leaves stale quotes there must
/// not be able to move the result, which is what `after = 999.0` proves.
fn near_leg(after: f64) -> LegSpec {
    let mut premiums = vec![50.0; BARS];
    premiums[NEAR_EXPIRY_BAR] = 0.0;
    for p in premiums.iter_mut().take(BARS).skip(NEAR_EXPIRY_BAR + 1) {
        *p = after;
    }
    (premiums, -1, NEAR_EXPIRY_BAR)
}

/// The far leg: bought for 80, worth `settles_at` at its own expiry, bar 30.
fn far_leg(settles_at: f64) -> LegSpec {
    let mut premiums = vec![80.0; BARS];
    premiums[FAR_EXPIRY_BAR] = settles_at;
    (premiums, 1, FAR_EXPIRY_BAR)
}

/// Run a spread whose legs carry their own expiries. No exit signal unless
/// `exit_at` is given, so only expiry can close the position.
fn calendar(legs: &[LegSpec], exit_at: Option<usize>, segment: Option<&str>) -> BacktestResult {
    let n = legs[0].0.len();
    let timestamps: Vec<i64> = (0..n as i64).map(|i| i * 300_000_000_000).collect();
    let underlying = vec![24_550.0; n];

    let mut entries = vec![false; n];
    entries[1] = true;
    let mut exits = vec![false; n];
    if let Some(bar) = exit_at {
        exits[bar] = true;
    }

    let config = SpreadConfig {
        base: BacktestConfig {
            initial_capital: 500_000.0,
            fees: 0.0,
            slippage: 0.0,
            fee_segment: segment.map(|s| s.to_string()),
            ..Default::default()
        },
        spread_type: SpreadType::Custom,
        leg_configs: legs
            .iter()
            .map(|(_, q, _)| LegConfig::new(OptionType::Call, STRIKE, *q, LOT))
            .collect(),
        leg_expiry_timestamps: Some(legs.iter().map(|(_, _, e)| timestamps[*e]).collect()),
        ..Default::default()
    };

    let premiums: Vec<Vec<f64>> = legs.iter().map(|(p, _, _)| p.clone()).collect();
    SpreadBacktest::new(config).run(&timestamps, &underlying, &premiums, &entries, &exits)
}

/// The near leg expiring must not take the far leg with it.
///
/// Sold at 50 and expiring worthless, the near leg keeps its premium:
/// (-1) * (0 - 50) * 75 = +3750. The far leg, bought at 80, is worth 100 at
/// its own expiry ten bars later: (+1) * (100 - 80) * 75 = +1500. The trade
/// made 5250. Through 0.7.4 the engine closed everything at bar 20 with the
/// far leg still at 80, reported 3750, and called it a settlement.
#[test]
fn a_calendar_survives_its_near_leg_expiry() {
    let result = calendar(&[near_leg(0.0), far_leg(100.0)], None, None);

    assert_eq!(result.trades.len(), 1, "one structure opened, one trade");
    let trade = &result.trades[0];
    assert_eq!(trade.exit_idx, FAR_EXPIRY_BAR, "closed when the LAST leg expired");
    assert_eq!(trade.exit_reason, ExitReason::Settlement, "every leg reached its own expiry");
    assert!(
        (trade.pnl - 5250.0).abs() < 1e-9,
        "near leg +3750 and far leg +1500; got {}",
        trade.pnl
    );
}

/// What the far leg does after the near one dies has to reach the result.
///
/// This is the sharper statement of the defect. The old engine returned the
/// same number whatever the far leg did, because it had already closed --
/// so the error was not a bias anyone could correct for, it was a result
/// uncorrelated with the trade. Two runs differing only in the far leg's
/// settlement price must differ by exactly that difference: (100 - 60) * 75.
#[test]
fn the_far_leg_still_moves_the_result_after_the_near_leg_expires() {
    let up = calendar(&[near_leg(0.0), far_leg(100.0)], None, None);
    let down = calendar(&[near_leg(0.0), far_leg(60.0)], None, None);

    let spread = up.trades[0].pnl - down.trades[0].pnl;
    assert!((spread - 3000.0).abs() < 1e-9, "a 40-point move on a 75 lot is 3000; got {spread}");
}

/// Settling a leg moves money between two pockets, not into the account.
///
/// At the settlement bar the leg's profit leaves the mark-to-market and is
/// credited to cash. Those are the two halves of the equity line, so equity
/// must not move at all. If it jumps, the same rupees are being counted
/// twice -- once as cash and once as an open position.
#[test]
fn settling_a_leg_does_not_move_equity() {
    // The near leg is already worthless a bar before it expires, and the far
    // leg does not move across the boundary. So nothing about the market
    // changes at bar 20 -- only the bookkeeping does, and equity must not
    // notice it at all.
    let mut near = vec![50.0; BARS];
    for p in near.iter_mut().skip(NEAR_EXPIRY_BAR - 1) {
        *p = 0.0;
    }
    let result = calendar(&[(near, -1, NEAR_EXPIRY_BAR), far_leg(100.0)], None, None);

    let before = result.equity_curve[NEAR_EXPIRY_BAR - 1];
    let at = result.equity_curve[NEAR_EXPIRY_BAR];
    assert!((at - before).abs() < 1e-9, "equity moved {} across the settlement bar", at - before);
}

/// A dead leg is never sold, so it owes nothing on the way out.
///
/// The near leg expired; the far leg was genuinely traded out on a signal
/// and owes real brokerage. With the itemized schedule at 20 per order,
/// that is two entry orders and one exit order -- 60, not the 80 a
/// whole-position exit would bill.
#[test]
fn a_settled_leg_pays_no_exit_cost_when_the_survivor_is_traded_out() {
    let result = calendar(&[near_leg(0.0), far_leg(100.0)], Some(25), Some("NFO-OPT"));

    let trade = &result.trades[0];
    assert_eq!(trade.exit_reason, ExitReason::Signal, "the survivor was traded out");
    assert!(trade.exit_fees > 0.0, "the surviving leg placed a real order");

    let breakdown = trade.fee_breakdown.expect("itemized schedule was configured");
    assert!(
        (breakdown.brokerage - 60.0).abs() < 1e-9,
        "two entry orders and one exit order at 20 each; got {}",
        breakdown.brokerage
    );
}

/// The reported costs still add up after a leg has settled.
///
/// `fees == entry_fees + exit_fees` is a documented invariant on `Trade`,
/// and a partial settlement is exactly where a fee could go missing from
/// one side without the total noticing.
#[test]
fn fees_equal_entry_plus_exit_after_a_partial_settlement() {
    let result = calendar(&[near_leg(0.0), far_leg(100.0)], Some(25), Some("NFO-OPT"));

    let trade = &result.trades[0];
    assert!((trade.fees - (trade.entry_fees + trade.exit_fees)).abs() < 1e-9);
    let breakdown = trade.fee_breakdown.expect("itemized schedule was configured");
    assert!((breakdown.total() - trade.fees).abs() < 1e-9, "itemization must sum to the bill");
}

/// A settled leg is frozen at what it settled for.
///
/// The near leg's series carries nonsense from the bar after it expires --
/// a stale 999 quote on a contract that no longer exists. If the freeze
/// leaked, that quote would reach the exit price and the P&L. The exit
/// price must read the near leg at 0 and the far leg at 100:
/// (-1)(0)(75) + (+1)(100)(75) = 7500.
#[test]
fn the_exit_price_reports_the_settled_leg_at_its_settlement_value() {
    let result = calendar(&[near_leg(999.0), far_leg(100.0)], None, None);

    let trade = &result.trades[0];
    assert!(
        (trade.exit_price - 7500.0).abs() < 1e-9,
        "a dead leg's stale quote reached the exit price; got {}",
        trade.exit_price
    );
    assert!((trade.pnl - 5250.0).abs() < 1e-9, "and it must not reach the P&L either");
}

/// Both legs expiring together behaves exactly as it always has.
///
/// This is the guard on everything already trading: straddles, strangles,
/// verticals, iron condors. When every leg shares one expiry there is no
/// gap to settle across, and the whole structure must close on that bar
/// with no exit cost, as before.
#[test]
fn a_same_expiry_structure_is_unchanged() {
    let (near_premiums, _, _) = near_leg(0.0);
    let (far_premiums, _, _) = far_leg(100.0);
    let legs = vec![(near_premiums, -1, NEAR_EXPIRY_BAR), (far_premiums, 1, NEAR_EXPIRY_BAR)];
    let result = calendar(&legs, None, Some("NFO-OPT"));

    let trade = &result.trades[0];
    assert_eq!(trade.exit_idx, NEAR_EXPIRY_BAR, "both legs died on the same bar");
    assert_eq!(trade.exit_reason, ExitReason::Settlement);
    assert_eq!(trade.exit_fees, 0.0, "neither leg was traded out");
    // Near +3750 as before; far leg still at 80 on its expiry bar, so 0.
    assert!((trade.pnl - (3750.0 - trade.entry_fees)).abs() < 1e-9, "got {}", trade.pnl);
}

/// An expiry list that does not line up with the legs is refused.
///
/// Expiries are matched to legs by position. A short list would leave the
/// trailing legs immortal and a long one would settle on a date belonging
/// to no leg -- both silently. Refusing outright is the only safe answer,
/// and `empty_result` is how this engine refuses.
#[test]
fn a_mismatched_expiry_vector_is_rejected() {
    let n = BARS;
    let timestamps: Vec<i64> = (0..n as i64).map(|i| i * 300_000_000_000).collect();
    let mut entries = vec![false; n];
    entries[1] = true;

    let config = SpreadConfig {
        base: BacktestConfig { initial_capital: 500_000.0, ..Default::default() },
        spread_type: SpreadType::Custom,
        leg_configs: vec![
            LegConfig::new(OptionType::Call, STRIKE, -1, LOT),
            LegConfig::new(OptionType::Call, STRIKE, 1, LOT),
        ],
        // Two legs, one expiry.
        leg_expiry_timestamps: Some(vec![timestamps[NEAR_EXPIRY_BAR]]),
        ..Default::default()
    };

    let result = SpreadBacktest::new(config).run(
        &timestamps,
        &vec![24_550.0; n],
        &[vec![50.0; n], vec![80.0; n]],
        &entries,
        &vec![false; n],
    );

    assert!(result.trades.is_empty(), "a refused run trades nothing");
    assert_eq!(result.equity_curve, vec![500_000.0; n], "and its equity never moves");
}

/// What the trade says it made is what the account actually gained.
///
/// The account is credited in three separate places -- costs at the open,
/// each leg on its own expiry bar, and the survivors at the close. If any
/// of them counts a leg twice or drops one, the final equity and the
/// reported P&L stop agreeing. They must not.
#[test]
fn the_reported_pnl_equals_what_the_account_actually_gained() {
    let result = calendar(&[near_leg(999.0), far_leg(100.0)], None, Some("NFO-OPT"));

    let gained = result.equity_curve[BARS - 1] - 500_000.0;
    let trade = &result.trades[0];
    assert!(
        (gained - trade.pnl).abs() < 1e-9,
        "the account gained {gained} but the trade reports {}",
        trade.pnl
    );
}
