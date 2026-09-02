//! Position-group margin for options that share an underlying and expiry.
//!
//! In plain words: a sold put protected by a bought put can only lose the
//! gap between them, and an exchange charges the pair far less than two
//! naked deposits. Charging each sold leg its full deposit refuses trades a
//! real account carries easily, so once legs on one underlying and expiry
//! are open together the session re-prices the SOLD legs as one group.
//!
//! The shape follows what an Indian broker's basket margin actually
//! returns for one-lot structures (measured 2026-09-02): a risk-scenario
//! component that is charged ONCE per group — the full scenario deposit on
//! the largest sold leg when some short side is uncovered, and only the
//! structure's intrinsic worst loss when every short is covered — plus an
//! exposure component on every sold leg's notional, less the net premium
//! the group collected. The result never drops below the structure's own
//! worst loss net of premium, and never below zero.
//!
//! Only sold legs are re-priced. A bought leg's premium is its cost and
//! stays locked as the kernel funded it. Sizing of a NEW sold leg still
//! uses its naked deposit — the group benefit arrives once the leg is on,
//! freeing capital for later entries and lowering the maintenance
//! requirement. That is the conservative order: a leg must be carriable on
//! its own before it may lean on its hedge.

use crate::core::types::Direction;
use crate::instruments::OptionRight;

/// One open option position as the grouping sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct OptionLeg {
    pub position_id: u64,
    pub kernel: usize,
    pub strike: f64,
    pub right: OptionRight,
    pub direction: Direction,
    pub size: f64,
    pub entry_price: f64,
    pub multiplier: f64,
    pub span_pct: f64,
    pub exposure_pct: f64,
}

impl OptionLeg {
    fn notional(&self) -> f64 {
        self.strike * self.size * self.multiplier
    }

    fn premium(&self) -> f64 {
        self.entry_price * self.size * self.multiplier
    }

    /// The deposit this leg would carry on its own.
    pub fn naked_deposit(&self) -> f64 {
        (self.span_pct + self.exposure_pct) * self.notional()
    }

    fn intrinsic_at(&self, spot: f64) -> f64 {
        let intrinsic = match self.right {
            OptionRight::Call => (spot - self.strike).max(0.0),
            OptionRight::Put => (self.strike - spot).max(0.0),
        };
        let sign = match self.direction {
            Direction::Long => 1.0,
            Direction::Short => -1.0,
        };
        sign * intrinsic * self.size * self.multiplier
    }
}

/// What one group of legs must keep locked, with the terms that made it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupRequirement {
    /// Locked margin for the group's SOLD legs together.
    pub total: f64,
    /// Worst intrinsic loss of the whole structure at expiry, before premium.
    pub intrinsic_worst_loss: f64,
    /// Scenario component: the largest sold leg's deposit when a short side
    /// is uncovered, else the intrinsic worst loss.
    pub span_component: f64,
    /// Exposure component over every sold leg's notional.
    pub exposure: f64,
    /// Premium collected on sold legs less premium paid on bought legs.
    pub net_premium: f64,
    /// Whether the structure can lose without limit on some side.
    pub uncovered: bool,
}

/// The group's requirement, or `None` when it holds no sold leg.
pub fn group_requirement(legs: &[OptionLeg]) -> Option<GroupRequirement> {
    let shorts: Vec<&OptionLeg> = legs.iter().filter(|l| l.direction == Direction::Short).collect();
    if shorts.is_empty() {
        return None;
    }

    // Piecewise-linear payoff: the minimum sits at zero, at a strike, or
    // runs off to one side. Evaluate at every kink plus a far spot.
    let far = legs.iter().map(|l| l.strike).fold(0.0_f64, f64::max) * 2.0 + 1.0;
    let mut spots: Vec<f64> = legs.iter().map(|l| l.strike).collect();
    spots.push(0.0);
    spots.push(far);
    let worst = spots
        .iter()
        .map(|&s| legs.iter().map(|l| l.intrinsic_at(s)).sum::<f64>())
        .fold(0.0_f64, f64::min);
    let intrinsic_worst_loss = (-worst).max(0.0);

    let side_size = |right: OptionRight, direction: Direction| -> f64 {
        legs.iter()
            .filter(|l| l.right == right && l.direction == direction)
            .map(|l| l.size * l.multiplier)
            .sum()
    };
    let uncovered = side_size(OptionRight::Call, Direction::Short)
        > side_size(OptionRight::Call, Direction::Long)
        || side_size(OptionRight::Put, Direction::Short)
            > side_size(OptionRight::Put, Direction::Long);

    let span_pct = shorts.iter().map(|l| l.span_pct).fold(0.0_f64, f64::max);
    let span_component = if uncovered {
        span_pct * shorts.iter().map(|l| l.notional()).fold(0.0_f64, f64::max)
    } else {
        intrinsic_worst_loss
    };
    let exposure: f64 = shorts.iter().map(|l| l.exposure_pct * l.notional()).sum();
    // The broker's basket figure nets the premium a multi-leg group
    // collected; its single-order figure for a lone sold leg does not. Keep
    // both shapes: a lone leg stays at its full deposit, a group nets.
    let net_premium: f64 = if legs.len() >= 2 {
        legs.iter()
            .map(|l| match l.direction {
                Direction::Short => l.premium(),
                Direction::Long => -l.premium(),
            })
            .sum()
    } else {
        0.0
    };

    // A covered structure can never lose more than its worst intrinsic
    // loss, so that (net of premium) is its floor. An uncovered one has no
    // such bound — its worst loss runs off to infinity — so the scenario
    // deposit is the whole requirement.
    let scenario = (span_component + exposure - net_premium).max(0.0);
    let total = if uncovered {
        scenario
    } else {
        scenario.max((intrinsic_worst_loss - net_premium).max(0.0))
    };
    Some(GroupRequirement {
        total,
        intrinsic_worst_loss,
        span_component,
        exposure,
        net_premium,
        uncovered,
    })
}

/// Split a group total across its sold legs in proportion to what each
/// would carry alone, so a leg's share moves with its own size.
pub fn apportion(legs: &[OptionLeg], total: f64) -> Vec<(usize, u64, f64)> {
    let shorts: Vec<&OptionLeg> = legs.iter().filter(|l| l.direction == Direction::Short).collect();
    let weight_sum: f64 = shorts.iter().map(|l| l.naked_deposit()).sum();
    shorts
        .iter()
        .map(|l| {
            let share = if weight_sum > 0.0 {
                total * l.naked_deposit() / weight_sum
            } else {
                total / shorts.len() as f64
            };
            (l.kernel, l.position_id, share)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPAN: f64 = 0.0975;
    const EXPOSURE: f64 = 0.02;

    fn leg(
        id: u64,
        strike: f64,
        right: OptionRight,
        direction: Direction,
        size: f64,
        premium: f64,
    ) -> OptionLeg {
        OptionLeg {
            position_id: id,
            kernel: id as usize,
            strike,
            right,
            direction,
            size,
            entry_price: premium,
            multiplier: 1.0,
            span_pct: SPAN,
            exposure_pct: EXPOSURE,
        }
    }

    fn within(actual: f64, measured: f64, tolerance: f64) -> bool {
        (actual - measured).abs() / measured <= tolerance
    }

    // The four broker figures these tests are held to (one lot each,
    // measured 2026-09-02, MIS and NRML identical):
    //   BANKNIFTY 57000 straddle sold, lot 30, spot 57,071: 1,88,616
    //   NIFTY 23850/23800 bull put spread, lot 65:            32,876
    //   NIFTY 23700/23800 P + 23900/24000 C condor, lot 65:   63,107
    //   BANKNIFTY 57000 CE sold alone, lot 30:               2,01,216

    #[test]
    fn a_lone_sold_leg_keeps_its_full_deposit() {
        let legs = [leg(1, 57_000.0, OptionRight::Call, Direction::Short, 30.0, 1_006.15)];
        let req = group_requirement(&legs).unwrap();
        assert!(req.uncovered);
        // span + exposure, no premium credit: the broker's single-order figure.
        let expected = (SPAN + EXPOSURE) * 57_000.0 * 30.0;
        assert!((req.total - expected).abs() < 1e-6, "{} vs {expected}", req.total);
        assert!(within(req.total, 201_216.0, 0.05), "{} vs broker 2,01,216", req.total);
    }

    #[test]
    fn a_short_straddle_pays_the_scenario_deposit_once() {
        let legs = [
            leg(1, 57_000.0, OptionRight::Call, Direction::Short, 30.0, 1_006.15),
            leg(2, 57_000.0, OptionRight::Put, Direction::Short, 30.0, 551.05),
        ];
        let req = group_requirement(&legs).unwrap();
        assert!(req.uncovered);
        let naked_sum: f64 = legs.iter().map(OptionLeg::naked_deposit).sum();
        assert!(req.total < naked_sum, "the pair must cost less than two naked deposits");
        assert!(within(req.total, 188_616.0, 0.05), "straddle {} vs broker 1,88,616", req.total);
    }

    #[test]
    fn a_bull_put_spread_is_exposure_plus_the_width_less_the_credit() {
        let legs = [
            leg(1, 23_850.0, OptionRight::Put, Direction::Short, 65.0, 120.0),
            leg(2, 23_800.0, OptionRight::Put, Direction::Long, 65.0, 102.05),
        ];
        let req = group_requirement(&legs).unwrap();
        assert!(!req.uncovered);
        assert!((req.intrinsic_worst_loss - 50.0 * 65.0).abs() < 1e-6);
        assert!(within(req.total, 32_876.0, 0.05), "vertical {} vs broker 32,876", req.total);
        // And nowhere near the naked deposit on the sold leg.
        assert!(req.total < legs[0].naked_deposit() / 4.0);
    }

    #[test]
    fn an_iron_condor_pays_one_wing_width_plus_exposure_on_both_shorts() {
        let legs = [
            leg(1, 23_700.0, OptionRight::Put, Direction::Long, 65.0, 60.0),
            leg(2, 23_800.0, OptionRight::Put, Direction::Short, 65.0, 102.0),
            leg(3, 23_900.0, OptionRight::Call, Direction::Short, 65.0, 110.0),
            leg(4, 24_000.0, OptionRight::Call, Direction::Long, 65.0, 64.05),
        ];
        let req = group_requirement(&legs).unwrap();
        assert!(!req.uncovered);
        assert!((req.intrinsic_worst_loss - 100.0 * 65.0).abs() < 1e-6);
        assert!(within(req.total, 63_107.0, 0.05), "condor {} vs broker 63,107", req.total);
    }

    #[test]
    fn losing_the_long_wing_makes_the_short_naked_again() {
        let covered = [
            leg(1, 23_850.0, OptionRight::Put, Direction::Short, 65.0, 120.0),
            leg(2, 23_800.0, OptionRight::Put, Direction::Long, 65.0, 102.05),
        ];
        let naked = [covered[0].clone()];
        let with_wing = group_requirement(&covered).unwrap().total;
        let without = group_requirement(&naked).unwrap().total;
        assert!(without > with_wing * 4.0, "naked {without} should dwarf covered {with_wing}");
    }

    #[test]
    fn the_group_never_charges_less_than_its_worst_loss_net_of_premium() {
        // Absurd percentages of zero: the floor still holds.
        let mut legs = [
            leg(1, 100.0, OptionRight::Put, Direction::Short, 10.0, 1.0),
            leg(2, 90.0, OptionRight::Put, Direction::Long, 10.0, 0.5),
        ];
        for l in legs.iter_mut() {
            l.span_pct = 0.0;
            l.exposure_pct = 0.0;
        }
        let req = group_requirement(&legs).unwrap();
        assert!((req.total - (100.0 - 5.0)).abs() < 1e-9, "{}", req.total);
    }

    #[test]
    fn only_bought_legs_means_nothing_to_regroup() {
        let legs = [leg(1, 100.0, OptionRight::Call, Direction::Long, 10.0, 2.0)];
        assert!(group_requirement(&legs).is_none());
    }

    #[test]
    fn apportioning_follows_each_sold_legs_own_deposit() {
        let legs = [
            leg(1, 57_000.0, OptionRight::Call, Direction::Short, 30.0, 1_006.15),
            leg(2, 57_000.0, OptionRight::Put, Direction::Short, 60.0, 551.05),
            leg(3, 56_000.0, OptionRight::Put, Direction::Long, 60.0, 300.0),
        ];
        let shares = apportion(&legs, 90_000.0);
        assert_eq!(shares.len(), 2);
        let total: f64 = shares.iter().map(|s| s.2).sum();
        assert!((total - 90_000.0).abs() < 1e-6);
        // The 60-lot put carries twice the 30-lot call's share (same strike
        // notional per contract, twice the size... strikes equal here).
        assert!((shares[1].2 - 2.0 * shares[0].2).abs() < 1e-6);
    }
}
