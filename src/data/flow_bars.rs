//! Information-driven bars: imbalance and runs.
//!
//! These sample by *signed order flow* rather than by time or gross volume,
//! so a bar closes when the market has revealed a set amount of directional
//! information. A quiet two-sided hour produces one bar; a burst of
//! one-sided buying produces several.
//!
//! Two close rules, each over three magnitudes (per trade, per unit of
//! volume, per unit of traded value):
//!
//! - **Imbalance** closes when net signed flow `|Σ bₜ·mₜ|` reaches the
//!   threshold. Two-sided flow cancels, so balanced trading never closes a
//!   bar however heavy it is.
//! - **Runs** closes when either side's own accumulation `max(Σ⁺, Σ⁻)`
//!   reaches it. Heavy two-sided flow *does* close a bar here, which is the
//!   distinction from imbalance.
//!
//! The threshold is `step`, fixed. López de Prado's formulation makes it
//! adaptive — an EWMA of expected imbalance, so bars carry roughly constant
//! information — but that is path-dependent on warmup and turns `step` into
//! an opaque tuning knob. A fixed threshold is deterministic, reproducible,
//! reads directly ("close every 10,000 shares of net imbalance"), and
//! matches how `step` already works for tick, volume, and value bars.
//!
//! Direction comes from the feed when it is known (`signed_volume`, from
//! buy/sell quantity deltas). Otherwise it falls back to the tick rule:
//! classify by price change against the previous record, carrying the
//! previous sign through an unchanged price. That is what makes these units
//! work over plain bars, which carry no flow data at all.

use crate::core::types::{OhlcvBar, Price};
use crate::data::aggregate::{BarBuilder, Partial, SourceRecord};
use crate::data::bar_spec::AggregationUnit;

/// What one unit of flow counts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlowMagnitude {
    /// One per record.
    Tick,
    /// Traded size.
    Volume,
    /// Traded size times price.
    Value,
}

/// How accumulated flow closes a bar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlowRule {
    /// Net signed flow: two-sided trading cancels.
    Imbalance,
    /// Larger one-sided accumulation: two-sided trading still closes a bar.
    Runs,
}

/// Imbalance and runs bars over signed order flow.
#[derive(Debug)]
pub struct FlowBarBuilder {
    magnitude: FlowMagnitude,
    rule: FlowRule,
    threshold: f64,
    partial: Option<Partial>,
    /// Accumulated buy-side and sell-side magnitude in the open bar.
    buy: f64,
    sell: f64,
    /// Previous close and sign, for the tick rule.
    prev_close: Option<Price>,
    prev_sign: f64,
    /// Timestamp of the last record absorbed, for stamping a flush.
    last_ts: i64,
}

impl FlowBarBuilder {
    pub fn new(magnitude: FlowMagnitude, rule: FlowRule, threshold: f64) -> Self {
        debug_assert!(threshold > 0.0, "threshold must be positive");
        Self {
            magnitude,
            rule,
            threshold,
            partial: None,
            buy: 0.0,
            sell: 0.0,
            prev_close: None,
            prev_sign: 1.0,
            last_ts: 0,
        }
    }

    /// Builder for one of the six signed-flow units, if this is one.
    pub fn for_unit(unit: AggregationUnit, threshold: f64) -> Option<Self> {
        let (magnitude, rule) = match unit {
            AggregationUnit::TickImbalance => (FlowMagnitude::Tick, FlowRule::Imbalance),
            AggregationUnit::TickRuns => (FlowMagnitude::Tick, FlowRule::Runs),
            AggregationUnit::VolumeImbalance => (FlowMagnitude::Volume, FlowRule::Imbalance),
            AggregationUnit::VolumeRuns => (FlowMagnitude::Volume, FlowRule::Runs),
            AggregationUnit::ValueImbalance => (FlowMagnitude::Value, FlowRule::Imbalance),
            AggregationUnit::ValueRuns => (FlowMagnitude::Value, FlowRule::Runs),
            _ => return None,
        };
        Some(Self::new(magnitude, rule, threshold))
    }

    /// Trade direction: the feed's split when known, else the tick rule.
    fn sign_of(&mut self, rec: &SourceRecord) -> f64 {
        if rec.signed_volume != 0.0 {
            return rec.signed_volume.signum();
        }
        let sign = match self.prev_close {
            Some(prev) if rec.close > prev => 1.0,
            Some(prev) if rec.close < prev => -1.0,
            // An unchanged price carries the prior direction through.
            Some(_) => self.prev_sign,
            None => 1.0,
        };
        self.prev_sign = sign;
        sign
    }

    fn magnitude_of(&self, rec: &SourceRecord) -> f64 {
        match self.magnitude {
            FlowMagnitude::Tick => 1.0,
            FlowMagnitude::Volume => rec.volume,
            FlowMagnitude::Value => rec.close * rec.volume,
        }
    }

    fn is_complete(&self) -> bool {
        match self.rule {
            FlowRule::Imbalance => (self.buy - self.sell).abs() >= self.threshold,
            FlowRule::Runs => self.buy.max(self.sell) >= self.threshold,
        }
    }

    fn reset(&mut self) {
        self.buy = 0.0;
        self.sell = 0.0;
    }
}

impl BarBuilder for FlowBarBuilder {
    fn push(&mut self, rec: &SourceRecord) -> Option<OhlcvBar> {
        let sign = self.sign_of(rec);
        self.prev_close = Some(rec.close);
        self.last_ts = rec.timestamp;
        let magnitude = self.magnitude_of(rec);
        if sign >= 0.0 {
            self.buy += magnitude;
        } else {
            self.sell += magnitude;
        }

        match &mut self.partial {
            Some(partial) => partial.absorb(rec),
            None => self.partial = Some(Partial::start(rec)),
        }

        // The record that crosses the threshold belongs to the bar it
        // completed, matching the count-based builders.
        if self.is_complete() {
            self.reset();
            return self.partial.take().map(|p| p.into_bar(rec.timestamp));
        }
        None
    }

    fn flush(&mut self) -> Option<OhlcvBar> {
        self.reset();
        // Stamped at the last record absorbed, so a partial bar at end of
        // data does not claim to cover time it never saw.
        let ts = self.last_ts;
        self.partial.take().map(|p| p.into_bar(ts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signed(ts: i64, price: Price, size: f64, signed: f64) -> SourceRecord {
        SourceRecord::signed_trade(ts, price, size, signed)
    }

    fn unsigned(ts: i64, price: Price, size: f64) -> SourceRecord {
        SourceRecord::trade(ts, price, size)
    }

    #[test]
    fn tick_imbalance_closes_on_net_signed_count() {
        let mut b = FlowBarBuilder::new(FlowMagnitude::Tick, FlowRule::Imbalance, 3.0);
        assert!(b.push(&signed(0, 100.0, 1.0, 1.0)).is_none());
        assert!(b.push(&signed(1, 100.0, 1.0, 1.0)).is_none());
        let bar = b.push(&signed(2, 100.0, 1.0, 1.0)).expect("three net buys");
        assert_eq!(bar.timestamp, 2);
    }

    #[test]
    fn balanced_flow_never_closes_an_imbalance_bar() {
        // The distinction from a volume bar: gross size does not matter.
        let mut b = FlowBarBuilder::new(FlowMagnitude::Volume, FlowRule::Imbalance, 10.0);
        for i in 0..20 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            assert!(
                b.push(&signed(i, 100.0, 5.0, sign)).is_none(),
                "two-sided flow cancels, however heavy"
            );
        }
    }

    #[test]
    fn balanced_flow_does_close_a_runs_bar() {
        // The same tape that never closes an imbalance bar closes this one.
        let mut b = FlowBarBuilder::new(FlowMagnitude::Volume, FlowRule::Runs, 10.0);
        let mut closed = 0;
        for i in 0..20 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            if b.push(&signed(i, 100.0, 5.0, sign)).is_some() {
                closed += 1;
            }
        }
        assert!(closed > 0, "one-sided accumulation reaches the threshold");
    }

    #[test]
    fn imbalance_and_runs_agree_on_one_directional_flow() {
        // With no opposing flow, net and one-sided accumulation are equal.
        let mut imbalance = FlowBarBuilder::new(FlowMagnitude::Volume, FlowRule::Imbalance, 10.0);
        let mut runs = FlowBarBuilder::new(FlowMagnitude::Volume, FlowRule::Runs, 10.0);
        let mut a = Vec::new();
        let mut b = Vec::new();
        for i in 0..10 {
            if let Some(bar) = imbalance.push(&signed(i, 100.0, 4.0, 1.0)) {
                a.push(bar.timestamp);
            }
            if let Some(bar) = runs.push(&signed(i, 100.0, 4.0, 1.0)) {
                b.push(bar.timestamp);
            }
        }
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }

    #[test]
    fn value_scales_magnitude_by_price() {
        // 100 * 1 = 100 of value on one record clears a threshold that the
        // same size at a lower price would not.
        let mut b = FlowBarBuilder::new(FlowMagnitude::Value, FlowRule::Imbalance, 100.0);
        assert!(b.push(&signed(0, 10.0, 1.0, 1.0)).is_none(), "10 of value");
        assert!(b.push(&signed(1, 100.0, 1.0, 1.0)).is_some(), "110 of value");
    }

    #[test]
    fn the_tick_rule_signs_unsigned_records() {
        // Rising prices read as buys, falling as sells.
        let mut b = FlowBarBuilder::new(FlowMagnitude::Tick, FlowRule::Imbalance, 3.0);
        assert!(b.push(&unsigned(0, 100.0, 1.0)).is_none()); // first: buy
        assert!(b.push(&unsigned(1, 101.0, 1.0)).is_none()); // up: buy
        assert!(b.push(&unsigned(2, 102.0, 1.0)).is_some(), "three up-ticks");
    }

    #[test]
    fn an_unchanged_price_carries_the_previous_direction() {
        let mut b = FlowBarBuilder::new(FlowMagnitude::Tick, FlowRule::Imbalance, 100.0);
        b.push(&unsigned(0, 100.0, 1.0));
        b.push(&unsigned(1, 99.0, 1.0)); // down: sell
        b.push(&unsigned(2, 99.0, 1.0)); // unchanged: still a sell
        assert_eq!(b.sell, 2.0, "the zero tick inherited the down direction");
    }

    #[test]
    fn the_feed_split_wins_over_the_tick_rule() {
        // Price rose, but the feed says the volume was sell-initiated.
        let mut b = FlowBarBuilder::new(FlowMagnitude::Volume, FlowRule::Imbalance, 100.0);
        b.push(&signed(0, 100.0, 5.0, 1.0));
        b.push(&signed(1, 101.0, 5.0, -5.0));
        assert_eq!(b.sell, 5.0, "the explicit split is authoritative");
    }

    #[test]
    fn a_closed_bar_summarizes_its_records() {
        let mut b = FlowBarBuilder::new(FlowMagnitude::Tick, FlowRule::Imbalance, 2.0);
        b.push(&signed(0, 100.0, 3.0, 1.0));
        let bar = b.push(&signed(1, 105.0, 4.0, 1.0)).expect("two net buys");
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.close, 105.0);
        assert_eq!(bar.high, 105.0);
        assert_eq!(bar.low, 100.0);
        assert_eq!(bar.volume, 7.0);
    }

    #[test]
    fn accumulation_resets_after_a_bar_closes() {
        let mut b = FlowBarBuilder::new(FlowMagnitude::Tick, FlowRule::Imbalance, 2.0);
        b.push(&signed(0, 100.0, 1.0, 1.0));
        assert!(b.push(&signed(1, 100.0, 1.0, 1.0)).is_some());
        assert_eq!(b.buy, 0.0);
        assert!(b.push(&signed(2, 100.0, 1.0, 1.0)).is_none(), "counting restarts");
    }

    #[test]
    fn for_unit_maps_every_signed_flow_variant() {
        use AggregationUnit::*;
        for (unit, magnitude, rule) in [
            (TickImbalance, FlowMagnitude::Tick, FlowRule::Imbalance),
            (TickRuns, FlowMagnitude::Tick, FlowRule::Runs),
            (VolumeImbalance, FlowMagnitude::Volume, FlowRule::Imbalance),
            (VolumeRuns, FlowMagnitude::Volume, FlowRule::Runs),
            (ValueImbalance, FlowMagnitude::Value, FlowRule::Imbalance),
            (ValueRuns, FlowMagnitude::Value, FlowRule::Runs),
        ] {
            let b = FlowBarBuilder::for_unit(unit, 1.0).expect("a signed-flow unit");
            assert_eq!(b.magnitude, magnitude);
            assert_eq!(b.rule, rule);
        }
        assert!(FlowBarBuilder::for_unit(AggregationUnit::Minute, 1.0).is_none());
    }
}
