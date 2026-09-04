//! Multi-position ledger.
//!
//! Generalizes the one-position assumption the kernel was built on. Two
//! policies:
//!
//! - [`PositionPolicy::Net`] — at most one open position, opened in the
//!   kernel's direction. This is the historical behavior; every arithmetic
//!   step matches the original [`PositionManager`] path bit-for-bit (the
//!   golden fixture suite enforces it).
//! - [`PositionPolicy::Independent`] — hedging: each opening order creates
//!   its own entry with its own direction, protective levels, and running
//!   extremes; longs and shorts coexist. Closes target a position id.
//!
//! [`PositionManager`]: crate::portfolio::position::PositionManager

use crate::core::types::{Direction, Position, Price, Timestamp, Trade};
use crate::execution::indian_costs::FeeBreakdown;
use crate::portfolio::position::ExitDetails;

/// How the ledger treats additional opening fills.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PositionPolicy {
    /// One net position at a time (historical behavior).
    #[default]
    Net,
    /// Independent concurrent positions, both directions (hedging).
    Independent,
}

/// One open position plus the bookkeeping the kernel used to hold globally.
#[derive(Debug, Clone)]
pub struct ManagedPosition {
    /// Ledger-assigned id, unique within a session.
    pub id: u64,
    /// Price/size/protective state (shared struct with the legacy path).
    pub position: Position,
    /// Entry timestamp, carried onto the trade record.
    pub entry_timestamp: Timestamp,
    /// Itemized entry costs, combined with exit costs at close.
    pub entry_breakdown: Option<FeeBreakdown>,
    /// The typed order that opened this position, when one did; a later
    /// slice of the same order adds to it instead of being refused as a
    /// second entry.
    pub entry_order_id: Option<u64>,
}

impl ManagedPosition {
    /// Whether this position's stop level is touched by the bar range.
    pub fn is_stop_hit(&self, low: Price, high: Price) -> bool {
        match self.position.stop_price {
            Some(stop) => match self.position.direction {
                Direction::Long => low <= stop,
                Direction::Short => high >= stop,
            },
            None => false,
        }
    }

    /// Whether this position's target level is touched by the bar range.
    pub fn is_target_hit(&self, low: Price, high: Price) -> bool {
        match self.position.target_price {
            Some(target) => match self.position.direction {
                Direction::Long => high >= target,
                Direction::Short => low <= target,
            },
            None => false,
        }
    }

    /// Ratchet a percent trailing stop off the running extreme.
    pub fn update_trailing_stop(&mut self, trail_percent: f64) {
        let new_stop = match self.position.direction {
            Direction::Long => self.position.highest_since_entry * (1.0 - trail_percent),
            Direction::Short => self.position.lowest_since_entry * (1.0 + trail_percent),
        };
        let improves = match (self.position.stop_price, self.position.direction) {
            (None, _) => true,
            (Some(current), Direction::Long) => new_stop > current,
            (Some(current), Direction::Short) => new_stop < current,
        };
        if improves {
            self.position.stop_price = Some(new_stop);
        }
    }
}

/// Open positions and trade-record bookkeeping for one instrument.
#[derive(Debug)]
pub struct PositionLedger {
    policy: PositionPolicy,
    symbol: String,
    open: Vec<ManagedPosition>,
    trade_counter: u64,
    next_position_id: u64,
    /// Contract point value; see `PositionManager::set_contract_multiplier`.
    contract_multiplier: f64,
}

impl PositionLedger {
    pub fn new(symbol: String, policy: PositionPolicy) -> Self {
        Self {
            policy,
            symbol,
            open: Vec::new(),
            trade_counter: 0,
            next_position_id: 0,
            contract_multiplier: 1.0,
        }
    }

    /// Set the contract point value used for PnL and notional calculations.
    pub fn set_contract_multiplier(&mut self, multiplier: f64) {
        self.contract_multiplier = if multiplier > 0.0 { multiplier } else { 1.0 };
    }

    #[inline]
    pub fn policy(&self) -> PositionPolicy {
        self.policy
    }

    /// Symbol this ledger tracks.
    #[inline]
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    #[inline]
    pub fn contract_multiplier(&self) -> f64 {
        self.contract_multiplier
    }

    /// Whether any position is open.
    #[inline]
    pub fn is_in_position(&self) -> bool {
        !self.open.is_empty()
    }

    /// Number of open positions.
    #[inline]
    pub fn open_count(&self) -> usize {
        self.open.len()
    }

    /// The earliest-opened position — the legacy single-position view.
    #[inline]
    pub fn first(&self) -> Option<&ManagedPosition> {
        self.open.first()
    }

    /// Mutable view of the earliest-opened position.
    #[inline]
    pub fn first_mut(&mut self) -> Option<&mut ManagedPosition> {
        self.open.first_mut()
    }

    /// All open positions, in opening order.
    #[inline]
    pub fn positions(&self) -> &[ManagedPosition] {
        &self.open
    }

    /// Mutable iteration over open positions, in opening order.
    #[inline]
    pub fn positions_mut(&mut self) -> impl Iterator<Item = &mut ManagedPosition> {
        self.open.iter_mut()
    }

    /// A position by ledger id.
    pub fn get(&self, id: u64) -> Option<&ManagedPosition> {
        self.open.iter().find(|p| p.id == id)
    }

    /// Mutable view of a position by ledger id.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut ManagedPosition> {
        self.open.iter_mut().find(|p| p.id == id)
    }

    /// Open a position; returns its ledger id, or `None` when the Net
    /// policy already holds one.
    #[allow(clippy::too_many_arguments)]
    pub fn open_position(
        &mut self,
        idx: usize,
        timestamp: Timestamp,
        price: Price,
        size: f64,
        direction: Direction,
        stop_price: Option<Price>,
        target_price: Option<Price>,
        entry_fees: f64,
        entry_breakdown: Option<FeeBreakdown>,
    ) -> Option<u64> {
        if self.policy == PositionPolicy::Net && !self.open.is_empty() {
            return None;
        }
        let mut position = Position::new();
        position.open(idx, price, size, direction, stop_price, target_price, entry_fees);
        let id = self.next_position_id;
        self.next_position_id += 1;
        self.open.push(ManagedPosition {
            id,
            position,
            entry_timestamp: timestamp,
            entry_breakdown,
            entry_order_id: None,
        });
        Some(id)
    }

    /// Add a fill to an open position: size grows, the entry price becomes
    /// the size-weighted average, entry fees accumulate. When
    /// `shift_protective` is set, stop and target move by the change in
    /// average entry so a level derived from the entry price (a percent or
    /// ATR stop) keeps its distance; explicit levels are left alone by the
    /// caller passing `false`. Returns the new average entry price.
    pub fn add_to_position(
        &mut self,
        id: u64,
        price: Price,
        size: f64,
        entry_fees: f64,
        shift_protective: bool,
    ) -> Option<Price> {
        let managed = self.get_mut(id)?;
        let pos = &mut managed.position;
        let old_avg = pos.entry_price;
        let total = pos.size + size;
        let new_avg = (pos.entry_price * pos.size + price * size) / total;
        pos.entry_price = new_avg;
        pos.size = total;
        pos.entry_fees += entry_fees;
        if shift_protective {
            let shift = new_avg - old_avg;
            pos.stop_price = pos.stop_price.map(|s| s + shift);
            pos.target_price = pos.target_price.map(|t| t + shift);
        }
        // The itemized entry breakdown describes the first slice only;
        // later slices carry their cost in `entry_fees` and the trade's
        // fee_breakdown is dropped rather than misreported.
        managed.entry_breakdown = None;
        Some(new_avg)
    }

    /// Close part of a position: a trade record for the `qty` slice (at the
    /// position's average entry, entry fees prorated) and the remainder
    /// stays open. `qty >= size` closes it whole via
    /// [`PositionLedger::close_position`].
    pub fn reduce_position(&mut self, id: u64, qty: f64, exit: ExitDetails) -> Option<Trade> {
        let managed = self.get_mut(id)?;
        if qty >= managed.position.size {
            return self.close_position(id, exit);
        }
        let fraction = qty / managed.position.size;
        let mut slice = managed.clone();
        slice.position.size = qty;
        slice.position.entry_fees = managed.position.entry_fees * fraction;
        slice.entry_breakdown = None;
        managed.position.size -= qty;
        managed.position.entry_fees -= slice.position.entry_fees;
        managed.entry_breakdown = None;
        let trade = self.create_trade(&slice, exit);
        self.trade_counter += 1;
        Some(trade)
    }

    /// Close a position by id and produce its trade record.
    ///
    /// Trade ids keep the legacy numbering: sequential in close order.
    pub fn close_position(&mut self, id: u64, exit: ExitDetails) -> Option<Trade> {
        let index = self.open.iter().position(|p| p.id == id)?;
        let managed = self.open.remove(index);
        let trade = self.create_trade(&managed, exit);
        self.trade_counter += 1;
        Some(trade)
    }

    /// Trade record arithmetic — identical to the legacy manager's.
    fn create_trade(&self, managed: &ManagedPosition, exit: ExitDetails) -> Trade {
        let ExitDetails {
            idx: exit_idx,
            timestamp: exit_timestamp,
            price: exit_price,
            entry_timestamp,
            reason: exit_reason,
            fees: exit_fees,
            fee_breakdown,
        } = exit;

        let pos = &managed.position;
        let multiplier = pos.direction.multiplier() * self.contract_multiplier;

        let gross_pnl = (exit_price - pos.entry_price) * pos.size * multiplier;
        let total_fees = pos.entry_fees + exit_fees;
        let pnl = gross_pnl - total_fees;

        let cost_basis = pos.entry_price * pos.size * self.contract_multiplier;
        let return_pct = if cost_basis > 0.0 { pnl / cost_basis * 100.0 } else { 0.0 };

        Trade {
            id: self.trade_counter,
            symbol: self.symbol.clone(),
            entry_idx: pos.entry_idx,
            exit_idx,
            entry_price: pos.entry_price,
            exit_price,
            size: pos.size,
            direction: pos.direction,
            pnl,
            return_pct,
            entry_time: entry_timestamp,
            exit_time: exit_timestamp,
            fees: total_fees,
            entry_fees: pos.entry_fees,
            exit_fees,
            fee_breakdown,
            exit_reason,
        }
    }

    /// Track bar extremes on every open position (for trailing stops).
    pub fn update_price(&mut self, high: Price, low: Price) {
        for managed in &mut self.open {
            managed.position.update_extremes(high, low);
        }
    }

    /// Direction-aware unrealized PnL across open positions.
    pub fn unrealized_total(&self, price: Price) -> f64 {
        self.open.iter().map(|p| p.position.unrealized_pnl(price) * self.contract_multiplier).sum()
    }

    /// Notional value of open positions at the given price (unsigned).
    pub fn notional_total(&self, price: Price) -> f64 {
        self.open.iter().map(|p| price * p.position.size * self.contract_multiplier).sum()
    }

    /// Total market value of open positions at the given price.
    ///
    /// `price * size` for every direction — the fully-funded model the
    /// engine has always used (shorts included; the golden suite pins it).
    /// Direction-aware marking arrives with the margin account layer, which
    /// owns short cash-flow properly.
    pub fn position_value(&self, price: Price) -> f64 {
        self.open.iter().map(|p| price * p.position.size * self.contract_multiplier).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::ExitReason;

    fn exit(idx: usize, price: Price) -> ExitDetails {
        ExitDetails {
            idx,
            timestamp: idx as i64,
            price,
            entry_timestamp: 0,
            reason: ExitReason::Signal,
            fees: 0.0,
            fee_breakdown: None,
        }
    }

    #[test]
    fn net_policy_refuses_second_position() {
        let mut ledger = PositionLedger::new("T".into(), PositionPolicy::Net);
        let a = ledger.open_position(0, 0, 100.0, 10.0, Direction::Long, None, None, 0.0, None);
        assert!(a.is_some());
        let b = ledger.open_position(1, 1, 101.0, 10.0, Direction::Long, None, None, 0.0, None);
        assert!(b.is_none());
        assert_eq!(ledger.open_count(), 1);
    }

    #[test]
    fn independent_policy_holds_both_directions() {
        let mut ledger = PositionLedger::new("T".into(), PositionPolicy::Independent);
        let long = ledger
            .open_position(0, 0, 100.0, 10.0, Direction::Long, None, None, 0.0, None)
            .unwrap();
        let short = ledger
            .open_position(1, 1, 102.0, 5.0, Direction::Short, None, None, 0.0, None)
            .unwrap();
        assert_eq!(ledger.open_count(), 2);
        assert_ne!(long, short);

        // Close the short at a profit; the long stays open.
        let trade = ledger.close_position(short, exit(2, 98.0)).unwrap();
        assert!((trade.pnl - (102.0 - 98.0) * 5.0).abs() < 1e-9);
        assert_eq!(ledger.open_count(), 1);
        assert_eq!(ledger.first().unwrap().id, long);
    }

    #[test]
    fn trade_ids_are_sequential_in_close_order() {
        let mut ledger = PositionLedger::new("T".into(), PositionPolicy::Independent);
        let a =
            ledger.open_position(0, 0, 100.0, 1.0, Direction::Long, None, None, 0.0, None).unwrap();
        let b =
            ledger.open_position(0, 0, 100.0, 1.0, Direction::Long, None, None, 0.0, None).unwrap();
        // Close b first: it takes trade id 0.
        assert_eq!(ledger.close_position(b, exit(1, 101.0)).unwrap().id, 0);
        assert_eq!(ledger.close_position(a, exit(2, 102.0)).unwrap().id, 1);
    }

    #[test]
    fn per_position_stops_and_trailing() {
        let mut ledger = PositionLedger::new("T".into(), PositionPolicy::Independent);
        let long = ledger
            .open_position(0, 0, 100.0, 1.0, Direction::Long, Some(95.0), None, 0.0, None)
            .unwrap();
        let short = ledger
            .open_position(0, 0, 100.0, 1.0, Direction::Short, Some(105.0), None, 0.0, None)
            .unwrap();

        // Bar range 97..103 hits neither stop.
        assert!(!ledger.get(long).unwrap().is_stop_hit(97.0, 103.0));
        assert!(!ledger.get(short).unwrap().is_stop_hit(97.0, 103.0));
        // 94 low hits the long's stop only.
        assert!(ledger.get(long).unwrap().is_stop_hit(94.0, 103.0));
        assert!(!ledger.get(short).unwrap().is_stop_hit(94.0, 103.0));

        // Trailing ratchets each side toward its own extreme.
        ledger.update_price(110.0, 94.0);
        ledger.get_mut(long).unwrap().update_trailing_stop(0.05);
        ledger.get_mut(short).unwrap().update_trailing_stop(0.05);
        assert!((ledger.get(long).unwrap().position.stop_price.unwrap() - 104.5).abs() < 1e-9);
        assert!((ledger.get(short).unwrap().position.stop_price.unwrap() - 98.7).abs() < 1e-9);
    }

    #[test]
    fn multiplier_scales_trade_pnl() {
        let mut ledger = PositionLedger::new("T".into(), PositionPolicy::Net);
        ledger.set_contract_multiplier(50.0);
        let id =
            ledger.open_position(0, 0, 100.0, 2.0, Direction::Long, None, None, 0.0, None).unwrap();
        let trade = ledger.close_position(id, exit(1, 101.0)).unwrap();
        assert!((trade.pnl - 1.0 * 2.0 * 50.0).abs() < 1e-9);
    }
}
