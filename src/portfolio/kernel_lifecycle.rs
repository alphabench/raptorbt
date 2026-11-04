//! Contract expiry, settlement and corporate actions on one kernel.
//!
//! A child of [`super`] (the kernel), so it reads the kernel's private
//! state. Every amount that moves cash is booked once through the ledger
//! (`PositionLedger::book`, i.e. `to_minor`), so cash, cost basis and realised
//! P&L still reconcile in integers across an expiry, a delivery, a split and
//! a dividend (`docs/quantitative-correctness/numerics.md`).
//!
//! The book (`portfolio::book`) owns what spans instruments: it fires an
//! expiry whose own bars have ended, delivers the underlying of a physically
//! settled line, and orders corporate actions against market data.

use super::*;
use crate::instruments::lifecycle::{
    SettlementPrice, SettlementSource, SettlementStyle, SettlementTerms,
};
use crate::instruments::spec::OptionRight;

/// Why a settlement or corporate action could not be applied. Stable
/// identifiers, reported on [`EngineEvent::LifecycleRefused`].
pub mod refusal {
    pub const UNDERLYING_PRICE_MISSING: &str = "underlying_price_missing";
    pub const PHYSICAL_NEEDS_UNDERLYING_PRICE: &str = "physical_settlement_needs_underlying_price";
    pub const PHYSICAL_NEEDS_PORTFOLIO: &str = "physical_settlement_needs_a_portfolio";
    pub const PHYSICAL_UNSUPPORTED: &str = "physical_settlement_unsupported";
    pub const NO_MARK_FOR_CASH_IN_LIEU: &str = "cash_in_lieu_needs_a_last_price";
}

/// A line that ended at expiry under physical settlement: what the book must
/// deliver on the underlying's kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ContractEnd {
    /// The underlying units delivered, in the contract's units times its
    /// multiplier: positive taken in, negative given away, `0.0` when an
    /// option expired out of the money.
    pub deliver: f64,
    /// The price the delivery is booked at.
    pub price: Price,
}

/// The settlement of one contract, before it is booked.
struct Resolved {
    /// The price the position closes at.
    close: Price,
    /// The per-unit value the settlement charges are levied on.
    value: Price,
    source: SettlementSource,
    /// `Some` for a physically settled contract: its delivery.
    physical: Option<Physical>,
}

#[derive(Clone, Copy)]
struct Physical {
    /// `Exercise`/`Assignment`/`Expiry` for an option, `Delivery` for a future.
    in_the_money: bool,
    /// The price the underlying is delivered at (the strike, or the
    /// future's settlement price).
    price: Price,
    /// `+1` receive for a call or a long future, `-1` for a put.
    toward: f64,
    option: bool,
}

/// What a split or bonus did to a kernel, for the book to report and add up.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SplitOutcome {
    /// Units paid in cash instead of delivered (signed: a short's is
    /// negative), at the mark on the new basis.
    pub fraction_units: f64,
    /// The cash that moved for them, signed as the account saw it.
    pub cash: f64,
}

/// A float within noise of a whole number is that number; otherwise its
/// floor. Sizes are non-negative.
fn snap_trunc(x: f64) -> f64 {
    let r = x.round();
    if (x - r).abs() < 1e-9 * x.abs().max(1.0) {
        r
    } else {
        x.floor()
    }
}

impl EngineKernel {
    /// Whether the kernel's owner delivers physically settled lines.
    pub(crate) fn set_delivers(&mut self, delivers: bool) {
        self.delivers = delivers;
    }

    /// Whether an underlying price has been supplied for settlement.
    pub(crate) fn has_underlying_price(&self) -> bool {
        self.underlying_price.is_some()
    }

    /// Whether the instrument trades in whole units (a share): a split that
    /// leaves a fraction pays it in cash.
    pub(crate) fn trades_whole_units(&self) -> bool {
        self.spec
            .as_ref()
            .is_some_and(|s| matches!(s.kind, InstrumentKind::Cash) && s.size_increment == 1.0)
    }

    /// Hold the settlement of this contract (the expired branch of a step
    /// still ends its orders) until the owner releases it: the underlying
    /// its terms read has not printed at the instant the expiry was noticed.
    pub(crate) fn hold_settlement(&mut self, held: bool) {
        self.settlement_held = held;
    }

    /// Take the lines a physical expiry ended since the last call.
    pub(crate) fn take_contract_ends(&mut self) -> Vec<ContractEnd> {
        std::mem::take(&mut self.contract_ends)
    }

    /// The signed units the kernel holds (long positive).
    pub(crate) fn net_units(&self) -> f64 {
        self.ledger
            .positions()
            .iter()
            .map(|m| m.position.size * m.position.direction.multiplier())
            .sum()
    }

    /// Whether expiry has anything left to end here: an open position, a
    /// working order, or a deferred signal.
    pub(crate) fn has_contract_state(&self) -> bool {
        self.ledger.open_count() > 0
            || self.orders.working().next().is_some()
            || self.deferred.is_some()
    }

    /// Whether the contract has expired at `ts`.
    pub(crate) fn expired_at(&self, ts: Timestamp) -> bool {
        self.spec.as_ref().is_some_and(|s| s.is_expired_at(ts))
    }

    /// End a contract whose expiry passed with no bar of its own at or
    /// after it: the book's timer, on the degenerate bar of the contract's
    /// last price stamped at the instant the expiry is noticed. The events
    /// are those the expired branch of a bar step gives.
    pub(crate) fn settle_at(&mut self, idx: usize, bar: &KernelBar) -> Vec<EngineEvent> {
        self.deferred = None;
        self.fee_model.set_time(bar.timestamp);
        let mut events = std::mem::take(&mut self.pending_events);
        events.extend(self.settle_expiry(idx, bar));
        self.cancel_all_orders(idx);
        events.append(&mut self.pending_events);
        events
    }

    /// Work out the settlement of the contract on `bar`: the price, its
    /// source, and the delivery of a physically settled line. A refusal is
    /// a stable reason: a price the terms need was not given, never a
    /// guess.
    fn resolve_settlement(&self, bar: &KernelBar) -> Result<Resolved, &'static str> {
        let Some(spec) = &self.spec else {
            return Ok(Resolved {
                close: bar.close,
                value: bar.close,
                source: SettlementSource::LastPrice,
                physical: None,
            });
        };
        let SettlementTerms { style, price: policy } = spec.settlement;
        let intrinsic = |under: Price| spec.settlement_value(bar.close, Some(under));
        let is_option = matches!(spec.kind, InstrumentKind::Option { .. });
        // The cash price of the contract itself, under the policy.
        let priced = |under_for_option: bool| -> Result<(Price, SettlementSource), &'static str> {
            match policy {
                SettlementPrice::Fixed(p) => Ok((p, SettlementSource::Declared)),
                SettlementPrice::LastPrice => Ok((bar.close, SettlementSource::LastPrice)),
                SettlementPrice::Underlying => match self.underlying_price {
                    Some(u) if under_for_option && is_option => {
                        Ok((intrinsic(u), SettlementSource::Underlying))
                    }
                    Some(u) => Ok((u, SettlementSource::Underlying)),
                    None => Err(refusal::UNDERLYING_PRICE_MISSING),
                },
                // An option cash-settles at the underlying's official close: without that price
                // the settlement is refused, as under `Underlying`, never made at the option's
                // own last trade (`LastPrice` asks for that).
                SettlementPrice::Auto => match self.underlying_price {
                    Some(u) if is_option && under_for_option => {
                        Ok((intrinsic(u), SettlementSource::Underlying))
                    }
                    None if is_option && under_for_option => Err(refusal::UNDERLYING_PRICE_MISSING),
                    _ => Ok((bar.close, SettlementSource::LastPrice)),
                },
            }
        };
        match style {
            SettlementStyle::Cash => {
                // `Fixed` is the contract's own price; an option settles
                // from the underlying only under `Underlying`/`Auto`.
                let (p, source) = priced(true)?;
                Ok(Resolved { close: p, value: p, source, physical: None })
            }
            SettlementStyle::Physical => match &spec.kind {
                InstrumentKind::Option { strike, right, binary: false, .. } => {
                    // The underlying's final price decides exercise.
                    let (under, source) = match policy {
                        SettlementPrice::Fixed(p) => (p, SettlementSource::Declared),
                        SettlementPrice::LastPrice => {
                            return Err(refusal::PHYSICAL_NEEDS_UNDERLYING_PRICE)
                        }
                        SettlementPrice::Underlying | SettlementPrice::Auto => {
                            match self.underlying_price {
                                Some(u) => (u, SettlementSource::Underlying),
                                None => return Err(refusal::PHYSICAL_NEEDS_UNDERLYING_PRICE),
                            }
                        }
                    };
                    let value = intrinsic(under);
                    Ok(Resolved {
                        close: 0.0,
                        value,
                        source,
                        physical: Some(Physical {
                            in_the_money: value > 0.0,
                            price: *strike,
                            toward: if *right == OptionRight::Call { 1.0 } else { -1.0 },
                            option: true,
                        }),
                    })
                }
                InstrumentKind::Contract { .. } => {
                    let (p, source) = priced(false)?;
                    Ok(Resolved {
                        close: p,
                        value: p,
                        source,
                        physical: Some(Physical {
                            in_the_money: true,
                            price: p,
                            toward: 1.0,
                            option: false,
                        }),
                    })
                }
                _ => Err(refusal::PHYSICAL_UNSUPPORTED),
            },
        }
    }

    /// Force-close the open positions at contract expiry.
    ///
    /// The price comes from the instrument's [`SettlementTerms`]
    /// (`resolve_settlement`) and its source rides on the
    /// [`EngineEvent::Settled`] this emits after each position's `Exited`.
    /// A price the terms need but the run never supplied refuses the
    /// settlement ([`EngineEvent::LifecycleRefused`], once): the position
    /// stays open for a later step to settle, it is not closed at a guess.
    ///
    /// The charges: a fee schedule's settlement row for the contract when
    /// the run was priced on a schedule that has one, else the spec's
    /// `settlement_fee` fraction of the settled value, else nothing.
    ///
    /// Physical settlement: the position ends at the contract's settlement
    /// price (an option at nothing; the exercise value is what is delivered
    /// against) and the line is queued in [`Self::take_contract_ends`] for
    /// the book to book on the underlying's kernel.
    pub(super) fn settle_expiry(&mut self, idx: usize, bar: &KernelBar) -> Vec<EngineEvent> {
        if !self.ledger.is_in_position() || self.settlement_held {
            return Vec::new();
        }
        let resolved = match self.resolve_settlement(bar) {
            Ok(r) if r.physical.is_some() && !self.delivers => {
                Err(refusal::PHYSICAL_NEEDS_PORTFOLIO)
            }
            other => other,
        };
        let resolved = match resolved {
            Ok(r) => r,
            Err(reason) => {
                if self.settlement_refused {
                    return Vec::new();
                }
                self.settlement_refused = true;
                return vec![EngineEvent::LifecycleRefused {
                    idx,
                    what: "settlement",
                    reason: reason.to_string(),
                }];
            }
        };
        self.settlement_refused = false;
        let style = self.spec.as_ref().map(|s| s.settlement.style).unwrap_or_default();
        let fee_rate = self.spec.as_ref().map(|s| s.settlement_fee).unwrap_or(0.0);
        let multiplier = self.multiplier();

        let ids: Vec<u64> = self.ledger.positions().iter().map(|p| p.id).collect();
        let mut events = Vec::new();
        for position_id in ids {
            let Some(managed) = self.ledger.get(position_id) else { continue };
            let entry_ts = managed.entry_timestamp;
            let entry_breakdown = managed.entry_breakdown;
            let (size, direction) = (managed.position.size, managed.position.direction);
            // The closing side of the line, priced per contract.
            let per_contract = resolved.value * multiplier;
            // A physically settled contract's own settlement is not charged by the
            // schedule's settlement row: what is owed is the delivery of the
            // underlying, a separate event priced on the cash market's delivery
            // rates (`deliver`), so the derivative's STT is not stacked on it.
            let settlement_row = if style == SettlementStyle::Physical {
                None
            } else {
                self.fee_model.settlement_breakdown(per_contract, size, direction, bar.timestamp)
            };
            let (settle_breakdown, settle_fee) = match settlement_row {
                Some(b) => {
                    let fees = self.booked_fee(b.total());
                    (
                        Some(crate::execution::costs::FeeBreakdown {
                            exact_total: Some(fees),
                            ..b
                        }),
                        fees,
                    )
                }
                None if fee_rate > 0.0 => {
                    (None, self.booked_fee(resolved.value * size * multiplier * fee_rate))
                }
                None => (None, 0.0),
            };
            let combined = match (entry_breakdown, settle_breakdown) {
                (Some(mut total), Some(exit)) => {
                    total.add(&exit);
                    Some(total)
                }
                (entry, exit) => entry.or(exit),
            };
            let Some(trade) = self.ledger.close_position(
                position_id,
                ExitDetails {
                    idx,
                    timestamp: bar.timestamp,
                    price: resolved.close,
                    entry_timestamp: entry_ts,
                    reason: ExitReason::Settlement,
                    fees: settle_fee,
                    fee_breakdown: combined,
                },
            ) else {
                continue;
            };
            self.credit_close(position_id, &trade);
            events.push(EngineEvent::Exited { idx, trade: Box::new(trade) });

            let (outcome, deliver, delivery_price) = match resolved.physical {
                None => ("settled", 0.0, resolved.close),
                Some(p) => {
                    let sign = direction.multiplier();
                    if p.option && !p.in_the_money {
                        ("expired", 0.0, p.price)
                    } else if p.option {
                        // a long exercises, a short is assigned
                        let name =
                            if direction == Direction::Long { "exercised" } else { "assigned" };
                        (name, sign * p.toward * size * multiplier, p.price)
                    } else {
                        ("delivered", sign * p.toward * size * multiplier, p.price)
                    }
                }
            };
            if resolved.physical.is_some() {
                self.contract_ends.push(ContractEnd { deliver, price: delivery_price });
            }
            events.push(EngineEvent::Settled {
                idx,
                outcome,
                style,
                source: resolved.source,
                price: resolved.close,
                size,
                direction,
                deliver,
                delivery_price,
            });
        }
        events
    }

    /// Book a physical delivery on this instrument (the underlying of an
    /// expired contract): `signed` units (positive taken in, negative given
    /// away; in this instrument's own units) at `price`, at `ts`. An
    /// opposite line closes first (reason [`ExitReason::Delivery`]); the
    /// rest opens or adds to a line. Charged as any fill is. Nothing is
    /// refused and no slippage applies: a delivery is the contract's
    /// obligation, not an order.
    pub(crate) fn deliver(
        &mut self,
        idx: usize,
        ts: Timestamp,
        signed: f64,
        price: Price,
    ) -> Vec<EngineEvent> {
        let mut events = Vec::new();
        if signed == 0.0 || !signed.is_finite() {
            return events;
        }
        let direction = if signed > 0.0 { Direction::Long } else { Direction::Short };
        self.fee_model.set_time(ts);
        let value = price * self.multiplier();
        let mut left = signed.abs();
        while left > 1e-12 {
            let Some(managed) =
                self.ledger.positions().iter().find(|m| m.position.direction != direction)
            else {
                break;
            };
            let (id, size, held) = (managed.id, managed.position.size, managed.position.direction);
            let (entry_ts, entry_breakdown) = (managed.entry_timestamp, managed.entry_breakdown);
            let qty = left.min(size);
            // Closing a held line with the delivery: a held long is given away
            // (a sell); a held short is bought back.
            let delivery =
                self.fee_model.delivery_breakdown(value, qty, held == Direction::Short, ts);
            let (breakdown, fees) = self.booked_charge(
                delivery.or_else(|| self.fee_model.breakdown(value, qty, held, false)),
                || self.fee_model.calculate(value, qty, held),
            );
            let exit = |fee_breakdown| ExitDetails {
                idx,
                timestamp: ts,
                price,
                entry_timestamp: entry_ts,
                reason: ExitReason::Delivery,
                fees,
                fee_breakdown,
            };
            let trade = if qty >= size {
                let combined = match (entry_breakdown, breakdown) {
                    (Some(mut total), Some(b)) => {
                        total.add(&b);
                        Some(total)
                    }
                    (entry, b) => entry.or(b),
                };
                let Some(trade) = self.ledger.close_position(id, exit(combined)) else { break };
                self.credit_close(id, &trade);
                trade
            } else {
                let Some(trade) = self.ledger.reduce_position(id, qty, exit(breakdown)) else {
                    break;
                };
                if matches!(self.account, AccountMode::Margin { .. }) {
                    let locked = self.margin.locked_for(id);
                    let _ = self.margin.set_locked(id, locked * (1.0 - qty / size));
                }
                self.credit_slice(&trade);
                trade
            };
            events.push(EngineEvent::Exited { idx, trade: Box::new(trade) });
            left -= qty;
        }
        if left > 1e-12 {
            let delivery =
                self.fee_model.delivery_breakdown(value, left, direction == Direction::Long, ts);
            let (breakdown, fees) = self.booked_charge(
                delivery.or_else(|| self.fee_model.breakdown(value, left, direction, true)),
                || self.fee_model.calculate(value, left, direction),
            );
            // A line the account cannot fund by rule is still delivered:
            // carried at its notional, as a cash account carries a line.
            let funding = self.entry_funding(direction, price).unwrap_or(None);
            let funding_cost = match funding {
                None => value * left,
                Some(EntryFunding::Rate { rate, .. }) => value * left * rate,
                Some(EntryFunding::Deposit { per_contract }) => per_contract * left,
                Some(EntryFunding::Unmargined) => 0.0,
            };
            let same = self.ledger.positions().iter().find(|m| m.position.direction == direction);
            let line = match same.map(|m| m.id) {
                Some(id) => {
                    self.ledger.add_to_position(id, price, left, fees, false).map(|_| (id, false))
                }
                None => self
                    .ledger
                    .open_position(idx, ts, price, left, direction, None, None, fees, breakdown)
                    .map(|id| (id, true)),
            };
            if let Some((id, opened)) = line {
                let fees_money = self.ledger.book(fees);
                match funding {
                    None => self.move_cash(-(self.ledger.notional(price, left) + fees_money)),
                    Some(_) if opened => {
                        self.margin.lock(id, funding_cost);
                        self.move_cash(-fees_money);
                    }
                    Some(_) => {
                        let locked = self.margin.locked_for(id);
                        let _ = self.margin.set_locked(id, locked + funding_cost);
                        self.move_cash(-fees_money);
                    }
                }
                events.push(EngineEvent::Entered { idx, price, size: left, direction });
            }
        }
        events
    }

    // -- corporate actions ---------------------------------------------------

    /// A split or bonus on this instrument: `ratio` new units per old. Every
    /// open position's size scales up and its prices (entry, stop, target,
    /// running extremes) scale down, so its booked cost basis is untouched
    /// to the minor unit. Working orders and algos are cancelled (their
    /// prices are on the old scale), as an exchange cancels them; their
    /// acknowledgments are in the next step's events.
    ///
    /// `whole`: the instrument trades in whole units (a share), so each
    /// direction's total keeps its whole units and the rest is paid in cash
    /// at `mark` (the last price, on the OLD basis) divided by `ratio`: the
    /// fraction closes as a trade slice at that price, with its share of the
    /// cost basis, no fees, reason [`ExitReason::CorporateAction`]. Without
    /// a mark the split is refused, not priced at a guess.
    pub(crate) fn rescale_for_split(
        &mut self,
        idx: usize,
        ts: Timestamp,
        ratio: f64,
        whole: bool,
        mark: Option<Price>,
    ) -> Result<(SplitOutcome, Vec<EngineEvent>), &'static str> {
        // The per-position plan first, so a refusal changes nothing.
        let mut cum = [0.0f64; 2];
        let mut prev = [0.0f64; 2];
        let mut plan: Vec<(u64, f64)> = Vec::new();
        for managed in self.ledger.positions() {
            let p = &managed.position;
            let side = usize::from(p.direction == Direction::Short);
            cum[side] += p.size * ratio;
            let kept = if whole { snap_trunc(cum[side]) } else { cum[side] };
            plan.push((managed.id, (p.size * ratio) - (kept - prev[side])));
            prev[side] = kept;
        }
        let fraction_total: f64 = plan.iter().map(|(_, f)| f.abs()).sum();
        let needs_mark = whole && plan.iter().any(|(_, f)| *f > 1e-9);
        if needs_mark && mark.is_none() {
            return Err(refusal::NO_MARK_FOR_CASH_IN_LIEU);
        }
        let _ = fraction_total;

        for managed in self.ledger.positions_mut() {
            let p = &mut managed.position;
            p.size *= ratio;
            p.entry_price /= ratio;
            p.stop_price = p.stop_price.map(|s| s / ratio);
            p.target_price = p.target_price.map(|t| t / ratio);
            p.highest_since_entry /= ratio;
            p.lowest_since_entry /= ratio;
        }
        self.last_atr /= ratio;
        self.underlying_price = self.underlying_price.map(|u| u / ratio);
        let mut events = Vec::new();
        for id in self.orders.cancel_all() {
            let client_id = self.orders.get(id).map(|o| o.client_id.clone()).unwrap_or_default();
            events.push(EngineEvent::OrderCanceled { idx, order_id: id, client_id });
        }
        self.deferred = None;

        let mut outcome = SplitOutcome::default();
        if let (true, Some(mark)) = (whole, mark) {
            let price = mark / ratio;
            for (id, fraction) in plan {
                if fraction <= 1e-9 {
                    continue;
                }
                let Some(managed) = self.ledger.get(id) else { continue };
                let (direction, size, entry_ts) =
                    (managed.position.direction, managed.position.size, managed.entry_timestamp);
                let exit = ExitDetails {
                    idx,
                    timestamp: ts,
                    price,
                    entry_timestamp: entry_ts,
                    reason: ExitReason::CorporateAction,
                    fees: 0.0,
                    fee_breakdown: None,
                };
                let before = self.cash;
                let trade = if fraction >= size {
                    let Some(t) = self.ledger.close_position(id, exit) else { continue };
                    self.credit_close(id, &t);
                    t
                } else {
                    let Some(t) = self.ledger.reduce_position(id, fraction, exit) else { continue };
                    if matches!(self.account, AccountMode::Margin { .. }) {
                        let locked = self.margin.locked_for(id);
                        let _ = self.margin.set_locked(id, locked * (1.0 - fraction / size));
                    }
                    self.credit_slice(&t);
                    t
                };
                outcome.fraction_units += fraction * direction.multiplier();
                outcome.cash += self.money_scale.to_f64(self.cash - before);
                events.push(EngineEvent::Exited { idx, trade: Box::new(trade) });
            }
        }
        Ok((outcome, events))
    }

    /// The cash a dividend of `amount` per unit pays the holder at its
    /// ex-date: the signed units held times the contract multiplier (a short
    /// pays it), booked once in minor units.
    pub(crate) fn dividend_cash(&self, amount: f64) -> Money {
        self.ledger.book(amount * self.net_units() * self.multiplier())
    }

    /// Credit (or, for a short, debit) booked dividend cash to this kernel.
    pub(crate) fn credit_dividend(&mut self, cash: Money) {
        self.move_cash(cash);
    }
}
