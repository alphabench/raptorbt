//! Order types and the order state machine.

use serde::{Deserialize, Serialize};

use crate::core::types::{Price, Timestamp};

/// Which way an order trades.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl OrderSide {
    /// Stable identifier for reporting, in the same snake_case as every
    /// other name that crosses into a result.
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            OrderSide::Buy => "buy",
            OrderSide::Sell => "sell",
        }
    }

    /// The opposite side.
    #[inline]
    pub fn flip(self) -> Self {
        match self {
            OrderSide::Buy => OrderSide::Sell,
            OrderSide::Sell => OrderSide::Buy,
        }
    }
}

/// How much an order trades.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QtySpec {
    /// Explicit unit/contract count.
    Units(f64),
    /// Fraction of available capital, resolved to units at fill time —
    /// equity moves while an order rests, so resolving at accept time would
    /// size against a stale account.
    CapitalFrac(f64),
    /// Whatever the open position holds when the order fills (close-all).
    FullPosition,
}

/// How a trailing offset is measured.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TrailOffset {
    /// Absolute price distance from the watermark.
    Price(f64),
    /// Basis points of the watermark.
    Bps(f64),
}

impl TrailOffset {
    /// Offset in price terms at the given watermark.
    #[inline]
    pub fn at(&self, watermark: Price) -> f64 {
        match self {
            TrailOffset::Price(p) => *p,
            TrailOffset::Bps(b) => watermark * b / 10_000.0,
        }
    }
}

/// Order flavor and its price levels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OrderKind {
    /// Fill at the engine's configured fill-price model on the current bar.
    Market,
    /// Rest until the market trades at or through `price`.
    Limit { price: Price },
    /// Become marketable once the market trades adversely through `trigger`.
    StopMarket { trigger: Price },
    /// Once triggered, rest as a limit at `price`.
    StopLimit { trigger: Price, price: Price },
    /// Become marketable once the market trades *favorably* to `trigger`
    /// (a buy triggers when price falls to it) — the stop's mirror.
    MarketIfTouched { trigger: Price },
    /// Favorable touch at `trigger`, then rest as a limit at `price`.
    LimitIfTouched { trigger: Price, price: Price },
    /// Fill at the next bar's open. Without partial fills the "remainder
    /// rests as a limit" phase never occurs, so this is exactly an
    /// at-the-open market order; the distinction returns with book depth.
    MarketToLimit,
    /// Stop whose trigger trails the running favorable extreme by `offset`.
    TrailingStopMarket { offset: TrailOffset },
    /// Trailing stop that, once triggered, rests as a limit `limit_offset`
    /// through the trigger (more marketable, bounded slippage).
    TrailingStopLimit { offset: TrailOffset, limit_offset: f64 },
}

/// How long an order stays working.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeInForce {
    /// Rest until filled or canceled.
    Gtc,
    /// Expire when the trading date rolls over past the submission date.
    /// The date is UTC unless `session_tz_offset_ns` shifts it.
    Day,
    /// Expire at an explicit timestamp.
    Gtd { expire_ns: Timestamp },
    /// Match against exactly one bar, then cancel.
    Ioc,
    /// Fill completely against one bar or cancel. With `partial_fills` on,
    /// a print smaller than the remainder cancels the order untouched;
    /// without it, behaves like [`TimeInForce::Ioc`].
    Fok,
    /// Market order queued to fill at the next bar's open.
    AtOpen,
    /// Market order queued to fill at the next bar's close.
    AtClose,
}

/// Order lifecycle states.
///
/// Transitions are enforced by [`Order::transition`]; anything not listed
/// there is a logic error, caught in debug builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    /// Created, not yet evaluated by the engine.
    Submitted,
    /// Working: resting in the engine's order book.
    Accepted,
    /// Stop trigger touched; a stop-limit now rests as a limit.
    Triggered,
    /// Working: some of the quantity has filled (`filled_qty`), the rest
    /// still rests. Only reachable with `partial_fills` on.
    PartiallyFilled,
    /// Terminal: filled.
    Filled,
    /// Terminal: canceled by the strategy or by IOC/FOK exhaustion.
    Canceled,
    /// Terminal: time-in-force expired.
    Expired,
    /// Terminal: refused (e.g. opening order while a position is open).
    Rejected,
}

impl OrderStatus {
    /// Stable identifier for reporting.
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            OrderStatus::Submitted => "submitted",
            OrderStatus::Accepted => "accepted",
            OrderStatus::Triggered => "triggered",
            OrderStatus::PartiallyFilled => "partially_filled",
            OrderStatus::Filled => "filled",
            OrderStatus::Canceled => "canceled",
            OrderStatus::Expired => "expired",
            OrderStatus::Rejected => "rejected",
        }
    }

    /// Whether this is an end state.
    #[inline]
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            OrderStatus::Filled
                | OrderStatus::Canceled
                | OrderStatus::Expired
                | OrderStatus::Rejected
        )
    }
}

impl OrderKind {
    /// Stable identifier for reporting. The prices a kind carries are
    /// reported separately, so this names the shape only.
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            OrderKind::Market => "market",
            OrderKind::Limit { .. } => "limit",
            OrderKind::StopMarket { .. } => "stop_market",
            OrderKind::StopLimit { .. } => "stop_limit",
            OrderKind::MarketIfTouched { .. } => "market_if_touched",
            OrderKind::LimitIfTouched { .. } => "limit_if_touched",
            OrderKind::MarketToLimit => "market_to_limit",
            OrderKind::TrailingStopMarket { .. } => "trailing_stop_market",
            OrderKind::TrailingStopLimit { .. } => "trailing_stop_limit",
        }
    }

    /// The resting limit price this kind names, if any.
    #[inline]
    pub fn limit_price(&self) -> Option<Price> {
        match self {
            OrderKind::Limit { price }
            | OrderKind::StopLimit { price, .. }
            | OrderKind::LimitIfTouched { price, .. } => Some(*price),
            _ => None,
        }
    }

    /// The trigger price this kind names, if any.
    #[inline]
    pub fn trigger_price(&self) -> Option<Price> {
        match self {
            OrderKind::StopMarket { trigger }
            | OrderKind::StopLimit { trigger, .. }
            | OrderKind::MarketIfTouched { trigger }
            | OrderKind::LimitIfTouched { trigger, .. } => Some(*trigger),
            _ => None,
        }
    }
}

impl TimeInForce {
    /// Stable identifier for reporting.
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            TimeInForce::Gtc => "gtc",
            TimeInForce::Day => "day",
            TimeInForce::Gtd { .. } => "gtd",
            TimeInForce::Ioc => "ioc",
            TimeInForce::Fok => "fok",
            TimeInForce::AtOpen => "at_open",
            TimeInForce::AtClose => "at_close",
        }
    }
}

/// One working or finished order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Order {
    /// Engine-assigned id, unique within a session, monotonically increasing
    /// in submission order — matching iterates ids ascending, so submission
    /// order is the deterministic tiebreak.
    pub id: u64,
    /// Caller-supplied identifier, echoed on every event.
    pub client_id: String,
    pub side: OrderSide,
    pub qty: QtySpec,
    pub kind: OrderKind,
    pub tif: TimeInForce,
    pub status: OrderStatus,
    /// Bar index on which the order was submitted. Resting orders begin
    /// matching on the *next* bar: an order cannot rest into a bar that had
    /// already closed when it was placed.
    pub submitted_idx: usize,
    /// Submission timestamp, for DAY expiry.
    pub submitted_ts: Timestamp,
    /// Protective stop attached to the position this order opens.
    pub stop_price: Option<Price>,
    /// Protective target attached to the position this order opens.
    pub target_price: Option<Price>,
    /// Limit orders only: reject instead of filling if marketable at the
    /// open of the first bar the order rests into.
    pub post_only: bool,
    /// Reject fills that would open a position (closing fills only).
    pub reduce_only: bool,
    /// One-triggers-other: held (not matched) until the parent order fills;
    /// canceled if the parent dies unfilled.
    pub parent_id: Option<u64>,
    /// Schedule that released this order, when it is an algo slice.
    /// Purely a back-pointer: slices match like any other order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algo_id: Option<u64>,
    /// One-cancels-other group: when any member fills, working siblings are
    /// canceled. One-updates-other reduces to this without partial fills.
    pub oco_group: Option<u64>,
    /// Trailing orders: running favorable extreme since acceptance.
    pub trail_watermark: Option<Price>,
    /// Trailing stop-limit: the limit price fixed at trigger time.
    pub trail_limit: Option<Price>,
    /// Units filled so far. Non-zero only while `PartiallyFilled` (or once
    /// `Filled`); a whole-fill engine leaves it at the full size on fill.
    #[serde(default)]
    pub filled_qty: f64,
}

/// What became of one order, once the run is over.
///
/// The engine's own account of an order that was placed — including one that
/// never filled. A backtest reports the trades it made; without this it
/// cannot report the trades it *tried* to make and could not, which is often
/// the whole explanation for a result. A strategy that looks like it has no
/// edge may simply have been refused two thirds of its entries.
///
/// `Order` carries almost all of this already; the two things it cannot know
/// are why the engine refused it (that lives in the event, not the book) and
/// what it actually filled at across possibly several slices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderRecord {
    pub id: u64,
    pub client_id: String,
    /// Instrument the order was placed on.
    pub symbol: String,
    /// `"buy"` / `"sell"`.
    pub side: &'static str,
    /// `"market"`, `"limit"`, `"stop_market"`, … — the order kind, named.
    pub kind: &'static str,
    /// Time in force: `"gtc"`, `"day"`, `"ioc"`, …
    pub tif: &'static str,
    /// Terminal (or final observed) state: `"filled"`, `"canceled"`,
    /// `"expired"`, `"rejected"`, `"accepted"`, …
    pub status: &'static str,
    pub submitted_idx: usize,
    pub submitted_ts: Timestamp,
    /// Units asked for, when the request named a number. `None` for a
    /// capital-fraction or whole-position request, where the size is only
    /// known once the engine prices it — "not stated", never a zero request.
    #[serde(default)]
    pub requested_qty: Option<f64>,
    /// Units actually filled, summed across slices. `0.0` for an order that
    /// never filled, which is a measurement and not a gap.
    #[serde(default)]
    pub filled_qty: f64,
    /// Size-weighted mean fill price across every slice, or `None` when the
    /// order never filled.
    #[serde(default)]
    pub avg_fill_price: Option<Price>,
    /// Bar index of the last fill slice, or `None` when it never filled.
    #[serde(default)]
    pub last_fill_idx: Option<usize>,
    /// How many separate fills it took. `1` is a clean fill; more is a
    /// partial-fill sequence.
    #[serde(default)]
    pub fill_slices: u32,
    /// Limit price, for the kinds that carry one.
    #[serde(default)]
    pub limit_price: Option<Price>,
    /// Trigger price, for stop and if-touched kinds.
    #[serde(default)]
    pub trigger_price: Option<Price>,
    /// Why the engine refused it, as the stable snake_case identifier
    /// (`"insufficient_margin"`, `"max_positions"`, …). `None` for an order
    /// that was not rejected — never an empty string, which would read as a
    /// rejection with no stated cause.
    #[serde(default)]
    pub reject_reason: Option<String>,
    /// One-triggers-other parent, when this order was held for one.
    #[serde(default)]
    pub parent_id: Option<u64>,
    /// One-cancels-other group, when it had siblings.
    #[serde(default)]
    pub oco_group: Option<u64>,
}

impl OrderRecord {
    /// The book's account of an order, before the event stream adds why it
    /// was refused and what it filled at.
    ///
    /// `symbol` is not on `Order` — one book serves one instrument, so the
    /// caller supplies it.
    pub fn from_order(order: &Order, symbol: &str) -> Self {
        Self {
            id: order.id,
            client_id: order.client_id.clone(),
            symbol: symbol.to_string(),
            side: order.side.as_str(),
            kind: order.kind.as_str(),
            tif: order.tif.as_str(),
            status: order.status.as_str(),
            submitted_idx: order.submitted_idx,
            submitted_ts: order.submitted_ts,
            // Only an explicit unit request states a size up front. A
            // capital fraction is not known until the engine prices it, and
            // reporting 0.0 there would read as "asked for nothing".
            requested_qty: match order.qty {
                QtySpec::Units(u) => Some(u),
                _ => None,
            },
            filled_qty: order.filled_qty,
            avg_fill_price: None,
            last_fill_idx: None,
            fill_slices: 0,
            limit_price: order.kind.limit_price(),
            trigger_price: order.kind.trigger_price(),
            reject_reason: None,
            parent_id: order.parent_id,
            oco_group: order.oco_group,
        }
    }

    /// Fold one fill slice in, keeping a size-weighted mean price.
    ///
    /// Weighted, not last-wins: an order that filled 90 units at 100 and 10
    /// at 130 paid 103 on average, and reporting 130 would misstate every
    /// partial fill.
    pub fn record_fill(&mut self, idx: usize, price: Price, size: f64) {
        let prior = self.avg_fill_price.unwrap_or(0.0) * self.filled_qty;
        self.filled_qty += size;
        self.avg_fill_price = if self.filled_qty > 0.0 {
            Some((prior + price * size) / self.filled_qty)
        } else {
            None
        };
        self.last_fill_idx = Some(idx);
        self.fill_slices += 1;
    }
}

impl Order {
    /// A plain unlinked GTC-style order in `Submitted` state.
    ///
    /// Construction sites set ids, timestamps, flags, and links on top;
    /// keeping one canonical literal avoids churn as fields grow.
    pub fn plain(side: OrderSide, qty: QtySpec, kind: OrderKind, tif: TimeInForce) -> Self {
        Self {
            id: 0,
            client_id: String::new(),
            side,
            qty,
            kind,
            tif,
            status: OrderStatus::Submitted,
            submitted_idx: 0,
            submitted_ts: 0,
            stop_price: None,
            target_price: None,
            post_only: false,
            reduce_only: false,
            parent_id: None,
            algo_id: None,
            oco_group: None,
            trail_watermark: None,
            trail_limit: None,
            filled_qty: 0.0,
        }
    }

    /// Whether the order is a stop-limit style order whose limit is live —
    /// triggered, or already partly filled at that limit.
    #[inline]
    pub fn limit_live(&self) -> bool {
        matches!(self.status, OrderStatus::Triggered | OrderStatus::PartiallyFilled)
    }

    /// Move to a new status, checking the transition is legal.
    ///
    /// Returns `false` (and leaves the order untouched) on an illegal
    /// transition; callers treat that as a bug, not a user error.
    #[must_use]
    pub fn transition(&mut self, to: OrderStatus) -> bool {
        use OrderStatus::*;
        let ok = matches!(
            (self.status, to),
            (Submitted, Accepted)
                | (Submitted, Rejected)
                | (Submitted, Canceled)
                | (Accepted, Triggered)
                | (Accepted, Filled)
                | (Accepted, Canceled)
                | (Accepted, Expired)
                | (Accepted, Rejected)
                | (Triggered, Filled)
                | (Triggered, Canceled)
                | (Triggered, Expired)
                | (Triggered, Rejected)
                | (Accepted, PartiallyFilled)
                | (Triggered, PartiallyFilled)
                | (PartiallyFilled, PartiallyFilled)
                | (PartiallyFilled, Filled)
                | (PartiallyFilled, Canceled)
                | (PartiallyFilled, Expired)
                | (PartiallyFilled, Rejected)
        );
        if ok {
            self.status = to;
        }
        debug_assert!(ok, "illegal order transition {:?} -> {to:?}", self.status);
        ok
    }

    /// The price a resting order would currently fill or trigger at.
    #[inline]
    pub fn working_price(&self) -> Option<Price> {
        let triggered = self.limit_live();
        match self.kind {
            OrderKind::Market | OrderKind::MarketToLimit => None,
            OrderKind::Limit { price } => Some(price),
            OrderKind::StopMarket { trigger } | OrderKind::MarketIfTouched { trigger } => {
                Some(trigger)
            }
            OrderKind::StopLimit { trigger, price }
            | OrderKind::LimitIfTouched { trigger, price } => {
                Some(if triggered { price } else { trigger })
            }
            OrderKind::TrailingStopMarket { offset } => {
                self.trail_watermark.map(|wm| self.trail_trigger(wm, offset))
            }
            OrderKind::TrailingStopLimit { offset, .. } => {
                if triggered {
                    self.trail_limit
                } else {
                    self.trail_watermark.map(|wm| self.trail_trigger(wm, offset))
                }
            }
        }
    }

    /// Trigger level implied by a watermark for a trailing order.
    ///
    /// A sell trail protects a long: trigger below the running high. A buy
    /// trail mirrors it below-market entry style: trigger above the running
    /// low.
    #[inline]
    pub fn trail_trigger(&self, watermark: Price, offset: TrailOffset) -> Price {
        match self.side {
            OrderSide::Sell => watermark - offset.at(watermark),
            OrderSide::Buy => watermark + offset.at(watermark),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(kind: OrderKind) -> Order {
        let mut o = Order::plain(OrderSide::Buy, QtySpec::Units(1.0), kind, TimeInForce::Gtc);
        o.id = 1;
        o.client_id = "t-1".into();
        o
    }

    #[test]
    fn legal_lifecycle_paths() {
        let mut o = order(OrderKind::Limit { price: 100.0 });
        assert!(o.transition(OrderStatus::Accepted));
        assert!(o.transition(OrderStatus::Filled));
        assert!(o.status.is_terminal());
    }

    #[test]
    fn stop_limit_triggers_then_fills() {
        let mut o = order(OrderKind::StopLimit { trigger: 105.0, price: 104.5 });
        assert!(o.transition(OrderStatus::Accepted));
        assert_eq!(o.working_price(), Some(105.0));
        assert!(o.transition(OrderStatus::Triggered));
        assert_eq!(o.working_price(), Some(104.5));
        assert!(o.transition(OrderStatus::Filled));
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn illegal_transition_is_refused() {
        let mut o = order(OrderKind::Market);
        assert!(o.transition(OrderStatus::Accepted));
        assert!(o.transition(OrderStatus::Filled));
        assert!(!o.transition(OrderStatus::Accepted));
        assert_eq!(o.status, OrderStatus::Filled);
    }
}
