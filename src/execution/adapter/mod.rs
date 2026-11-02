//! Where orders leave the engine.
//!
//! A strategy never talks to a venue. It queues commands, and the runtime
//! routes each one either to the in-process matcher (the default: backtests,
//! and anything that has not chosen an adapter) or to an [`ExecutionAdapter`]:
//! a real broker connection, or the built-in [`PaperVenue`].
//!
//! # Contract
//!
//! * **Requests are fire and forget.** `submit`, `cancel` and `modify` return
//!   once the request has been handed over. What became of it comes back only
//!   as [`ExecutionReport`]s, which may repeat, arrive late or arrive out of
//!   order (`docs/architecture/order-lifecycle.md`). `Ok(())` therefore means
//!   "sent", never "accepted".
//! * **Reports enter the engine one way:** the runtime's `push_report`, as an
//!   event in the same single-threaded stream as market data. An adapter never
//!   touches engine state from another thread.
//! * **Identity is deterministic.** An order's venue-facing id is
//!   [`client_order_id`]`(strategy, seq)`: the same strategy making the same
//!   commands in the same order always produces the same ids, so a replay or a
//!   recovery can tell what was already sent. The engine never sends an id
//!   twice.
//!
//! # What the in-process path keeps
//!
//! The in-process matcher is not an adapter. It is the direct path the
//! backtest hot loop takes, so a run with no adapter pays nothing for this
//! module (no dynamic dispatch per event, no per-event allocation).

mod paper;

use std::collections::BTreeMap;

pub use paper::PaperVenue;

use crate::execution::orders::{OrderKind, OrderSide, TimeInForce};

/// The venue-facing id of a strategy's `seq`-th order: `s{strategy}-{seq}`.
///
/// A pure function of the strategy's index and its own count of orders sent,
/// so it is stable across replays.
pub fn client_order_id(strategy: u32, seq: u64) -> String {
    format!("s{strategy}-{seq}")
}

/// Split an id made by [`client_order_id`] back into `(strategy, seq)`.
pub fn parse_client_order_id(id: &str) -> Option<(u32, u64)> {
    let rest = id.strip_prefix('s')?;
    let (strategy, seq) = rest.split_once('-')?;
    Some((strategy.parse().ok()?, seq.parse().ok()?))
}

/// One order, as the engine hands it to a venue.
///
/// The quantity is always an explicit unit count: capital fractions and
/// "whatever the position holds" are resolved before an order leaves the
/// engine.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderRequest {
    /// Deterministic, unique per order, see [`client_order_id`].
    pub client_order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    /// Units (contracts, shares), positive.
    pub quantity: f64,
    /// The order type with its limit and trigger prices.
    pub kind: OrderKind,
    pub tif: TimeInForce,
    /// The submitting strategy's index.
    pub strategy_id: u32,
    /// The venue account the order is for, when the node was given one.
    pub account_id: Option<String>,
    /// The order may only reduce an open position.
    pub reduce_only: bool,
    /// A limit order that must rest: refused rather than executed on arrival.
    pub post_only: bool,
    /// Hold this order until the named order (a `client_order_id`) fills;
    /// drop it if that order ends without filling. A venue with no native
    /// support refuses with [`AdapterError::Unsupported`].
    pub parent_client_order_id: Option<String>,
    /// Free-form labels. The engine sets `client_id` to the id the strategy
    /// itself gave the order.
    pub tags: BTreeMap<String, String>,
    /// Event time (ns) of the decision that produced the request.
    pub submitted_ns: i64,
    /// The instrument's own event ordinal at that decision. A simulated venue
    /// needs it to keep an order from matching on the event that produced it;
    /// a real venue ignores it.
    pub submitted_idx: usize,
}

/// A change to a working order. `None` leaves a field as it is.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct OrderChange {
    /// The new *total* order quantity.
    pub quantity: Option<f64>,
    pub limit_price: Option<f64>,
    pub trigger_price: Option<f64>,
}

/// Why a request was not handed to the venue.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AdapterError {
    /// The adapter knows the venue will not take the order (bad parameters,
    /// a closed market). Definitive: the engine rejects the order.
    #[error("order refused: {0}")]
    Refused(String),
    /// The adapter cannot express the request at all (an order type or link
    /// the venue has no equivalent for). The engine rejects the order.
    #[error("not supported: {0}")]
    Unsupported(String),
    /// The request may or may not have reached the venue (a dropped
    /// connection). The order stays as it was and the error is raised to the
    /// caller; what happened is settled by the reports that follow, or by
    /// reconciliation.
    #[error("venue unavailable: {0}")]
    Unavailable(String),
}

/// A venue the engine can send orders to.
///
/// Implementations hold no engine state and must not call back into it.
pub trait ExecutionAdapter {
    /// Send a new order.
    fn submit(&mut self, req: &OrderRequest) -> Result<(), AdapterError>;

    /// Ask for a working order to be cancelled. `client_order_id` is the id
    /// the order was submitted with.
    fn cancel(&mut self, client_order_id: &str) -> Result<(), AdapterError>;

    /// Ask for a working order's quantity or prices to change.
    fn modify(&mut self, client_order_id: &str, change: &OrderChange) -> Result<(), AdapterError>;

    /// Make the named working orders one-cancels-other: the first to fill
    /// cancels the rest. Optional: a venue with no such thing keeps the
    /// default, and the engine refuses the link.
    fn link_oco(&mut self, _client_order_ids: &[String]) -> Result<(), AdapterError> {
        Err(AdapterError::Unsupported("one-cancels-other links".into()))
    }

    /// The venue this adapter simulates, if it is a simulation that advances
    /// with the market. A real venue returns `None`.
    fn as_simulation(&mut self) -> Option<&mut PaperVenue> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_order_ids_round_trip_and_depend_only_on_their_inputs() {
        assert_eq!(client_order_id(3, 0), "s3-0");
        assert_eq!(client_order_id(3, 0), client_order_id(3, 0));
        assert_ne!(client_order_id(3, 1), client_order_id(4, 1));
        assert_eq!(parse_client_order_id("s12-340"), Some((12, 340)));
        assert_eq!(parse_client_order_id("12-340"), None);
        assert_eq!(parse_client_order_id("s12"), None);
        assert_eq!(parse_client_order_id("sx-1"), None);
    }
}
