//! Shared multi-strategy runtime: one copy of market data per process, read
//! in place by many strategies.
//!
//! * [`aggregators`]: one shared bar aggregator per distinct aggregation.
//! * [`ids`]: dense integer handles and symbol interning.
//! * [`store`]: immutable, `Arc`-backed columns for bars, ticks and depth.
//! * [`cursor`]: lazy k-way merge over a store, without a materialised schedule.
//! * [`ring`]: bounded row storage for live data, with backpressure.
//! * [`live`]: the rings of a live node and the ordering of pushed events.
//! * [`atr_cache`]: one ATR series per instrument and period.
//! * [`commands`]: a strategy's queued commands and intents, as typed data.
//! * [`indicators`]: one shared streaming core per distinct indicator key.
//! * [`memory`]: exact heap accounting.
//! * [`router`]: CSR subscription indexes.
//! * [`groups`]: several strategies on one account.
//! * [`runner`]: the core that drives many strategy books over one store.

pub mod aggregators;
pub mod atr_cache;
pub mod commands;
pub mod cursor;
pub mod groups;
pub mod ids;
pub mod indicators;
pub mod live;
pub mod memory;
pub mod ring;
pub mod router;
pub mod runner;
pub mod store;

#[cfg(test)]
mod aggregators_tests;
#[cfg(test)]
mod cursor_tests;
#[cfg(test)]
mod indicators_bank_tests;
#[cfg(test)]
mod indicators_tests;

pub use aggregators::{AggKey, AggregatorHub, AggregatorHubBuilder, Deliveries};
pub use atr_cache::{AtrCache, AtrStats};
pub use cursor::{EventKind, EventRef, MergeCursor};
pub use ids::{AggId, IndId, InstrumentId, InstrumentTable, StrategyId};
pub use indicators::{
    IndInput, IndKey, IndKind, IndSpec, IndicatorBank, IndicatorRegistry, InputField,
    LocalStreamId, Ohlc, PrimaryOrder, Reading, Registration, Reveal, SlotId,
};
pub use live::{LiveFeed, RingKind};
pub use memory::HeapSize;
pub use ring::{ColumnSource, ConsumerId, RingColumns, RingError, RowMiss, RowRing, TickRow};
pub use router::{
    AggSubscriptions, AggSubscriptionsBuilder, SubscriptionIndex, SubscriptionIndexBuilder,
};
pub use runner::{
    Current, InstrumentParams, MemoryReport, Pushed, RuntimeBuilder, RuntimeCore, StrategySlot,
};
pub use store::{BarColumns, DepthColumns, MarketStore, TickColumns, TickSlices};
