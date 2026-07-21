//! Market data: bar aggregation specs, streaming builders, event streams,
//! and the deterministic multi-stream merge.

pub mod aggregate;
pub mod book;
pub mod bar_spec;
pub mod events;
pub mod feed;
pub mod flow_bars;
pub mod renko;

pub use aggregate::{builder_for, builder_for_with, BarBuilder, SourceRecord, IST_OFFSET_NS};
pub use book::{BookLevel, BookSide, DepthTick, OrderBook, BOOK_DEPTH};
pub use bar_spec::{AggregationUnit, BarSpec, BuilderParams, SpecError};
pub use flow_bars::{FlowBarBuilder, FlowMagnitude, FlowRule};
pub use renko::RenkoBarBuilder;
pub use events::{
    tick_data_to_events, DepthRef, EventPayload, MarketEvent, QuoteTick, TradeTick,
};
pub use feed::EventFeed;
