//! Market data: bar aggregation specs, streaming builders, event streams,
//! and the deterministic multi-stream merge.

pub mod aggregate;
pub mod aggregate_resample;
pub mod bar_spec;
pub mod book;
pub mod events;
pub mod feed;
pub mod flow_bars;
pub mod renko;
pub mod session_bars;

pub use aggregate::{builder_for, builder_for_with, BarBuilder, SourceRecord, IST_OFFSET_NS};
pub use aggregate_resample::{
    CloseOn, Closed, ClosedBy, Gaps, ResampleError, ResampleOptions, ResampledBar,
};
pub use bar_spec::{AggregationUnit, BarSpec, BarStamp, BuilderParams, SpecError};
pub use book::{BookLevel, BookSide, DepthTick, OrderBook, BOOK_DEPTH};
pub use events::{tick_data_to_events, DepthRef, EventPayload, MarketEvent, QuoteTick, TradeTick};
pub use feed::{ensure_time_sorted, EventFeed};
pub use flow_bars::{FlowBarBuilder, FlowMagnitude, FlowRule};
pub use renko::RenkoBarBuilder;
pub use session_bars::{builder_for_session, follows_calendar};
