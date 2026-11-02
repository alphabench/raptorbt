//! The cursor must reproduce the event session's sealed schedule exactly.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed, TestRunner};

use crate::core::types::{BacktestConfig, Direction, OhlcvBar, TickData};
use crate::data::book::{BookLevel, DepthTick};
use crate::data::events::{DepthRef, EventPayload, MarketEvent, QuoteTick, TradeTick};
use crate::portfolio::kernel::{KernelBar, StepInput};
use crate::portfolio::ledger::PositionPolicy;
use crate::portfolio::session::{EventSession, ScheduleData};
use crate::runtime::cursor::{EventKind, EventRef, MergeCursor};
use crate::runtime::store::{BarColumns, DepthColumns, MarketStore, TickColumns};

#[test]
fn event_ref_fits_in_32_bytes() {
    // Row and ordinal are 64-bit so a feed that never stops cannot wrap them.
    assert!(std::mem::size_of::<EventRef>() <= 32, "{}", std::mem::size_of::<EventRef>());
}

#[test]
fn phases_match_market_event() {
    let payloads = [
        (
            EventKind::Trade,
            EventPayload::Trade(TradeTick {
                timestamp: 0,
                price: 1.0,
                size: 1.0,
                signed_size: 0.0,
                oi: 0.0,
            }),
        ),
        (EventKind::Quote, EventPayload::Quote(QuoteTick::without_sizes(0, 1.0, 1.0))),
        (EventKind::Depth, EventPayload::Depth(DepthRef { slot: 0, timestamp: 0 })),
        (
            EventKind::Bar,
            EventPayload::Bar(OhlcvBar {
                timestamp: 0,
                open: 1.0,
                high: 1.0,
                low: 1.0,
                close: 1.0,
                volume: 0.0,
            }),
        ),
    ];
    for (kind, payload) in payloads {
        let event = MarketEvent { instrument: 0, stream: 0, payload };
        assert_eq!(kind.phase(), event.phase());
    }
}

/// One depth row: gap, price, size.
type DepthRow = (i64, f64, f64);

/// One tick row: gap, ltp, bid, ask, buy, sell, ltq, oi, bid_qty, ask_qty.
type TickRow = (i64, f64, f64, f64, f64, f64, f64, f64, f64, f64);

#[derive(Debug, Clone)]
struct Inst {
    bar_start: i64,
    bars: Vec<(i64, f64, f64)>,
    ticks: Option<(i64, Vec<TickRow>)>,
    depth: Option<(i64, Vec<DepthRow>)>,
    /// Which optional tick columns are all zero (and so absent in the store):
    /// buy, sell, oi, ltq, bid_qty, ask_qty.
    absent: [bool; 6],
}

/// Zero (missing data) about half the time, else a positive value.
fn maybe(max: f64) -> impl Strategy<Value = f64> {
    prop_oneof![Just(0.0), 1.0f64..max]
}

fn inst_strategy() -> impl Strategy<Value = Inst> {
    let row = (
        0i64..4,
        maybe(200.0),
        maybe(200.0),
        maybe(200.0),
        maybe(50.0),
        maybe(50.0),
        maybe(50.0),
        maybe(500.0),
        maybe(80.0),
        maybe(80.0),
    );
    (
        0i64..6,
        prop::collection::vec((0i64..4, 1.0f64..200.0, 0.0f64..1000.0), 0..12),
        prop::option::of((0i64..6, prop::collection::vec(row, 0..16))),
        prop::option::of((
            0i64..6,
            prop::collection::vec((0i64..4, 1.0f64..200.0, 1.0f64..90.0), 0..8),
        )),
        prop::array::uniform6(any::<bool>()),
    )
        .prop_map(|(bar_start, bars, ticks, depth, absent)| Inst {
            bar_start,
            bars,
            ticks,
            depth,
            absent,
        })
}

/// Running sums from `start`; a zero gap repeats the timestamp.
fn cumulative(start: i64, gaps: impl Iterator<Item = i64>) -> Vec<i64> {
    let mut t = start;
    gaps.map(|g| {
        t += g;
        t
    })
    .collect()
}

fn kernel_bars(inst: &Inst) -> Vec<KernelBar> {
    let ts = cumulative(inst.bar_start, inst.bars.iter().map(|b| b.0));
    inst.bars
        .iter()
        .zip(ts)
        .map(|(&(_, px, vol), timestamp)| KernelBar {
            timestamp,
            open: px,
            high: px * 1.01,
            low: px * 0.99,
            close: px * 1.005,
            volume: vol,
        })
        .collect()
}

/// The tick data the session sees (every column full length, `ltq` possibly
/// empty) and the one the store sees (all-zero columns dropped).
fn tick_data(inst: &Inst) -> Option<(TickData, TickData)> {
    let (start, rows) = inst.ticks.as_ref()?;
    let ts = cumulative(*start, rows.iter().map(|r| r.0));
    let col = |f: fn(&TickRow) -> f64, zero: bool| -> Vec<f64> {
        rows.iter().map(|r| if zero { 0.0 } else { f(r) }).collect()
    };
    let [no_buy, no_sell, no_oi, no_ltq, no_bq, no_aq] = inst.absent;
    let full = TickData {
        timestamps: ts,
        ltp: col(|r| r.1, false),
        bid: col(|r| r.2, false),
        ask: col(|r| r.3, false),
        buy_qty_delta: col(|r| r.4, no_buy),
        sell_qty_delta: col(|r| r.5, no_sell),
        ltq: if no_ltq { vec![] } else { col(|r| r.6, false) },
        oi: col(|r| r.7, no_oi),
        bid_qty: col(|r| r.8, no_bq),
        ask_qty: col(|r| r.9, no_aq),
    };
    let mut sparse = full.clone();
    for (absent, column) in [
        (no_buy, &mut sparse.buy_qty_delta),
        (no_sell, &mut sparse.sell_qty_delta),
        (no_oi, &mut sparse.oi),
        (no_bq, &mut sparse.bid_qty),
        (no_aq, &mut sparse.ask_qty),
    ] {
        if absent {
            column.clear();
        }
    }
    Some((full, sparse))
}

fn depth_ticks(inst: &Inst) -> Option<Vec<DepthTick>> {
    let (start, rows) = inst.depth.as_ref()?;
    let ts = cumulative(*start, rows.iter().map(|r| r.0));
    Some(
        rows.iter()
            .zip(ts)
            .map(|(&(_, px, size), t)| {
                let bid = [BookLevel { price: px, size }];
                let ask = [BookLevel { price: px + 1.0, size: f64::NAN }];
                DepthTick::from_levels(t, &bid, &ask)
            })
            .collect(),
    )
}

/// Comparable form of one event: instrument, local_idx, kind, then the
/// timestamp and payload fields as raw bits.
type Seen = (usize, usize, EventKind, Vec<u64>);

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn depth_bits(d: &DepthTick) -> Vec<u64> {
    let mut out = vec![d.timestamp as u64, d.bid_len as u64, d.ask_len as u64];
    for level in d.bids.iter().chain(d.asks.iter()) {
        out.extend(bits(&[level.price, level.size]));
    }
    out
}

fn with_ts(ts: i64, mut payload: Vec<u64>) -> Vec<u64> {
    payload.insert(0, ts as u64);
    payload
}

fn session_schedule(insts: &[Inst]) -> Vec<Seen> {
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let mut session = EventSession::new(config);
    for (i, inst) in insts.iter().enumerate() {
        let id = session.add_instrument(
            format!("I{i}"),
            Direction::Long,
            None,
            None,
            PositionPolicy::Net,
        );
        session.set_bars(id, kernel_bars(inst)).unwrap();
        if let Some((full, _)) = tick_data(inst) {
            session.set_ticks(id, full).unwrap();
        }
        if let Some(depth) = depth_ticks(inst) {
            session.set_depth(id, depth).unwrap();
        }
    }
    session.seal();
    let mut out = Vec::new();
    while let Some(entry) = session.current() {
        let (kind, payload) = match entry.data {
            ScheduleData::Bar(b) => {
                (EventKind::Bar, bits(&[b.open, b.high, b.low, b.close, b.volume]))
            }
            ScheduleData::Trade(t) => {
                (EventKind::Trade, bits(&[t.price, t.size, t.signed_size, t.oi]))
            }
            ScheduleData::Quote(q) => {
                (EventKind::Quote, bits(&[q.bid, q.ask, q.bid_size, q.ask_size]))
            }
            ScheduleData::Depth(h) => {
                (EventKind::Depth, depth_bits(&session.depth_at(h.slot).unwrap()))
            }
        };
        out.push((entry.instrument, entry.local_idx, kind, with_ts(entry.timestamp(), payload)));
        session.apply_current(StepInput::default());
    }
    out
}

fn build_store(insts: &[Inst]) -> MarketStore {
    let mut store = MarketStore::new();
    for (i, inst) in insts.iter().enumerate() {
        let label = format!("I{i}");
        let id = store.add_instrument(&label).unwrap();
        if !inst.bars.is_empty() {
            let bars = BarColumns::from_kernel_bars(&label, &kernel_bars(inst)).unwrap();
            store.set_bars(id, bars).unwrap();
        }
        if let Some((_, sparse)) = tick_data(inst) {
            store.set_ticks(id, TickColumns::from_tick_data(&label, &sparse).unwrap()).unwrap();
        }
        if let Some(depth) = depth_ticks(inst) {
            store.set_depth(id, DepthColumns::from_slice(&label, &depth).unwrap()).unwrap();
        }
    }
    store
}

fn cursor_schedule(store: &MarketStore) -> Vec<Seen> {
    let mut cursor = MergeCursor::new(store);
    let mut out = Vec::new();
    let mut last_bar: Vec<Option<u32>> = vec![None; store.len()];
    while let Some(e) = cursor.next() {
        let payload = match e.kind {
            EventKind::Bar => {
                let b = cursor.bar(&e).unwrap();
                assert_eq!(b.timestamp, e.ts);
                // The watermark is the row just yielded, and strictly rises.
                assert_eq!(cursor.visible_row(e.instrument), Some(e.row as u32));
                let prev = &mut last_bar[e.instrument.index()];
                assert!(prev.is_none_or(|r| u64::from(r) < e.row));
                *prev = Some(e.row as u32);
                bits(&[b.open, b.high, b.low, b.close, b.volume])
            }
            EventKind::Trade => {
                let t = cursor.trade(&e).unwrap();
                bits(&[t.price, t.size, t.signed_size, t.oi])
            }
            EventKind::Quote => {
                let q = cursor.quote(&e).unwrap();
                bits(&[q.bid, q.ask, q.bid_size, q.ask_size])
            }
            EventKind::Depth => depth_bits(&cursor.depth(&e).unwrap()),
        };
        // Only the accessor matching the kind answers.
        assert_eq!(cursor.bar(&e).is_some(), e.kind == EventKind::Bar);
        assert_eq!(cursor.trade(&e).is_some(), e.kind == EventKind::Trade);
        out.push((e.instrument.index(), e.local_idx as usize, e.kind, with_ts(e.ts, payload)));
    }
    out
}

#[test]
fn cursor_matches_the_sealed_session_schedule() {
    const CASES: u32 = 400;
    let mut runner = TestRunner::new(Config {
        cases: CASES,
        failure_persistence: None,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(0x5EED_C0DE),
        ..Config::default()
    });
    let strategy = prop::collection::vec(inst_strategy(), 1..5);
    let nonempty = std::cell::Cell::new(0usize);
    let mixed = std::cell::Cell::new(0usize);
    runner
        .run(&strategy, |insts| {
            let expected = session_schedule(&insts);
            let store = build_store(&insts);
            let actual = cursor_schedule(&store);
            if !expected.is_empty() {
                nonempty.set(nonempty.get() + 1);
            }
            let kinds: std::collections::HashSet<_> =
                expected.iter().map(|e| e.2.phase()).collect();
            if kinds.len() >= 3 {
                mixed.set(mixed.get() + 1);
            }
            prop_assert_eq!(expected, actual);
            Ok(())
        })
        .unwrap();
    // Guard against a generator that degenerates into trivial cases.
    let (nonempty, mixed) = (nonempty.get(), mixed.get());
    assert!(nonempty > CASES as usize * 9 / 10, "only {nonempty} non-empty cases");
    assert!(mixed > CASES as usize / 4, "only {mixed} cases mixed three or more kinds");
}

#[test]
fn empty_store_and_empty_streams_yield_nothing() {
    let mut store = MarketStore::new();
    assert_eq!(MergeCursor::new(&store).count(), 0);
    let id = store.add_instrument("A").unwrap();
    let empty = TickData {
        timestamps: vec![],
        ltp: vec![],
        bid: vec![],
        ask: vec![],
        buy_qty_delta: vec![],
        sell_qty_delta: vec![],
        oi: vec![],
        ltq: vec![],
        bid_qty: vec![],
        ask_qty: vec![],
    };
    store.set_ticks(id, TickColumns::from_tick_data("A", &empty).unwrap()).unwrap();
    let mut cursor = MergeCursor::new(&store);
    assert!(cursor.next().is_none());
    assert_eq!(cursor.visible_row(id), None);
}

#[test]
fn tick_watermark_follows_yielded_rows() {
    let mut store = MarketStore::new();
    let id = store.add_instrument("A").unwrap();
    let ticks = TickData {
        timestamps: vec![1, 2, 3],
        ltp: vec![10.0, 0.0, 11.0],
        bid: vec![0.0; 3],
        ask: vec![0.0; 3],
        buy_qty_delta: vec![],
        sell_qty_delta: vec![],
        oi: vec![],
        ltq: vec![1.0, 1.0, 1.0],
        bid_qty: vec![],
        ask_qty: vec![],
    };
    store.set_ticks(id, TickColumns::from_tick_data("A", &ticks).unwrap()).unwrap();
    let mut cursor = MergeCursor::new(&store);
    assert_eq!(cursor.visible_tick_row(id), None);
    assert_eq!(cursor.next().map(|e| e.row), Some(0));
    assert_eq!(cursor.visible_tick_row(id), Some(0));
    // Row 1 has no last price and no book: it never yields.
    assert_eq!(cursor.next().map(|e| e.row), Some(2));
    assert_eq!(cursor.visible_tick_row(id), Some(2));
    assert!(cursor.next().is_none());
}

#[test]
fn a_filtered_cursor_is_the_full_one_minus_the_dropped_instruments() {
    let mut store = MarketStore::new();
    for (name, offset) in [("A", 0i64), ("B", 1), ("C", 0)] {
        let id = store.add_instrument(name).unwrap();
        let ts: Vec<i64> = (0..6).map(|i| i * 2 + offset).collect();
        let px = vec![1.0; ts.len()];
        store
            .set_bars(
                id,
                BarColumns::from_vecs(name, ts, px.clone(), px.clone(), px.clone(), px.clone(), px)
                    .unwrap(),
            )
            .unwrap();
    }
    let keep = |id: crate::runtime::ids::InstrumentId| id.index() != 1;
    let full: Vec<EventRef> = MergeCursor::new(&store).filter(|e| keep(e.instrument)).collect();
    let filtered: Vec<EventRef> = MergeCursor::filtered(&store, keep).collect();
    assert_eq!(full.len(), 12);
    assert_eq!(filtered, full);
}
