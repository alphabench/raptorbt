//! A run cut into windows delivers the events of the whole tape: the same order, groups,
//! payloads, rows and ordinals.

use super::*;
use crate::accounts::AccountMode;
use crate::core::types::BacktestConfig;
use crate::data::book::{BookLevel, DepthTick};
use crate::portfolio::ledger::PositionPolicy;
use crate::runtime::runner::{InstrumentParams, RuntimeBuilder};
use crate::runtime::store::{BarColumns, DepthColumns, TickColumns, TickSlices};

const INSTRUMENTS: usize = 4;

/// A small deterministic generator.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// One instrument's rows before they are put into a store.
#[derive(Clone, Default)]
struct Raw {
    /// `(ts, close)`; the other columns follow from it.
    bars: Vec<(i64, f64)>,
    /// `(ts, ltp, bid, ask)`; a zero is "no print" or "no quote", so a row can be neither.
    ticks: Vec<(i64, f64, f64, f64)>,
    depth: Vec<(i64, f64)>,
}

/// Instruments 0 and 1 have bars only, 2 has bars and ticks, 3 has ticks and depth. Timestamps
/// are on a coarse grid so that many events, of every kind, share a time, and a time may repeat
/// inside one instrument.
fn make_raw(seed: u64) -> Vec<Raw> {
    let mut rng = Lcg(seed);
    (0..INSTRUMENTS)
        .map(|i| {
            let mut raw = Raw::default();
            let mut t = 1 + rng.below(3) as i64;
            if i < 3 {
                for _ in 0..30 + rng.below(30) {
                    raw.bars.push((t, 100.0 + rng.below(500) as f64 / 10.0));
                    t += rng.below(4) as i64;
                }
            }
            let mut t = 1 + rng.below(3) as i64;
            if i >= 2 {
                for _ in 0..60 + rng.below(60) {
                    let px = |rng: &mut Lcg| match rng.below(4) {
                        0 => 0.0,
                        _ => 50.0 + rng.below(100) as f64,
                    };
                    raw.ticks.push((t, px(&mut rng), px(&mut rng), px(&mut rng)));
                    t += rng.below(3) as i64;
                }
            }
            let mut t = 1 + rng.below(3) as i64;
            if i == 3 {
                for _ in 0..20 {
                    raw.depth.push((t, 40.0 + rng.below(10) as f64));
                    t += rng.below(4) as i64;
                }
            }
            raw
        })
        .collect()
}

/// A store of the rows stamped in `(after, upto]`, for every instrument, in the run's order.
fn store_of(raws: &[Raw], after: i64, upto: i64) -> MarketStore {
    let within = |t: i64| after < t && t <= upto;
    let mut store = MarketStore::new();
    for (i, raw) in raws.iter().enumerate() {
        let id = store.add_instrument(&format!("S{i}")).unwrap();
        let bars: Vec<&(i64, f64)> = raw.bars.iter().filter(|b| within(b.0)).collect();
        if !raw.bars.is_empty() {
            let col = |f: fn(f64) -> f64| bars.iter().map(|b| f(b.1)).collect::<Vec<_>>();
            store
                .set_bars(
                    id,
                    BarColumns::from_vecs(
                        "bars",
                        bars.iter().map(|b| b.0).collect(),
                        col(|c| c - 0.5),
                        col(|c| c + 1.0),
                        col(|c| c - 1.0),
                        col(|c| c),
                        col(|c| c * 10.0),
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        if !raw.ticks.is_empty() {
            let ticks: Vec<&(i64, f64, f64, f64)> =
                raw.ticks.iter().filter(|t| within(t.0)).collect();
            let ts: Vec<i64> = ticks.iter().map(|t| t.0).collect();
            let ltp: Vec<f64> = ticks.iter().map(|t| t.1).collect();
            let bid: Vec<f64> = ticks.iter().map(|t| t.2).collect();
            let ask: Vec<f64> = ticks.iter().map(|t| t.3).collect();
            let slices = TickSlices {
                ts: &ts,
                ltp: &ltp,
                bid: Some(&bid),
                ask: Some(&ask),
                ..TickSlices::default()
            };
            store.set_ticks(id, TickColumns::from_slices("ticks", &slices).unwrap()).unwrap();
        }
        if !raw.depth.is_empty() {
            let snaps: Vec<DepthTick> = raw
                .depth
                .iter()
                .filter(|d| within(d.0))
                .map(|&(t, px)| {
                    DepthTick::from_levels(
                        t,
                        &[BookLevel { price: px, size: 5.0 }],
                        &[BookLevel { price: px + 1.0, size: 6.0 }],
                    )
                })
                .collect();
            store.set_depth(id, DepthColumns::from_slice("depth", &snaps).unwrap()).unwrap();
        }
    }
    store
}

fn core_over(store: &MarketStore, subscribed: &[u32]) -> RuntimeCore {
    let params = (0..store.len()).map(|_| InstrumentParams::default()).collect();
    let mut b = RuntimeBuilder::new(
        store.clone(),
        params,
        BacktestConfig::default(),
        AccountMode::Cash,
        PositionPolicy::Net,
    )
    .unwrap();
    let ids: Vec<InstrumentId> = subscribed.iter().map(|&i| InstrumentId::new(i)).collect();
    b.add_strategy(&ids).unwrap();
    b.build().unwrap()
}

/// What an event is, with its payload, as text: row and ordinal included.
fn show(event: &EventRef, payload: &Pushed) -> String {
    format!("{event:?} {payload:?}")
}

/// Every event of `store` as the run's own cursor gives it, with the `(timestamp, phase)` of the
/// group it is in.
fn whole(store: &MarketStore, core: &RuntimeCore) -> Vec<((i64, u8), String)> {
    let mut cursor = MergeCursor::filtered(store, |id| !core.subscribers(id).is_empty());
    let mut out = Vec::new();
    while let Some(key) = cursor.peek_group() {
        let (event, payload) = RuntimeCore::pull(&mut cursor).unwrap().unwrap();
        out.push((key, show(&event, &payload)));
    }
    out
}

/// The same, read in windows cut after each timestamp of `cuts` (ascending).
fn windowed(raws: &[Raw], cuts: &[i64], core: &mut RuntimeCore) -> Vec<((i64, u8), String)> {
    let mut out = Vec::new();
    let mut after = i64::MIN;
    for &upto in cuts.iter().chain(std::iter::once(&i64::MAX)) {
        let store = store_of(raws, after, upto);
        let mut cursor = core.advance_window(&store).unwrap();
        while let Some(key) = cursor.peek_group() {
            let (event, payload) = cursor.pull().unwrap().unwrap();
            out.push((key, show(&event, &payload)));
        }
        core.end_window(cursor).unwrap();
        after = upto;
    }
    out
}

fn last_ts(raws: &[Raw]) -> i64 {
    raws.iter()
        .flat_map(|r| {
            r.bars
                .iter()
                .map(|b| b.0)
                .chain(r.ticks.iter().map(|t| t.0))
                .chain(r.depth.iter().map(|d| d.0))
        })
        .max()
        .unwrap()
}

#[test]
fn windows_cut_anywhere_deliver_the_events_of_the_whole_tape() {
    for seed in 0..40 {
        let raws = make_raw(seed);
        let full = store_of(&raws, i64::MIN, i64::MAX);
        let want = whole(&full, &core_over(&full, &[0, 1, 2, 3]));
        assert!(want.len() > 100, "the tape is not trivial");
        let end = last_ts(&raws);
        let mut rng = Lcg(seed + 1000);
        // Cut after every timestamp, after every few, after random ones, and never.
        let every: Vec<i64> = (0..end).collect();
        let few: Vec<i64> = (0..end).step_by(7).collect();
        let mut random: Vec<i64> = (0..8).map(|_| rng.below(end as u64) as i64).collect();
        random.sort_unstable();
        random.dedup();
        for (what, cuts) in [("every", every), ("few", few), ("random", random), ("none", vec![])] {
            let mut core = core_over(&full, &[0, 1, 2, 3]);
            let got = windowed(&raws, &cuts, &mut core);
            assert_eq!(got.len(), want.len(), "seed {seed}, cut {what}: events");
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g, w, "seed {seed}, cut {what}, event {i}");
            }
        }
    }
}

#[test]
fn an_instrument_nobody_subscribes_to_is_skipped_in_a_window_as_in_the_run() {
    let raws = make_raw(7);
    let full = store_of(&raws, i64::MIN, i64::MAX);
    let want = whole(&full, &core_over(&full, &[1, 2]));
    let mut core = core_over(&full, &[1, 2]);
    let got = windowed(&raws, &[10, 11, 40], &mut core);
    assert_eq!(got, want);
    assert!(got.iter().all(|(_, e)| !e.contains("instrument: InstrumentId(0)")));
}

#[test]
fn the_base_counts_rows_of_each_kind_and_events_across_windows() {
    let raws = make_raw(3);
    let full = store_of(&raws, i64::MIN, i64::MAX);
    let mut core = core_over(&full, &[0, 1, 2, 3]);
    let mut whole_core = core_over(&full, &[0, 1, 2, 3]);
    windowed(&raws, &[5, 20], &mut core);
    let mut one = whole_core.advance_window(&full).unwrap();
    while one.pull().unwrap().is_some() {}
    whole_core.end_window(one).unwrap();
    for (i, raw) in raws.iter().enumerate() {
        let id = InstrumentId::new(i as u32);
        let (a, b) = (core.window_base(), whole_core.window_base());
        assert_eq!(a.bar_rows(id), raw.bars.len() as u64);
        assert_eq!(a.tick_rows(id), raw.ticks.len() as u64);
        assert_eq!(a.depth_rows(id), raw.depth.len() as u64);
        assert_eq!(a.events(id), b.events(id), "instrument {i}");
    }
    assert_eq!(core.window_base().windows(), 3);
    assert_eq!(core.window_base().last_ts(), whole_core.window_base().last_ts());
}

#[test]
fn a_window_that_starts_at_or_before_the_last_event_is_refused_and_changes_nothing() {
    let raws = make_raw(11);
    let full = store_of(&raws, i64::MIN, i64::MAX);
    let mut core = core_over(&full, &[0, 1, 2, 3]);
    let first = store_of(&raws, i64::MIN, 12);
    let mut cursor = core.advance_window(&first).unwrap();
    while cursor.pull().unwrap().is_some() {}
    core.end_window(cursor).unwrap();
    let last = core.window_base().last_ts().expect("the first window had events");
    let before = core.window_base().clone();

    // The same rows again: they overlap.
    match core.advance_window(&first) {
        Err(WindowError::Overlap { first_ts, last_ts }) => {
            assert!(first_ts <= last_ts && last_ts == last)
        }
        other => panic!("expected an overlap, got {:?}", other.err()),
    }
    // A window that splits the last timestamp: it starts at `last`.
    let split = store_of(&raws, last - 1, last + 3);
    assert!(matches!(core.advance_window(&split), Err(WindowError::Overlap { .. })));
    assert_eq!(core.window_base(), &before, "a refused window is not counted");

    // The window that follows is accepted.
    let next = store_of(&raws, last, i64::MAX);
    let mut cursor = core.advance_window(&next).unwrap();
    while cursor.pull().unwrap().is_some() {}
    core.end_window(cursor).unwrap();
}

#[test]
fn empty_windows_and_a_window_of_unsubscribed_rows_are_fine() {
    let raws = make_raw(5);
    let full = store_of(&raws, i64::MIN, i64::MAX);
    let mut core = core_over(&full, &[1]);
    let empty = store_of(&raws, i64::MAX - 1, i64::MAX);
    let mut cursor = core.advance_window(&empty).unwrap();
    assert!(cursor.pull().unwrap().is_none());
    core.end_window(cursor).unwrap();
    assert_eq!(core.window_base().last_ts(), None);
    // Rows of an instrument nobody reads still count as rows.
    let all = store_of(&raws, i64::MIN, i64::MAX);
    let mut cursor = core.advance_window(&all).unwrap();
    while cursor.pull().unwrap().is_some() {}
    core.end_window(cursor).unwrap();
    assert_eq!(core.window_base().bar_rows(InstrumentId::new(0)), raws[0].bars.len() as u64);
}

#[test]
fn a_window_must_be_read_to_its_end_before_it_is_closed() {
    let raws = make_raw(2);
    let full = store_of(&raws, i64::MIN, i64::MAX);
    let mut core = core_over(&full, &[0, 1, 2, 3]);
    let mut cursor = core.advance_window(&full).unwrap();
    cursor.pull().unwrap().unwrap();
    let err = core.end_window(cursor).unwrap_err().to_string();
    assert!(err.contains("every event"), "{err}");
}

#[test]
fn a_window_over_other_instruments_than_the_run_is_refused() {
    let raws = make_raw(2);
    let full = store_of(&raws, i64::MIN, i64::MAX);
    let mut core = core_over(&full, &[0]);
    let short = store_of(&raws[..2], i64::MIN, i64::MAX);
    assert!(matches!(core.advance_window(&short), Err(WindowError::Other(_))));
}
