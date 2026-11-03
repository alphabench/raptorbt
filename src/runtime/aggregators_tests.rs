//! The hub must hand every strategy exactly what private builders would.

use std::collections::HashSet;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed, TestRunner};

use crate::core::types::OhlcvBar;
use crate::data::{
    builder_for_with, AggregationUnit, BarBuilder, BarSpec, BarStamp, BuilderParams, SourceRecord,
    IST_OFFSET_NS,
};
use crate::portfolio::kernel::KernelBar;
use crate::runtime::aggregators::{AggKey, AggregatorHub, AggregatorHubBuilder};
use crate::runtime::ids::{AggId, InstrumentId, StrategyId};

const MINUTE_NS: i64 = 60_000_000_000;
const NEG_5H_NS: i64 = -5 * 3_600_000_000_000;

/// A raw subscription as a strategy would declare it.
#[derive(Debug, Clone, Copy)]
struct Sub {
    instrument: u32,
    step: u32,
    unit: AggregationUnit,
    brick: f64,
    tz: i64,
    stamp: BarStamp,
}

type Bits = (i64, u64, u64, u64, u64, u64);

fn bits(b: &OhlcvBar) -> Bits {
    (
        b.timestamp,
        b.open.to_bits(),
        b.high.to_bits(),
        b.low.to_bits(),
        b.close.to_bits(),
        b.volume.to_bits(),
    )
}

fn sub_strategy(instruments: u32) -> impl Strategy<Value = Sub> {
    let unit_and_step = prop_oneof![
        (1u32..4).prop_map(|s| (s * 5, AggregationUnit::Minute)),
        (1u32..3).prop_map(|s| (s, AggregationUnit::Hour)),
        (1u32..3).prop_map(|s| (s, AggregationUnit::Day)),
        (1u32..4).prop_map(|s| (s, AggregationUnit::Tick)),
        (1u32..4).prop_map(|s| (s * 40, AggregationUnit::Volume)),
        (1u32..4).prop_map(|s| (s * 3000, AggregationUnit::Value)),
        (1u32..3).prop_map(|s| (s, AggregationUnit::Renko)),
    ];
    let brick = prop_oneof![
        Just(0.0),
        Just(-0.0),
        Just(f64::NAN),
        Just(-1.0),
        Just(0.5),
        Just(1.0),
        Just(2.0)
    ];
    let tz = prop_oneof![Just(0), Just(IST_OFFSET_NS), Just(NEG_5H_NS)];
    let stamp = prop_oneof![Just(BarStamp::Open), Just(BarStamp::Close)];
    (0..instruments, unit_and_step, brick, tz, stamp).prop_map(
        |(instrument, (step, unit), brick, tz, stamp)| Sub {
            instrument,
            step,
            unit,
            brick,
            tz,
            stamp,
        },
    )
}

/// One primary bar event: instrument, ts gap in minutes, return, range, volume.
type Raw = (u32, u8, f64, f64, f64);

#[derive(Debug, Clone)]
struct Case {
    instruments: u32,
    strategies: Vec<Vec<Sub>>,
    events: Vec<Raw>,
}

fn case_strategy() -> impl Strategy<Value = Case> {
    (1u32..4).prop_flat_map(|instruments| {
        let subs =
            prop::collection::vec(prop::collection::vec(sub_strategy(instruments), 0..7), 1..9);
        // Gap 0 repeats a timestamp; 5% of gaps jump a session (~16 h).
        let gap = prop_oneof![
            20 => Just(1u8), 4 => Just(0u8), 3 => 2u8..10, 1 => Just(255u8)
        ];
        let event = (0..instruments, gap, -3.0f64..3.0, 0.0f64..2.0, 0.0f64..120.0);
        (subs, prop::collection::vec(event, 0..260)).prop_map(move |(strategies, events)| Case {
            instruments,
            strategies,
            events,
        })
    })
}

/// Materialise per-instrument bars, in event order.
fn bars_of(case: &Case) -> Vec<(InstrumentId, KernelBar)> {
    let n = case.instruments as usize;
    let mut ts = vec![1_700_000_000_000_000_000i64; n];
    let mut px = vec![100.0f64; n];
    case.events
        .iter()
        .map(|&(inst, gap, ret, range, vol)| {
            let i = inst as usize;
            ts[i] += if gap == 255 { 960 * MINUTE_NS } else { gap as i64 * MINUTE_NS };
            let open = px[i];
            let close = (open + ret).max(1.0);
            px[i] = close;
            let bar = KernelBar {
                timestamp: ts[i],
                open,
                high: open.max(close) + range,
                low: (open.min(close) - range).max(0.5),
                close,
                volume: vol.round(),
            };
            (InstrumentId::new(inst), bar)
        })
        .collect()
}

/// Today's path: one private builder per subscription, fed in subscription
/// order, draining `next_pending` after each completed bar.
fn private_run(subs: &[Sub], bars: &[(InstrumentId, KernelBar)]) -> Vec<(u16, Bits)> {
    let mut builders: Vec<Box<dyn BarBuilder + Send + Sync>> = subs
        .iter()
        .map(|s| {
            let spec = BarSpec::new(s.step, s.unit).unwrap();
            builder_for_with(spec, s.tz, BuilderParams { brick_size: s.brick }).unwrap()
        })
        .collect();
    let mut out = Vec::new();
    for (inst, bar) in bars {
        for (local, sub) in subs.iter().enumerate() {
            if sub.instrument != inst.raw() {
                continue;
            }
            let rec = SourceRecord {
                timestamp: sub.stamp.bucket_ts(sub.unit, bar.timestamp),
                open: bar.open,
                high: bar.high,
                low: bar.low,
                close: bar.close,
                volume: bar.volume,
                signed_volume: 0.0,
            };
            let mut done = builders[local].push(&rec);
            while let Some(b) = done {
                out.push((local as u16, bits(&b)));
                done = builders[local].next_pending();
            }
        }
    }
    out
}

/// What the key should be, stated without going through `AggKey`.
fn expected_key(s: &Sub) -> (u32, u32, AggregationUnit, u64, i64, bool) {
    let time =
        matches!(s.unit, AggregationUnit::Minute | AggregationUnit::Hour | AggregationUnit::Day);
    let brick = if s.unit == AggregationUnit::Renko {
        if s.brick > 0.0 {
            s.brick.to_bits()
        } else {
            (s.step as f64).to_bits()
        }
    } else {
        0
    };
    (
        s.instrument,
        s.step,
        s.unit,
        brick,
        if time { s.tz } else { 0 },
        time && s.stamp == BarStamp::Close,
    )
}

fn register_all(case: &Case) -> (AggregatorHub, usize) {
    let mut builder = AggregatorHubBuilder::new(case.instruments as usize);
    let mut distinct = HashSet::new();
    for (s, subs) in case.strategies.iter().enumerate() {
        for (local, sub) in subs.iter().enumerate() {
            let key = AggKey::new(
                InstrumentId::new(sub.instrument),
                BarSpec::new(sub.step, sub.unit).unwrap(),
                sub.brick,
                sub.tz,
                sub.stamp,
            )
            .unwrap();
            let before = distinct.len();
            distinct.insert(expected_key(sub));
            let (agg, got_local) = builder.register(StrategyId::new(s as u32), key).unwrap();
            assert_eq!(got_local as usize, local);
            if distinct.len() > before {
                // A new key takes the next id: first-registration order.
                assert_eq!(agg.index(), before);
            } else {
                assert!(agg.index() < before);
            }
        }
    }
    (builder.build().unwrap(), distinct.len())
}

#[test]
fn hub_matches_private_aggregators() {
    const CASES: u32 = 400;
    let mut runner = TestRunner::new(Config {
        cases: CASES,
        failure_persistence: None,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(0xA66_5EED),
        ..Config::default()
    });
    let shared_cases = std::cell::Cell::new(0usize);
    let delivered = std::cell::Cell::new(0usize);
    runner
        .run(&case_strategy(), |case| {
            let bars = bars_of(&case);
            let (mut hub, distinct) = register_all(&case);
            prop_assert_eq!(hub.aggregators(), distinct);
            let total: usize = case.strategies.iter().map(Vec::len).sum();
            prop_assert_eq!(hub.subscriptions(), total);
            if distinct < total {
                shared_cases.set(shared_cases.get() + 1);
            }

            let mut got: Vec<Vec<(u16, Bits)>> = vec![Vec::new(); case.strategies.len()];
            for (inst, bar) in &bars {
                hub.on_bar(*inst, bar);
                for (s, sink) in got.iter_mut().enumerate() {
                    for (local, b) in hub.deliveries(StrategyId::new(s as u32)) {
                        sink.push((local, bits(b)));
                    }
                }
            }
            for (s, subs) in case.strategies.iter().enumerate() {
                let want = private_run(subs, &bars);
                delivered.set(delivered.get() + want.len());
                prop_assert_eq!(&got[s], &want, "strategy {}", s);
            }
            Ok(())
        })
        .unwrap();
    // The generator must actually exercise sharing and deliveries.
    assert!(shared_cases.get() > CASES as usize / 4, "{}", shared_cases.get());
    assert!(delivered.get() > 10_000, "{}", delivered.get());
}

fn instrument_key(instrument: u32, tz: i64) -> AggKey {
    AggKey::new(
        InstrumentId::new(instrument),
        BarSpec::new(1, AggregationUnit::Day).unwrap(),
        0.0,
        tz,
        BarStamp::Open,
    )
    .unwrap()
}

#[test]
fn output_affecting_parameters_split_aggregators() {
    let day = |tz, stamp| {
        AggKey::new(
            InstrumentId::new(0),
            BarSpec::new(1, AggregationUnit::Day).unwrap(),
            0.0,
            tz,
            stamp,
        )
        .unwrap()
    };
    assert_ne!(day(0, BarStamp::Open), day(IST_OFFSET_NS, BarStamp::Open));
    assert_ne!(day(0, BarStamp::Open), day(0, BarStamp::Close));
    assert_ne!(instrument_key(0, 0), instrument_key(1, 0));

    let renko = |brick: f64, step: u32| {
        AggKey::new(
            InstrumentId::new(0),
            BarSpec::new(step, AggregationUnit::Renko).unwrap(),
            brick,
            0,
            BarStamp::Open,
        )
        .unwrap()
    };
    assert_ne!(renko(0.5, 1), renko(1.0, 1));
    // Every non-positive or NaN brick means "use the step".
    assert_eq!(renko(0.0, 2), renko(-0.0, 2));
    assert_eq!(renko(f64::NAN, 2), renko(-3.0, 2));
    assert_eq!(renko(0.0, 2), renko(2.0, 2));
    assert!(AggKey::new(
        InstrumentId::new(0),
        BarSpec::new(1, AggregationUnit::Renko).unwrap(),
        f64::INFINITY,
        0,
        BarStamp::Open
    )
    .is_err());

    // Parameters a unit ignores cannot split it.
    let tick = |brick: f64, tz: i64, stamp| {
        AggKey::new(
            InstrumentId::new(0),
            BarSpec::new(3, AggregationUnit::Tick).unwrap(),
            brick,
            tz,
            stamp,
        )
        .unwrap()
    };
    assert_eq!(tick(f64::NAN, IST_OFFSET_NS, BarStamp::Close), tick(1.0, 0, BarStamp::Open));
}

fn bar(ts_min: i64, px: f64) -> KernelBar {
    KernelBar { timestamp: ts_min * MINUTE_NS, open: px, high: px, low: px, close: px, volume: 1.0 }
}

#[test]
fn duplicate_subscription_delivers_twice_and_ids_follow_registration() {
    let key = |step| {
        AggKey::new(
            InstrumentId::new(0),
            BarSpec::new(step, AggregationUnit::Tick).unwrap(),
            0.0,
            0,
            BarStamp::Open,
        )
        .unwrap()
    };
    let mut b = AggregatorHubBuilder::new(1);
    let s = StrategyId::new(0);
    assert_eq!(b.register(s, key(2)).unwrap(), (AggId::new(0), 0));
    assert_eq!(b.register(s, key(1)).unwrap(), (AggId::new(1), 1));
    assert_eq!(b.register(s, key(2)).unwrap(), (AggId::new(0), 2));
    assert_eq!(b.register(StrategyId::new(1), key(1)).unwrap(), (AggId::new(1), 0));
    let mut hub = b.build().unwrap();
    assert_eq!((hub.aggregators(), hub.subscriptions()), (2, 4));
    assert_eq!(hub.subscribers(AggId::new(1)), &[(StrategyId::new(0), 1), (StrategyId::new(1), 0)]);

    let i = InstrumentId::new(0);
    hub.on_bar(i, &bar(0, 1.0));
    // First tick: only the 1-tick builder completes.
    let first: Vec<u16> = hub.deliveries(s).map(|(l, _)| l).collect();
    assert_eq!(first, [1]);
    hub.on_bar(i, &bar(1, 1.0));
    let second: Vec<u16> = hub.deliveries(s).map(|(l, _)| l).collect();
    assert_eq!(second, [0, 1, 2]);
    assert_eq!(hub.push_calls(), 4);
}

#[test]
fn unsubscribed_instrument_and_strategy_see_nothing() {
    let mut b = AggregatorHubBuilder::new(3);
    let key = AggKey::new(
        InstrumentId::new(1),
        BarSpec::new(1, AggregationUnit::Tick).unwrap(),
        0.0,
        0,
        BarStamp::Open,
    )
    .unwrap();
    b.register(StrategyId::new(2), key).unwrap();
    let mut hub = b.build().unwrap();
    assert!(hub.on_bar(InstrumentId::new(0), &bar(0, 1.0)).is_empty());
    assert!(hub.on_bar(InstrumentId::new(7), &bar(0, 1.0)).is_empty());
    assert_eq!(hub.deliveries(StrategyId::new(2)).count(), 0);
    assert_eq!(hub.deliveries(StrategyId::new(9)).count(), 0);
    hub.on_bar(InstrumentId::new(1), &bar(0, 1.0));
    assert_eq!(hub.deliveries(StrategyId::new(2)).count(), 1);
    // Switching instruments hides stale emissions.
    hub.on_bar(InstrumentId::new(0), &bar(1, 1.0));
    assert_eq!(hub.deliveries(StrategyId::new(2)).count(), 0);
}

#[test]
fn registration_refuses_bad_input() {
    let mut b = AggregatorHubBuilder::new(1);
    let key = AggKey::new(
        InstrumentId::new(1),
        BarSpec::new(1, AggregationUnit::Tick).unwrap(),
        0.0,
        0,
        BarStamp::Open,
    )
    .unwrap();
    assert!(b.register(StrategyId::new(0), key).is_err());
}

#[test]
fn in_progress_bars_are_not_flushed() {
    let mut b = AggregatorHubBuilder::new(1);
    let key = AggKey::new(
        InstrumentId::new(0),
        BarSpec::new(5, AggregationUnit::Minute).unwrap(),
        0.0,
        0,
        BarStamp::Open,
    )
    .unwrap();
    b.register(StrategyId::new(0), key).unwrap();
    let mut hub = b.build().unwrap();
    let mut seen = 0;
    for m in 0..4 {
        hub.on_bar(InstrumentId::new(0), &bar(m, 1.0));
        seen += hub.deliveries(StrategyId::new(0)).count();
    }
    assert_eq!(seen, 0);
}

#[test]
fn steady_state_does_not_grow_buffers() {
    let mut b = AggregatorHubBuilder::new(1);
    for step in 1..4 {
        let key = AggKey::new(
            InstrumentId::new(0),
            BarSpec::new(step, AggregationUnit::Minute).unwrap(),
            0.0,
            0,
            BarStamp::Open,
        )
        .unwrap();
        b.register(StrategyId::new(0), key).unwrap();
    }
    let mut hub = b.build().unwrap();
    for m in 0..600 {
        hub.on_bar(InstrumentId::new(0), &bar(m, 1.0));
    }
    let settled = crate::runtime::HeapSize::heap_bytes(&hub);
    for m in 600..6000 {
        hub.on_bar(InstrumentId::new(0), &bar(m, 1.0));
    }
    assert_eq!(crate::runtime::HeapSize::heap_bytes(&hub), settled);
}

// -- trade prints ---------------------------------------------------------------

/// A unit and step a tick run can aggregate prints into, flow units included.
fn print_unit() -> impl Strategy<Value = (u32, AggregationUnit)> {
    prop_oneof![
        (1u32..4).prop_map(|s| (s * 10, AggregationUnit::Second)),
        (1u32..3).prop_map(|s| (s, AggregationUnit::Minute)),
        (1u32..3).prop_map(|s| (s, AggregationUnit::Day)),
        (1u32..4).prop_map(|s| (s * 2, AggregationUnit::Tick)),
        (1u32..4).prop_map(|s| (s * 50, AggregationUnit::Volume)),
        (1u32..4).prop_map(|s| (s * 5000, AggregationUnit::Value)),
        (1u32..3).prop_map(|s| (s, AggregationUnit::Renko)),
        (2u32..5).prop_map(|s| (s, AggregationUnit::TickImbalance)),
        (2u32..5).prop_map(|s| (s * 40, AggregationUnit::VolumeImbalance)),
    ]
}

fn print_sub(instruments: u32) -> impl Strategy<Value = Sub> {
    let brick = prop_oneof![Just(0.0), Just(f64::NAN), Just(0.5), Just(2.0)];
    let tz = prop_oneof![Just(0), Just(IST_OFFSET_NS)];
    let stamp = prop_oneof![Just(BarStamp::Open), Just(BarStamp::Close)];
    (0..instruments, print_unit(), brick, tz, stamp).prop_map(
        |(instrument, (step, unit), brick, tz, stamp)| Sub {
            instrument,
            step,
            unit,
            brick,
            tz,
            stamp,
        },
    )
}

/// One print: instrument, gap in seconds (0 repeats a timestamp), return, size.
type Print = (u32, u8, f64, f64);

#[derive(Debug, Clone)]
struct PrintCase {
    instruments: u32,
    strategies: Vec<Vec<Sub>>,
    /// Primary bars per instrument.
    primary: Vec<Option<(u32, AggregationUnit, i64)>>,
    prints: Vec<Print>,
}

fn print_case() -> impl Strategy<Value = PrintCase> {
    (1u32..4).prop_flat_map(|instruments| {
        let subs = prop::collection::vec(prop::collection::vec(print_sub(instruments), 0..6), 1..6);
        let primary = prop::collection::vec(
            prop::option::of((print_unit(), prop_oneof![Just(0), Just(IST_OFFSET_NS)])),
            instruments as usize,
        );
        let gap = prop_oneof![20 => 1u8..4, 5 => Just(0u8), 1 => Just(255u8)];
        let print = (0..instruments, gap, -2.0f64..2.0, 0.0f64..90.0);
        (subs, primary, prop::collection::vec(print, 0..320)).prop_map(
            move |(strategies, primary, prints)| PrintCase {
                instruments,
                strategies,
                primary: primary.into_iter().map(|p| p.map(|((s, u), tz)| (s, u, tz))).collect(),
                prints,
            },
        )
    })
}

/// The prints as `(instrument, ts, price, size)`, timestamps per instrument.
fn prints_of(case: &PrintCase) -> Vec<(InstrumentId, i64, f64, f64)> {
    let n = case.instruments as usize;
    let mut ts = vec![1_700_000_000_000_000_000i64; n];
    let mut px = vec![100.0f64; n];
    case.prints
        .iter()
        .map(|&(inst, gap, ret, size)| {
            let i = inst as usize;
            ts[i] += if gap == 255 { 960 * MINUTE_NS } else { gap as i64 * 1_000_000_000 };
            px[i] = (px[i] + ret).max(1.0);
            (InstrumentId::new(inst), ts[i], px[i], size.round())
        })
        .collect()
}

#[test]
fn trade_feed_matches_private_push_trade() {
    const CASES: u32 = 300;
    let mut runner = TestRunner::new(Config {
        cases: CASES,
        failure_persistence: None,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(0x7_4ADE),
        ..Config::default()
    });
    let delivered = std::cell::Cell::new(0usize);
    let primaries = std::cell::Cell::new(0usize);
    runner
        .run(&print_case(), |case| {
            let prints = prints_of(&case);
            let mut builder = AggregatorHubBuilder::new(case.instruments as usize);
            for (s, subs) in case.strategies.iter().enumerate() {
                for sub in subs {
                    let key = AggKey::new(
                        InstrumentId::new(sub.instrument),
                        BarSpec::new(sub.step, sub.unit).unwrap(),
                        sub.brick,
                        sub.tz,
                        sub.stamp,
                    )
                    .unwrap();
                    builder.register(StrategyId::new(s as u32), key).unwrap();
                }
            }
            for (i, p) in case.primary.iter().enumerate() {
                if let Some((step, unit, tz)) = p {
                    let spec = BarSpec::new(*step, *unit).unwrap();
                    builder.set_primary(InstrumentId::new(i as u32), spec, *tz).unwrap();
                }
            }
            let mut hub = builder.build().unwrap();

            // `BarAggregator.push_trade(ts, price, size)` per subscription,
            // draining `next_pending`; the primary keeps only the first bar.
            let mut private: Vec<Vec<Box<dyn BarBuilder + Send + Sync>>> = case
                .strategies
                .iter()
                .map(|subs| {
                    subs.iter()
                        .map(|s| {
                            let spec = BarSpec::new(s.step, s.unit).unwrap();
                            builder_for_with(spec, s.tz, BuilderParams { brick_size: s.brick })
                                .unwrap()
                        })
                        .collect()
                })
                .collect();
            let mut primary_private: Vec<Option<Box<dyn BarBuilder + Send + Sync>>> = case
                .primary
                .iter()
                .map(|p| {
                    p.map(|(step, unit, tz)| {
                        let spec = BarSpec::new(step, unit).unwrap();
                        builder_for_with(spec, tz, BuilderParams { brick_size: 0.0 }).unwrap()
                    })
                })
                .collect();

            for &(inst, ts, price, size) in &prints {
                let rec = SourceRecord::signed_trade(ts, price, size, 0.0);
                let want_primary =
                    primary_private[inst.index()].as_mut().and_then(|b| b.push(&rec));
                let got_primary = hub.primary_trade(inst, ts, price, size);
                prop_assert_eq!(got_primary.as_ref().map(bits), want_primary.as_ref().map(bits));
                primaries.set(primaries.get() + usize::from(got_primary.is_some()));

                hub.on_trade(inst, ts, price, size);
                for (s, subs) in case.strategies.iter().enumerate() {
                    let mut want = Vec::new();
                    for (local, sub) in subs.iter().enumerate() {
                        if sub.instrument != inst.raw() {
                            continue;
                        }
                        let mut done = private[s][local].push(&rec);
                        while let Some(b) = done {
                            want.push((local as u16, bits(&b)));
                            done = private[s][local].next_pending();
                        }
                    }
                    let got: Vec<_> = hub
                        .deliveries(StrategyId::new(s as u32))
                        .map(|(l, b)| (l, bits(b)))
                        .collect();
                    delivered.set(delivered.get() + want.len());
                    prop_assert_eq!(got, want, "strategy {}", s);
                }
            }
            Ok(())
        })
        .unwrap();
    assert!(delivered.get() > 5_000, "{}", delivered.get());
    assert!(primaries.get() > 1_000, "{}", primaries.get());
}

#[test]
fn a_quote_or_book_between_prints_delivers_nothing() {
    let mut builder = AggregatorHubBuilder::new(1);
    let spec = BarSpec::new(2, AggregationUnit::Tick).unwrap();
    let key = AggKey::new(InstrumentId::new(0), spec, 0.0, 0, BarStamp::Open).unwrap();
    builder.register(StrategyId::new(0), key).unwrap();
    let mut hub = builder.build().unwrap();
    let inst = InstrumentId::new(0);
    hub.on_trade(inst, 1, 10.0, 1.0);
    assert_eq!(hub.on_trade(inst, 2, 11.0, 1.0).len(), 1);
    assert_eq!(hub.deliveries(StrategyId::new(0)).count(), 1);
    hub.skip(inst);
    assert!(hub.emissions().is_empty());
    assert_eq!(hub.deliveries(StrategyId::new(0)).count(), 0);
    // No primary configured: no primary bar.
    assert!(hub.primary_trade(inst, 3, 12.0, 1.0).is_none());
}

#[test]
fn a_newest_instrument_gets_aggregations_that_follow_the_strategies_and_retire_frees_them() {
    let key = |instrument, step| {
        AggKey::new(
            InstrumentId::new(instrument),
            BarSpec::new(step, AggregationUnit::Tick).unwrap(),
            0.0,
            0,
            BarStamp::Open,
        )
        .unwrap()
    };
    let (s0, s1) = (StrategyId::new(0), StrategyId::new(1));
    let mut b = AggregatorHubBuilder::new(1);
    assert_eq!(b.register(s0, key(0, 1)).unwrap(), (AggId::new(0), 0));
    assert_eq!(b.register(s1, key(0, 1)).unwrap(), (AggId::new(0), 0));
    let mut hub = b.build().unwrap();

    hub.add_instrument(None).unwrap();
    // s0 declared two subscriptions in all; its stream ids go on from its own list.
    assert_eq!(hub.subscribe(s0, key(1, 1), None).unwrap(), (AggId::new(1), 1));
    assert_eq!(hub.subscribe(s1, key(1, 1), None).unwrap(), (AggId::new(1), 1));
    assert_eq!(hub.subscribe(s0, key(1, 2), None).unwrap(), (AggId::new(2), 2));
    assert_eq!(hub.subscribers(AggId::new(1)), &[(s0, 1), (s1, 1)]);
    assert!(hub.subscribe(s0, key(0, 5), None).is_err(), "only the newest instrument grows");

    let (i0, i1) = (InstrumentId::new(0), InstrumentId::new(1));
    hub.on_bar(i1, &bar(0, 1.0));
    assert_eq!(hub.deliveries(s0).map(|(l, _)| l).collect::<Vec<_>>(), [1]);
    assert_eq!(hub.deliveries(s1).map(|(l, _)| l).collect::<Vec<_>>(), [1]);
    hub.on_bar(i0, &bar(1, 1.0));
    assert_eq!(hub.deliveries(s0).map(|(l, _)| l).collect::<Vec<_>>(), [0]);

    hub.retire_instrument(i1).unwrap();
    assert!(hub.on_bar(i1, &bar(2, 1.0)).is_empty());
    assert_eq!(hub.deliveries(s0).count(), 0);
    hub.on_bar(i0, &bar(3, 1.0));
    assert_eq!(hub.deliveries(s0).map(|(l, _)| l).collect::<Vec<_>>(), [0]);
}
