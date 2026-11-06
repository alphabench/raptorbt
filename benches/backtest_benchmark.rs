//! Benchmark for RaptorBT backtesting performance.

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use raptorbt::core::types::{BacktestConfig, CompiledSignals, Direction, OhlcvData};
use raptorbt::data::{
    builder_for_with, AggregationUnit, BarBuilder, BarSpec, BarStamp, BuilderParams, EventFeed,
    EventPayload, MarketEvent, SourceRecord, TradeTick,
};
use raptorbt::indicators::streaming::{StreamingIndicator, StreamingRsi, StreamingSma};
use raptorbt::indicators::trend::{ema, sma};
use raptorbt::portfolio::engine::PortfolioEngine;
use raptorbt::portfolio::KernelBar;
use raptorbt::runtime::{
    AggKey, AggregatorHub, AggregatorHubBuilder, BarColumns, HeapSize, IndInput, IndKey, IndSpec,
    IndicatorBank, IndicatorRegistry, InstrumentId, MarketStore, MergeCursor, Ohlc, Reveal, SlotId,
    StrategyId, SubscriptionIndex, SubscriptionIndexBuilder,
};

/// Generate sample OHLCV data.
fn generate_sample_data(n: usize) -> OhlcvData<'static> {
    let mut open = vec![100.0; n];
    let mut high = vec![101.0; n];
    let mut low = vec![99.0; n];
    let mut close = vec![100.0; n];

    // Create a trending pattern
    for i in 1..n {
        let change = (i as f64 * 0.1).sin() * 2.0;
        close[i] = close[i - 1] + change;
        open[i] = close[i - 1];
        high[i] = close[i].max(open[i]) + 1.0;
        low[i] = close[i].min(open[i]) - 1.0;
    }

    OhlcvData {
        timestamps: (0..n as i64).collect(),
        open: open.into(),
        high: high.into(),
        low: low.into(),
        close: close.into(),
        volume: vec![1000.0; n].into(),
    }
}

/// Generate sample trading signals based on SMA crossover.
fn generate_sample_signals(
    close: &[f64],
    fast_period: usize,
    slow_period: usize,
) -> CompiledSignals {
    let n = close.len();
    let fast_sma = sma(close, fast_period).unwrap_or_else(|_| vec![0.0; n]);
    let slow_sma = sma(close, slow_period).unwrap_or_else(|_| vec![0.0; n]);

    let mut entries = vec![false; n];
    let mut exits = vec![false; n];

    for i in 1..n {
        // Entry: fast crosses above slow
        if fast_sma[i] > slow_sma[i] && fast_sma[i - 1] <= slow_sma[i - 1] {
            entries[i] = true;
        }
        // Exit: fast crosses below slow
        if fast_sma[i] < slow_sma[i] && fast_sma[i - 1] >= slow_sma[i - 1] {
            exits[i] = true;
        }
    }

    CompiledSignals {
        symbol: "BENCH".to_string(),
        entries,
        exits,
        position_sizes: None,
        direction: Direction::Long,
        weight: 1.0,
    }
}

fn bench_single_backtest(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_backtest");

    for size in [1000, 5000, 10000, 50000].iter() {
        group.bench_with_input(BenchmarkId::new("bars", size), size, |b, &size| {
            let ohlcv = generate_sample_data(size);
            let signals = generate_sample_signals(&ohlcv.close, 10, 30);
            let config = BacktestConfig::default();
            let engine = PortfolioEngine::new(config);

            b.iter(|| {
                let result = engine.run_single(black_box(&ohlcv), black_box(&signals));
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_sma(c: &mut Criterion) {
    let mut group = c.benchmark_group("sma");

    for size in [1000, 5000, 10000, 50000].iter() {
        group.bench_with_input(BenchmarkId::new("data_size", size), size, |b, &size| {
            let ohlcv = generate_sample_data(size);

            b.iter(|| {
                let result = sma(black_box(&ohlcv.close), black_box(20));
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_ema(c: &mut Criterion) {
    let mut group = c.benchmark_group("ema");

    for size in [1000, 5000, 10000, 50000].iter() {
        group.bench_with_input(BenchmarkId::new("data_size", size), size, |b, &size| {
            let ohlcv = generate_sample_data(size);

            b.iter(|| {
                let result = ema(black_box(&ohlcv.close), black_box(20));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Build `streams` time-sorted trade streams of `per_stream` events each.
/// Timestamps are staggered per stream with a deterministic pseudo-random
/// gap so the heap sees realistic interleaving rather than lockstep ties.
fn build_feed(streams: usize, per_stream: usize) -> EventFeed {
    let mut feed = EventFeed::new();
    for s in 0..streams {
        let mut ts = s as i64;
        let mut state = (s as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let events: Vec<MarketEvent> = (0..per_stream)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ts += 1 + (state % 1000) as i64;
                MarketEvent {
                    instrument: s as u32,
                    stream: s as u32,
                    payload: EventPayload::Trade(TradeTick {
                        timestamp: ts,
                        price: 100.0,
                        size: 1.0,
                        signed_size: 0.0,
                        oi: 0.0,
                    }),
                }
            })
            .collect();
        feed.add_stream(events);
    }
    feed
}

/// k-way merge in `src/data/feed.rs` (`EventFeed` is public). The feed is
/// consumed by iteration, so each iteration rebuilds it; `build_only` is the
/// construction baseline to subtract from `build_and_merge` (priming the heap
/// plus draining 250 x 2000 events).
fn bench_event_feed_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_feed_merge");
    group.sample_size(20);
    let per_stream = 2000;
    for streams in [10usize, 250] {
        group.throughput(Throughput::Elements((streams * per_stream) as u64));
        group.bench_with_input(BenchmarkId::new("build_only", streams), &streams, |b, &n| {
            b.iter(|| black_box(build_feed(n, per_stream)));
        });
        group.bench_with_input(BenchmarkId::new("build_and_merge", streams), &streams, |b, &n| {
            b.iter(|| {
                let feed = build_feed(n, per_stream);
                let mut acc = 0i64;
                for e in feed {
                    acc = acc.wrapping_add(e.timestamp());
                }
                black_box(acc)
            });
        });
    }
    group.finish();
}

/// The same timestamps as `build_feed`, as bar columns in a store: `streams`
/// instruments of `per_stream` bars each.
fn build_store(streams: usize, per_stream: usize) -> MarketStore {
    let mut store = MarketStore::new();
    for s in 0..streams {
        let mut ts = s as i64;
        let mut state = (s as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let stamps: Vec<i64> = (0..per_stream)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ts += 1 + (state % 1000) as i64;
                ts
            })
            .collect();
        let px = vec![100.0; per_stream];
        let vol = vec![1.0; per_stream];
        let label = format!("S{s}");
        let id = store.add_instrument(&label).unwrap();
        let bars = BarColumns::from_slices(&label, &stamps, &px, &px, &px, &px, &vol).unwrap();
        store.set_bars(id, bars).unwrap();
    }
    store
}

/// Lazy merge in `src/runtime/cursor.rs` over the same shape of data as
/// `event_feed_merge`. The store is built once and shared; `merge` creates a
/// cursor (priming the heap) and drains 250 x 2000 events, so compare it with
/// `event_feed_merge/build_and_merge`, which also pays to materialise events.
fn bench_runtime_cursor(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime_cursor");
    group.sample_size(20);
    let per_stream = 2000;
    for streams in [10usize, 250] {
        let store = build_store(streams, per_stream);
        group.throughput(Throughput::Elements((streams * per_stream) as u64));
        group.bench_with_input(BenchmarkId::new("merge", streams), &streams, |b, _| {
            b.iter(|| {
                let mut acc = 0i64;
                for e in MergeCursor::new(&store) {
                    acc = acc.wrapping_add(e.ts);
                }
                black_box(acc)
            });
        });
    }
    group.finish();
}

/// 250 instruments of random-walk bars in a store, for the indicator bench.
fn build_walk_store(streams: usize, per_stream: usize) -> MarketStore {
    let mut store = MarketStore::new();
    for s in 0..streams {
        let mut state = (s as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut ts = s as i64;
        let mut px = 100.0f64;
        let (mut stamps, mut o, mut h, mut l, mut c) = (vec![], vec![], vec![], vec![], vec![]);
        for _ in 0..per_stream {
            ts += 1 + (next() % 1000) as i64;
            let open = px;
            px *= 1.0 + ((next() % 2001) as f64 - 1000.0) / 100_000.0;
            stamps.push(ts);
            o.push(open);
            h.push(open.max(px) * 1.001);
            l.push(open.min(px) * 0.999);
            c.push(px);
        }
        let label = format!("S{s}");
        let id = store.add_instrument(&label).unwrap();
        let vol = vec![1.0; per_stream];
        store
            .set_bars(id, BarColumns::from_slices(&label, &stamps, &o, &h, &l, &c, &vol).unwrap())
            .unwrap();
    }
    store
}

/// Shared indicator cells against a private core per strategy
/// (`src/runtime/indicators.rs`). 2,000 strategies each run SMA(10), SMA(30)
/// and RSI(14) on 5 of 250 instruments, so each instrument has about 40
/// readers. Both sides walk the same merged bar stream and read all three
/// values of every subscribed strategy after each bar.
fn bench_runtime_indicator_bank(c: &mut Criterion) {
    const INSTRUMENTS: usize = 250;
    const BARS: usize = 2000;
    const STRATEGIES: usize = 2000;
    const PER_STRATEGY: usize = 5;

    let store = build_walk_store(INSTRUMENTS, BARS);

    // Each strategy's five distinct instruments, from a fixed generator.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut picks: Vec<[usize; PER_STRATEGY]> = Vec::with_capacity(STRATEGIES);
    for _ in 0..STRATEGIES {
        let mut set = [usize::MAX; PER_STRATEGY];
        let mut n = 0;
        while n < PER_STRATEGY {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let i = (state % INSTRUMENTS as u64) as usize;
            if !set[..n].contains(&i) {
                set[n] = i;
                n += 1;
            }
        }
        picks.push(set);
    }

    // Shared side: registry -> bank, and each instrument's readers.
    let mut registry = IndicatorRegistry::new();
    let mut registered: Vec<(StrategyId, InstrumentId, [SlotId; 3])> = Vec::new();
    for (s, set) in picks.iter().enumerate() {
        let strategy = StrategyId::new(s as u32);
        for &i in set {
            let input = IndInput::Primary(InstrumentId::new(i as u32));
            let mut slot =
                |spec| registry.register(strategy, IndKey::new(input, spec), None).unwrap().slot;
            let slots = [slot(IndSpec::Sma(10)), slot(IndSpec::Sma(30)), slot(IndSpec::Rsi(14))];
            registered.push((strategy, InstrumentId::new(i as u32), slots));
        }
    }
    // Handles are resolved once; every bank built from this registry shares them.
    let handles = IndicatorBank::new(&registry);
    let mut readers: Vec<Vec<(Reveal, [SlotId; 3])>> = vec![Vec::new(); INSTRUMENTS];
    for (strategy, inst, slots) in registered {
        let handle = handles.primary_handle(strategy, inst).unwrap();
        readers[inst.index()].push((handle, slots));
    }
    // Private side: three cores per (strategy, instrument), and who owns them.
    let mut owners: Vec<Vec<usize>> = vec![Vec::new(); INSTRUMENTS];
    for (s, set) in picks.iter().enumerate() {
        for (k, &i) in set.iter().enumerate() {
            owners[i].push(s * PER_STRATEGY + k);
        }
    }
    let new_cores = || -> Vec<(StreamingSma, StreamingSma, StreamingRsi)> {
        (0..STRATEGIES * PER_STRATEGY)
            .map(|_| (StreamingSma::new(10), StreamingSma::new(30), StreamingRsi::new(14)))
            .collect()
    };

    let run_shared = |bank: &IndicatorBank| -> f64 {
        let mut acc = 0.0;
        for e in MergeCursor::new(&store) {
            let k = store.bar(e.instrument, e.row as u32).unwrap();
            bank.begin_event();
            bank.emit_primary(
                e.instrument,
                Ohlc {
                    open: k.open,
                    high: k.high,
                    low: k.low,
                    close: k.close,
                    volume: k.volume,
                    ts: k.timestamp,
                },
            )
            .unwrap();
            for (handle, slots) in &readers[e.instrument.index()] {
                bank.reveal(*handle, 1);
                for slot in slots {
                    acc += bank.read(*slot).values[0];
                }
            }
        }
        acc
    };
    let run_private = |cores: &mut Vec<(StreamingSma, StreamingSma, StreamingRsi)>| -> (f64, u64) {
        let (mut acc, mut updates) = (0.0, 0u64);
        for e in MergeCursor::new(&store) {
            let close = store.bar(e.instrument, e.row as u32).unwrap().close;
            for &o in &owners[e.instrument.index()] {
                let (fast, slow, rsi) = &mut cores[o];
                acc += fast.update(close).unwrap_or(0.0);
                acc += slow.update(close).unwrap_or(0.0);
                acc += rsi.update(close).unwrap_or(0.0);
                updates += 3;
            }
        }
        (acc, updates)
    };

    // Report the work and memory once; criterion measures the time.
    let bank = IndicatorBank::new(&registry);
    let shared_acc = run_shared(&bank);
    let (private_acc, private_updates) = run_private(&mut new_cores());
    assert_eq!(shared_acc.to_bits(), private_acc.to_bits(), "bank and private cores disagree");
    // Private cores: the structs plus the windows they allocate (SMA 10 and
    // 30 keep period + 1 values, RSI 14 two seed buffers of 14).
    let private_heap = STRATEGIES
        * PER_STRATEGY
        * (std::mem::size_of::<(StreamingSma, StreamingSma, StreamingRsi)>() + (11 + 31 + 28) * 8);
    eprintln!(
        "runtime_indicator_bank: {} registrations -> {} cells; core updates shared {} vs private {}; heap shared {} B vs private {} B",
        bank.registrations(),
        bank.cells(),
        bank.updates(),
        private_updates,
        bank.heap_bytes(),
        private_heap,
    );

    let mut group = c.benchmark_group("runtime_indicator_bank");
    group.sample_size(10);
    group.throughput(Throughput::Elements((INSTRUMENTS * BARS) as u64));
    // The part both sides share: the merge and the bar lookup.
    group.bench_function("walk_only", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for e in MergeCursor::new(&store) {
                acc += store.bar(e.instrument, e.row as u32).unwrap().close;
            }
            black_box(acc)
        });
    });
    group.bench_function("shared", |b| {
        b.iter_batched(
            || IndicatorBank::new(&registry),
            |bank| black_box(run_shared(&bank)),
            BatchSize::LargeInput,
        );
    });
    group.bench_function("private", |b| {
        b.iter_batched(
            new_cores,
            |mut cores| black_box(run_private(&mut cores)),
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

/// Shared aggregators in `src/runtime/aggregators.rs` against one private
/// builder per (strategy, instrument, timeframe): 250 instruments x 2,000
/// 1-minute bars, 2,000 strategies each on 5 instruments x 5m/15m/1h.
///
/// Both sides walk the same events and read every delivered bar. `build`
/// (registration) is excluded from the timed runs. The push-call totals are
/// printed once.
fn bench_runtime_agg_hub(c: &mut Criterion) {
    const INSTRUMENTS: usize = 250;
    const BARS: usize = 2000;
    const STRATEGIES: usize = 2000;
    const PER_STRATEGY: usize = 5;
    const MINUTES: [u32; 3] = [5, 15, 60];

    let bars: Vec<KernelBar> = (0..BARS)
        .map(|i| {
            let px = 100.0 + (i % 17) as f64;
            KernelBar {
                timestamp: 1_700_000_000_000_000_000 + i as i64 * 60_000_000_000,
                open: px,
                high: px + 1.0,
                low: px - 1.0,
                close: px + 0.5,
                volume: 10.0,
            }
        })
        .collect();

    // Deterministic draw of 5 distinct instruments per strategy.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let picks: Vec<Vec<u32>> = (0..STRATEGIES)
        .map(|_| {
            let mut chosen = Vec::with_capacity(PER_STRATEGY);
            while chosen.len() < PER_STRATEGY {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let i = (state % INSTRUMENTS as u64) as u32;
                if !chosen.contains(&i) {
                    chosen.push(i);
                }
            }
            chosen
        })
        .collect();
    let spec = |m: u32| {
        let (step, unit) =
            if m == 60 { (1, AggregationUnit::Hour) } else { (m, AggregationUnit::Minute) };
        BarSpec::new(step, unit).unwrap()
    };

    let index: SubscriptionIndex = {
        let mut b = SubscriptionIndexBuilder::new(INSTRUMENTS);
        for (s, list) in picks.iter().enumerate() {
            let ids: Vec<InstrumentId> = list.iter().map(|&i| InstrumentId::new(i)).collect();
            b.register(StrategyId::new(s as u32), &ids).unwrap();
        }
        b.build().unwrap()
    };
    let build_hub = || -> AggregatorHub {
        let mut b = AggregatorHubBuilder::new(INSTRUMENTS);
        for (s, list) in picks.iter().enumerate() {
            for &i in list {
                for m in MINUTES {
                    let key =
                        AggKey::new(InstrumentId::new(i), spec(m), 0.0, 0, BarStamp::Open).unwrap();
                    b.register(StrategyId::new(s as u32), key).unwrap();
                }
            }
        }
        b.build().unwrap()
    };
    type Private = Vec<Vec<(u32, [Box<dyn BarBuilder + Send + Sync>; 3])>>;
    let build_private = || -> Private {
        picks
            .iter()
            .map(|list| {
                list.iter()
                    .map(|&i| {
                        let mk =
                            |m| builder_for_with(spec(m), 0, BuilderParams::default()).unwrap();
                        (i, [mk(5), mk(15), mk(60)])
                    })
                    .collect()
            })
            .collect()
    };

    let run_shared = |hub: &mut AggregatorHub| -> u64 {
        let mut seen = 0u64;
        for bar in &bars {
            for inst in 0..INSTRUMENTS as u32 {
                let id = InstrumentId::new(inst);
                hub.on_bar(id, bar);
                for &s in index.subscribers(id) {
                    for (_, b) in hub.deliveries(s) {
                        seen = seen.wrapping_add(b.timestamp as u64);
                    }
                }
            }
        }
        seen
    };
    let run_private = |private: &mut Private, pushes: &mut u64| -> u64 {
        let mut seen = 0u64;
        for bar in &bars {
            let rec = SourceRecord::from_bar(&raptorbt::core::types::OhlcvBar {
                timestamp: bar.timestamp,
                open: bar.open,
                high: bar.high,
                low: bar.low,
                close: bar.close,
                volume: bar.volume,
            });
            for inst in 0..INSTRUMENTS as u32 {
                for &s in index.subscribers(InstrumentId::new(inst)) {
                    let (_, builders) = private[s.index()]
                        .iter_mut()
                        .find(|(i, _)| *i == inst)
                        .expect("strategy holds the instrument");
                    for builder in builders.iter_mut() {
                        *pushes += 1;
                        if let Some(b) = builder.push(&rec) {
                            seen = seen.wrapping_add(b.timestamp as u64);
                        }
                    }
                }
            }
        }
        seen
    };

    {
        let mut hub = build_hub();
        let shared_seen = run_shared(&mut hub);
        let mut private = build_private();
        let mut private_pushes = 0u64;
        let private_seen = run_private(&mut private, &mut private_pushes);
        assert_eq!(shared_seen, private_seen, "shared and private runs must deliver alike");
        let subs = hub.subscriptions();
        eprintln!(
            "runtime_agg_hub: {} subscriptions -> {} aggregators; push calls shared {} vs private {} ({:.1}x); index heap {} B ({:.1} B/subscription)",
            subs,
            hub.aggregators(),
            hub.push_calls(),
            private_pushes,
            private_pushes as f64 / hub.push_calls() as f64,
            hub.heap_bytes() + index.heap_bytes(),
            (hub.heap_bytes() + index.heap_bytes()) as f64 / subs as f64,
        );
    }

    let mut group = c.benchmark_group("runtime_agg_hub");
    group.sample_size(10);
    group.throughput(Throughput::Elements((INSTRUMENTS * BARS) as u64));
    group.bench_function("shared", |b| {
        b.iter_batched(&build_hub, |mut hub| black_box(run_shared(&mut hub)), BatchSize::LargeInput)
    });
    group.bench_function("private", |b| {
        b.iter_batched(
            &build_private,
            |mut private| {
                let mut pushes = 0;
                black_box(run_private(&mut private, &mut pushes))
            },
            BatchSize::LargeInput,
        )
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_single_backtest,
    bench_sma,
    bench_ema,
    bench_event_feed_merge,
    bench_runtime_cursor,
    bench_runtime_agg_hub,
    bench_runtime_indicator_bank
);
criterion_main!(benches);
