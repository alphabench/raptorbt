//! `RuntimeCore` against standalone sessions.

use super::*;
use crate::data::{AggregationUnit, BarSpec, BarStamp};
use crate::indicators::streaming::{StreamingIndicator, StreamingSma};
use crate::portfolio::session::{EventSession, ScheduleData};
use crate::runtime::indicators::{IndInput, IndSpec};
use crate::runtime::store::BarColumns;
use crate::runtime::LiveFeed;

const INSTRUMENTS: usize = 5;

/// A small deterministic generator.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn unit(&mut self) -> f64 {
        (self.next() % 10_000) as f64 / 10_000.0
    }
}

/// Instrument `i` has `lens[i]` bars on a shared 10 ns grid starting at a
/// per-instrument phase, so many events tie on timestamp across instruments.
fn make_store(seed: u64, lens: [usize; INSTRUMENTS]) -> MarketStore {
    let mut rng = Lcg(seed);
    let mut store = MarketStore::new();
    for (i, &n) in lens.iter().enumerate() {
        let id = store.add_instrument(&format!("S{i}")).unwrap();
        let (mut ts, mut o, mut h, mut l, mut c, mut v) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        let mut price = 100.0 + 10.0 * i as f64;
        let mut t = (i as i64 % 2) * 10;
        for _ in 0..n {
            let open = price;
            price = (price * (1.0 + (rng.unit() - 0.5) * 0.08)).max(1.0);
            ts.push(t);
            o.push(open);
            h.push(open.max(price) + rng.unit());
            l.push((open.min(price) - rng.unit()).max(0.5));
            c.push(price);
            v.push(100.0 + rng.unit() * 900.0);
            // Sometimes two bars share a timestamp.
            t += if rng.next().is_multiple_of(9) { 0 } else { 10 * (1 + (rng.next() % 2) as i64) };
        }
        store.set_bars(id, BarColumns::from_vecs("t", ts, o, h, l, c, v).unwrap()).unwrap();
    }
    store
}

fn config() -> BacktestConfig {
    BacktestConfig { fees: 0.001, max_drawdown_pct: Some(12.0), ..BacktestConfig::default() }
}

fn params(atr: usize) -> Vec<InstrumentParams> {
    (0..INSTRUMENTS)
        .map(|i| InstrumentParams {
            direction: if i % 3 == 2 { Direction::Short } else { Direction::Long },
            atr_period: atr,
            ..InstrumentParams::default()
        })
        .collect()
}

/// What a strategy does on an event: a pure function of the strategy and the
/// event as the strategy sees it (local instrument, ordinal).
fn intent(_sid: u32, local: usize, local_idx: usize) -> StepInput {
    let h = (local as u64 * 104_729 + local_idx as u64 * 1_299_709)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        >> 40;
    StepInput {
        entry: h.is_multiple_of(5),
        exit: h % 7 == 1,
        size_mult: h.is_multiple_of(3).then_some(0.5),
        ..StepInput::default()
    }
}

fn ids(list: &[u32]) -> Vec<InstrumentId> {
    list.iter().map(|&i| InstrumentId::new(i)).collect()
}

type Trace = Vec<(usize, usize)>;

struct Outcome {
    trades: String,
    orders: usize,
    metrics: String,
    equity: Vec<u64>,
    returns: Vec<u64>,
    drawdown: Vec<u64>,
    halted: bool,
    halted_at: Option<usize>,
    per_instrument: String,
    trace: Trace,
}

fn digest(out: SessionOutcome, trace: Trace) -> Outcome {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    Outcome {
        trades: format!("{:?}", out.result.trades),
        orders: out.result.orders.len(),
        metrics: format!("{:?}", out.result.metrics),
        equity: bits(&out.result.equity_curve),
        returns: bits(&out.result.returns),
        drawdown: bits(&out.result.drawdown_curve),
        halted: out.halted,
        halted_at: out.halted_at,
        per_instrument: format!("{:?}", out.instruments),
        trace,
    }
}

fn assert_same(got: &Outcome, want: &Outcome, what: &str) {
    assert_eq!(got.trace, want.trace, "{what}: events seen");
    assert_eq!(got.trades, want.trades, "{what}: trades");
    assert_eq!(got.orders, want.orders, "{what}: orders");
    assert_eq!(got.metrics, want.metrics, "{what}: metrics");
    assert_eq!(got.equity, want.equity, "{what}: equity");
    assert_eq!(got.returns, want.returns, "{what}: returns");
    assert_eq!(got.drawdown, want.drawdown, "{what}: drawdown");
    assert_eq!(got.halted, want.halted, "{what}: halted");
    assert_eq!(got.halted_at, want.halted_at, "{what}: halted_at");
    assert_eq!(got.per_instrument, want.per_instrument, "{what}: per instrument");
}

fn build(
    store: MarketStore,
    mode: AccountMode,
    atr: usize,
    subsets: &[Vec<u32>],
) -> (RuntimeCore, Vec<StrategyId>) {
    let mut b =
        RuntimeBuilder::new(store, params(atr), config(), mode, PositionPolicy::Net).unwrap();
    let sids = subsets.iter().map(|s| b.add_strategy(&ids(s)).unwrap()).collect();
    (b.build().unwrap(), sids)
}

/// Drive the core over the whole tape; each strategy applies `intent`.
fn run_core(core: &mut RuntimeCore, sids: &[StrategyId]) -> Vec<Outcome> {
    let store = core.store().clone();
    let mut cursor = MergeCursor::new(&store);
    let mut traces: Vec<Trace> = vec![Vec::new(); sids.len()];
    while let Some(e) = core.advance(&mut cursor).unwrap() {
        let subs: Vec<StrategyId> = core.subscribers(e.instrument).to_vec();
        for sid in subs {
            let local = core.local_of(sid, e.instrument).unwrap();
            let mut input = intent(sid.raw(), local, e.local_idx as usize);
            input.atr = core.atr_at(sid, local, e.row as usize);
            traces[sid.index()].push((local, e.local_idx as usize));
            core.apply(sid, input).unwrap();
        }
    }
    sids.iter().zip(traces).map(|(&sid, t)| digest(core.finish(sid).unwrap(), t)).collect()
}

/// The legacy shape: a session over only the strategy's symbols.
fn run_standalone(
    store: &MarketStore,
    mode: AccountMode,
    atr: usize,
    sid: u32,
    subset: &[u32],
) -> Outcome {
    let mut session = EventSession::with_account(config(), mode);
    let p = params(atr);
    for &g in subset {
        let id = InstrumentId::new(g);
        let i = session.add_instrument(
            store.symbols().symbol(id).unwrap().to_string(),
            p[g as usize].direction,
            None,
            None,
            PositionPolicy::Net,
        );
        let b = store.bars(id).unwrap();
        let bars = (0..b.len() as u32).map(|r| b.bar(r).unwrap()).collect();
        session.set_bars(i, bars).unwrap();
    }
    session.seal();
    let atr_series: Vec<Option<Vec<f64>>> = subset
        .iter()
        .map(|&g| {
            let b = store.bars(InstrumentId::new(g)).unwrap();
            (atr > 0)
                .then(|| crate::indicators::volatility::atr(&b.high, &b.low, &b.close, atr))
                .and_then(Result::ok)
        })
        .collect();
    let mut trace = Vec::new();
    while let Some(entry) = session.current() {
        let mut input = intent(sid, entry.instrument, entry.local_idx);
        input.atr = atr_series[entry.instrument].as_ref().map_or(0.0, |s| s[entry.local_idx]);
        trace.push((entry.instrument, entry.local_idx));
        session.apply_current(input);
    }
    digest(session.finish(), trace)
}

fn subsets() -> Vec<Vec<u32>> {
    vec![
        vec![0, 1, 2, 3, 4],
        vec![1, 3],
        vec![4],
        vec![0, 2, 4],
        vec![2, 3],
        vec![0, 1],
        vec![1, 2, 3, 4],
    ]
}

#[test]
fn books_over_one_store_match_standalone_sessions() {
    for (seed, mode) in [
        (1u64, AccountMode::Cash),
        (2, AccountMode::Cash),
        (3, AccountMode::Margin { leverage: 3.0 }),
        (4, AccountMode::Margin { leverage: 5.0 }),
    ] {
        for atr in [0usize, 5] {
            let lens = [180, 150, 0, 210, 99];
            let subsets = subsets();
            let (mut core, sids) = build(make_store(seed, lens), mode, atr, &subsets);
            let got = run_core(&mut core, &sids);
            let store = make_store(seed, lens);
            let mut any_trades = false;
            for (i, subset) in subsets.iter().enumerate() {
                let want = run_standalone(&store, mode, atr, i as u32, subset);
                assert_same(&got[i], &want, &format!("seed {seed} atr {atr} strategy {i}"));
                any_trades |= got[i].trades != "[]";
            }
            assert!(any_trades, "the scenario must trade");
        }
    }
}

#[test]
fn a_subset_sees_its_standalone_ordinals_not_the_universes() {
    // Strategy 1 holds only instrument 3; its ordinals must be 3's own bar
    // rows whatever the other instruments and strategies do around it.
    let lens = [120, 130, 140, 150, 160];
    let (mut core, sids) = build(make_store(9, lens), AccountMode::Cash, 0, &subsets());
    let got = run_core(&mut core, &sids);
    let only_three = &got[1]; // subsets()[1] = [1, 3]
    let locals_three: Vec<usize> =
        only_three.trace.iter().filter(|(l, _)| *l == 1).map(|(_, i)| *i).collect();
    assert_eq!(locals_three, (0..150).collect::<Vec<_>>());
    let alone = &got[2]; // [4]
    assert_eq!(alone.trace, (0..160).map(|i| (0, i)).collect::<Trace>());
}

#[test]
fn registration_order_does_not_change_any_strategy_result() {
    let lens = [100, 90, 80, 110, 70];
    let subs = subsets();
    let (mut core, sids) = build(make_store(5, lens), AccountMode::Cash, 3, &subs);
    let forward = run_core(&mut core, &sids);

    let reversed: Vec<Vec<u32>> = subs.iter().rev().cloned().collect();
    let (mut core, sids) = build(make_store(5, lens), AccountMode::Cash, 3, &reversed);
    let backward = run_core(&mut core, &sids);
    for (i, f) in forward.iter().enumerate() {
        assert_same(&backward[subs.len() - 1 - i], f, &format!("strategy {i}"));
    }
}

#[test]
fn an_unsubscribed_strategy_is_never_stepped_and_has_no_trace() {
    let (mut core, sids) =
        build(make_store(7, [30, 30, 30, 30, 30]), AccountMode::Cash, 0, &[vec![0], vec![4]]);
    let out = run_core(&mut core, &sids);
    assert_eq!(out[0].trace.len(), 30);
    assert_eq!(out[1].trace.len(), 30);
}

#[test]
fn registration_is_validated() {
    let mut b = RuntimeBuilder::new(
        make_store(1, [3; INSTRUMENTS]),
        params(0),
        config(),
        AccountMode::Cash,
        PositionPolicy::Net,
    )
    .unwrap();
    assert!(b.add_strategy(&[]).is_err());
    assert!(b.add_strategy(&ids(&[2, 1])).is_err());
    assert!(b.add_strategy(&ids(&[1, 1])).is_err());
    assert!(b.add_strategy(&ids(&[9])).is_err());
    let sid = b.add_strategy(&ids(&[1, 3])).unwrap();
    assert_eq!(b.local_of(sid, InstrumentId::new(3)), Some(1));
    assert_eq!(b.local_of(sid, InstrumentId::new(2)), None);
    assert_eq!(b.instrument_of(sid, 0), Some(InstrumentId::new(1)));
    let other = AggKey::new(
        InstrumentId::new(2),
        BarSpec::new(2, AggregationUnit::Tick).unwrap(),
        0.0,
        0,
        BarStamp::Close,
    )
    .unwrap();
    assert!(b.register_aggregation(sid, other).is_err());
}

#[test]
fn watermark_follows_the_cursor() {
    let (mut core, _) = build(make_store(11, [4, 4, 4, 4, 4]), AccountMode::Cash, 0, &[vec![0, 1]]);
    let store = core.store().clone();
    let mut cursor = MergeCursor::new(&store);
    let mut seen = [0u32; INSTRUMENTS];
    while let Some(e) = core.advance(&mut cursor).unwrap() {
        seen[e.instrument.index()] += 1;
        for (i, &count) in seen.iter().enumerate() {
            let id = InstrumentId::new(i as u32);
            assert_eq!(core.watermark(id), u64::from(count));
            assert_eq!(cursor.visible_row(id).map(|r| r + 1), (count > 0).then_some(count));
        }
    }
}

#[test]
fn the_slot_is_small() {
    assert!(std::mem::size_of::<StrategySlot>() <= 64, "{}", std::mem::size_of::<StrategySlot>());
    assert_eq!(std::mem::size_of::<StrategySlot>(), 16);
}

#[test]
fn shared_indicators_and_aggregations_reach_every_subscriber_as_private_ones_would() {
    // Strategies 0 and 2 read the same SMA on instrument 1 and the same
    // 3-bar aggregation of it; strategy 1 does not trade instrument 1.
    let store = make_store(21, [60, 60, 60, 60, 60]);
    let mut b =
        RuntimeBuilder::new(store, params(0), config(), AccountMode::Cash, PositionPolicy::Net)
            .unwrap();
    let s0 = b.add_strategy(&ids(&[0, 1])).unwrap();
    let s1 = b.add_strategy(&ids(&[0, 2])).unwrap();
    let s2 = b.add_strategy(&ids(&[1, 3])).unwrap();
    let inst1 = InstrumentId::new(1);
    let agg_key = AggKey::new(
        inst1,
        BarSpec::new(3, AggregationUnit::Tick).unwrap(),
        0.0,
        0,
        BarStamp::Close,
    )
    .unwrap();
    let mut slots = Vec::new();
    let mut local_stream = Vec::new();
    for &s in &[s0, s2] {
        let (agg, local) = b.register_aggregation(s, agg_key).unwrap();
        local_stream.push(local);
        let prim = b
            .register_indicator(s, IndKey::new(IndInput::Primary(inst1), IndSpec::Sma(4)), None)
            .unwrap();
        let comp = b
            .register_indicator(
                s,
                IndKey::new(IndInput::Composite(agg), IndSpec::Sma(2)),
                Some(LocalStreamId(u32::from(local))),
            )
            .unwrap();
        slots.push((s, prim.slot, comp.slot));
    }
    let mut core = b.build().unwrap();
    assert_eq!(core.bank().cells(), 2, "two readers share each cell");
    assert_eq!(core.hub().aggregators(), 1);

    // Private references, one pair per strategy, fed in the strategy's order.
    let mut private: Vec<(StreamingSma, StreamingSma)> =
        slots.iter().map(|_| (StreamingSma::new(4), StreamingSma::new(2))).collect();
    let store = core.store().clone();
    let mut cursor = MergeCursor::new(&store);
    let mut checked = 0;
    // Bars each private reference has been fed: `update_count`.
    let mut fed = vec![(0u64, 0u64); slots.len()];
    while let Some(e) = core.advance(&mut cursor).unwrap() {
        let subs: Vec<StrategyId> = core.subscribers(e.instrument).to_vec();
        for sid in subs {
            let Some(pos) = slots.iter().position(|s| s.0 == sid) else { continue };
            let (_, prim_slot, comp_slot) = slots[pos];
            let emitted: Vec<_> = core.deliveries(sid).map(|(l, bar)| (l, *bar)).collect();
            let mut k = 0;
            for (local, bar) in &emitted {
                k += 1;
                let want = private[pos].1.update(bar.close);
                fed[pos].1 += 1;
                core.bank().reveal_stream(sid, LocalStreamId(u32::from(*local)), k);
                assert_eq!(core.bank().updates_seen(comp_slot), fed[pos].1);
                let got = core.bank().read(comp_slot);
                assert_eq!(got.initialized, want.is_some());
                assert_eq!(got.values[0].to_bits(), want.map_or(0, f64::to_bits));
                checked += 1;
            }
            if e.instrument == inst1 {
                let bar = core.current().unwrap().bar;
                let want = private[pos].0.update(bar.close);
                fed[pos].0 += 1;
                // Not yet revealed to this strategy: the count before the bar.
                assert_eq!(core.bank().updates_seen(prim_slot), fed[pos].0 - 1);
                core.bank().reveal_primary(sid, inst1);
                assert_eq!(core.bank().updates_seen(prim_slot), fed[pos].0);
                let got = core.bank().read(prim_slot);
                assert_eq!(got.initialized, want.is_some());
                assert_eq!(got.values[0].to_bits(), want.map_or(0, f64::to_bits));
                checked += 1;
            }
            core.apply(sid, StepInput::default()).unwrap();
        }
        let _ = s1;
    }
    assert!(checked > 100);
    // One update per cell per emission, however many readers.
    assert_eq!(core.bank().updates(), 60 + 20);
    assert!(core.hub().push_calls() <= 60 * 2);
}

#[test]
fn memory_report_categories_add_up_and_scale_per_strategy() {
    let (core, _) = build(make_store(3, [50; INSTRUMENTS]), AccountMode::Cash, 5, &subsets());
    let r = core.memory_report();
    assert_eq!(r.strategies, subsets().len());
    assert_eq!(r.shared_store, core.store().heap_bytes());
    assert!(r.engine_per_strategy > 0 && r.portfolio_per_strategy > 0);
    assert_eq!(
        r.total(),
        r.shared_store
            + r.aggregators
            + r.indicators
            + r.index
            + r.atr
            + r.engine_per_strategy
            + r.portfolio_per_strategy
            + r.curves
    );
    // ATR is held once per instrument, not once per subscriber.
    let series = INSTRUMENTS * (16 + 50 * 8);
    assert!(r.atr >= series && r.atr < 2 * series);
}

// -- ticks, quotes and depth ----------------------------------------------------

use crate::core::types::TickData;
use crate::data::book::{BookLevel, DepthTick};
use crate::portfolio::kernel::KernelBar as Bar;
use crate::runtime::store::{DepthColumns, TickColumns};

/// Instruments 0, 1 and 4 carry ticks (4 also bars), 2 carries bars only and 3
/// is empty. Instrument 1 has depth.
struct World {
    bars: Vec<Vec<Bar>>,
    ticks: Vec<Option<TickData>>,
    depth: Vec<Vec<DepthTick>>,
}

fn make_world(seed: u64) -> World {
    let mut rng = Lcg(seed);
    let mut world = World {
        bars: vec![Vec::new(); INSTRUMENTS],
        ticks: (0..INSTRUMENTS).map(|_| None).collect(),
        depth: vec![Vec::new(); INSTRUMENTS],
    };
    for i in [0usize, 1, 4] {
        let n = 260;
        let mut t = TickData {
            timestamps: Vec::new(),
            ltp: Vec::new(),
            bid: Vec::new(),
            ask: Vec::new(),
            buy_qty_delta: Vec::new(),
            sell_qty_delta: Vec::new(),
            oi: Vec::new(),
            ltq: Vec::new(),
            bid_qty: Vec::new(),
            ask_qty: Vec::new(),
        };
        let mut price = 100.0 + 7.0 * i as f64;
        let mut ts = (i as i64 % 2) * 5;
        for _ in 0..n {
            price = (price * (1.0 + (rng.unit() - 0.5) * 0.02)).max(1.0);
            let print = !rng.next().is_multiple_of(7);
            let two_sided = rng.next() % 10 < 7;
            t.timestamps.push(ts);
            t.ltp.push(if print { price } else { 0.0 });
            t.bid.push(if two_sided { price - 0.05 } else { 0.0 });
            t.ask.push(if two_sided { price + 0.05 } else { 0.0 });
            t.buy_qty_delta.push((rng.unit() * 20.0).round());
            t.sell_qty_delta.push((rng.unit() * 20.0).round());
            t.oi.push(0.0);
            t.ltq.push(if rng.next().is_multiple_of(2) {
                (rng.unit() * 30.0).round()
            } else {
                0.0
            });
            t.bid_qty.push(if two_sided { (rng.unit() * 50.0).round() } else { 0.0 });
            t.ask_qty.push(if two_sided { (rng.unit() * 50.0).round() } else { 0.0 });
            if i == 1 && rng.next().is_multiple_of(4) {
                let mut level = |side: f64, k: usize| BookLevel {
                    price: price + side * (0.05 + 0.05 * k as f64),
                    size: 10.0 + (rng.unit() * 40.0).round(),
                };
                let bids: Vec<_> = (0..5).map(|k| level(-1.0, k)).collect();
                let asks: Vec<_> = (0..5).map(|k| level(1.0, k)).collect();
                world.depth[i].push(DepthTick::from_levels(ts, &bids, &asks));
            }
            // Ties on timestamp across rows and across instruments.
            ts += if rng.next().is_multiple_of(5) { 0 } else { 5 };
        }
        if i == 4 {
            world.bars[i] = (0..40)
                .map(|k| {
                    let p = 100.0 + k as f64 * 0.1;
                    Bar {
                        timestamp: k * 30,
                        open: p,
                        high: p + 1.0,
                        low: p - 1.0,
                        close: p,
                        volume: 9.0,
                    }
                })
                .collect();
        }
        world.ticks[i] = Some(t);
    }
    world.bars[2] = (0..60)
        .map(|k| {
            let p = 90.0 + (k % 9) as f64;
            Bar { timestamp: k * 20, open: p, high: p + 1.0, low: p - 1.0, close: p, volume: 5.0 }
        })
        .collect();
    world
}

fn world_store(w: &World) -> MarketStore {
    let mut store = MarketStore::new();
    for i in 0..INSTRUMENTS {
        let id = store.add_instrument(&format!("S{i}")).unwrap();
        if !w.bars[i].is_empty() {
            store.set_bars(id, BarColumns::from_kernel_bars("b", &w.bars[i]).unwrap()).unwrap();
        }
        if let Some(t) = &w.ticks[i] {
            store.set_ticks(id, TickColumns::from_tick_data("t", t).unwrap()).unwrap();
        }
        if !w.depth[i].is_empty() {
            store.set_depth(id, DepthColumns::from_slice("d", &w.depth[i]).unwrap()).unwrap();
        }
    }
    store
}

fn run_standalone_world(w: &World, mode: AccountMode, sid: u32, subset: &[u32]) -> Outcome {
    let mut session = EventSession::with_account(config(), mode);
    let p = params(0);
    for &g in subset {
        let i = session.add_instrument(
            format!("S{g}"),
            p[g as usize].direction,
            None,
            None,
            PositionPolicy::Net,
        );
        let g = g as usize;
        if !w.bars[g].is_empty() {
            session.set_bars(i, w.bars[g].clone()).unwrap();
        }
        if let Some(t) = &w.ticks[g] {
            session.set_ticks(i, t.clone()).unwrap();
        }
        if !w.depth[g].is_empty() {
            session.set_depth(i, w.depth[g].clone()).unwrap();
        }
    }
    session.seal();
    let mut trace = Vec::new();
    while let Some(entry) = session.current() {
        trace.push((entry.instrument, entry.local_idx));
        session.apply_current(intent(sid, entry.instrument, entry.local_idx));
    }
    digest(session.finish(), trace)
}

#[test]
fn tick_books_over_one_store_match_standalone_sessions() {
    let subsets: Vec<Vec<u32>> =
        vec![vec![0, 1, 2, 3, 4], vec![1], vec![0, 4], vec![1, 2, 4], vec![2, 3]];
    for (seed, mode) in [
        (1u64, AccountMode::Cash),
        (2, AccountMode::Margin { leverage: 4.0 }),
        (3, AccountMode::Cash),
    ] {
        let world = make_world(seed);
        let (mut core, sids) = build(world_store(&world), mode, 0, &subsets);
        let got = run_core(&mut core, &sids);
        let mut any_trades = false;
        for (i, subset) in subsets.iter().enumerate() {
            let want = run_standalone_world(&world, mode, i as u32, subset);
            assert_same(&got[i], &want, &format!("seed {seed} strategy {i}"));
            any_trades |= got[i].trades != "[]";
        }
        assert!(any_trades, "the scenario must trade");
    }
}

/// Replay `store`'s tape into a core that holds no data, one pushed event at
/// a time, and apply the same intents as [`run_core`].
fn run_pushed(store: &MarketStore, mode: AccountMode, subsets: &[Vec<u32>]) -> Vec<Outcome> {
    let mut empty = MarketStore::new();
    for i in 0..store.len() {
        empty.add_instrument(&format!("S{i}")).unwrap();
    }
    let (mut core, sids) = build(empty, mode, 0, subsets);
    let mut cursor = MergeCursor::new(store);
    let mut traces: Vec<Trace> = vec![Vec::new(); sids.len()];
    while let Some(e) = cursor.next() {
        let payload = match e.kind {
            EventKind::Bar => Pushed::Bar(cursor.bar(&e).unwrap()),
            EventKind::Trade => Pushed::Trade(cursor.trade(&e).unwrap()),
            EventKind::Quote => Pushed::Quote(cursor.quote(&e).unwrap()),
            EventKind::Depth => Pushed::Depth(cursor.depth(&e).unwrap()),
        };
        core.advance_pushed(e, payload).unwrap();
        for sid in core.subscribers(e.instrument).to_vec() {
            let local = core.local_of(sid, e.instrument).unwrap();
            traces[sid.index()].push((local, e.local_idx as usize));
            core.apply(sid, intent(sid.raw(), local, e.local_idx as usize)).unwrap();
        }
        core.clear_current();
    }
    sids.iter().zip(traces).map(|(&sid, t)| digest(core.finish(sid).unwrap(), t)).collect()
}

#[test]
fn a_pushed_tape_gives_the_results_of_the_same_tape_read_from_a_store() {
    let subsets: Vec<Vec<u32>> = vec![vec![0, 1, 2, 3, 4], vec![1], vec![0, 4], vec![2, 3]];
    for (seed, mode) in [(1u64, AccountMode::Cash), (2, AccountMode::Margin { leverage: 4.0 })] {
        let world = make_world(seed);
        let store = world_store(&world);
        let pushed = run_pushed(&store, mode, &subsets);
        let (mut core, sids) = build(world_store(&world), mode, 0, &subsets);
        let batch = run_core(&mut core, &sids);
        let mut any_trades = false;
        for i in 0..subsets.len() {
            assert_same(&pushed[i], &batch[i], &format!("seed {seed} strategy {i}"));
            any_trades |= batch[i].trades != "[]";
        }
        assert!(any_trades, "the scenario must trade");
    }
}

#[test]
fn ordinals_count_every_event_of_an_instrument() {
    let world = make_world(5);
    let (mut core, sids) = build(world_store(&world), AccountMode::Cash, 0, &[vec![1]]);
    let out = run_core(&mut core, &sids);
    let n = out[0].trace.len();
    assert!(n > 260, "prints, quotes and books all count: {n}");
    assert_eq!(out[0].trace, (0..n).map(|i| (0, i)).collect::<Trace>());
}

#[test]
fn atr_stops_are_refused_on_an_instrument_with_ticks() {
    let world = make_world(1);
    let built = RuntimeBuilder::new(
        world_store(&world),
        params(5),
        config(),
        AccountMode::Cash,
        PositionPolicy::Net,
    );
    assert!(built.is_err());
    assert!(built.err().unwrap().to_string().contains("S0"));
}

#[test]
fn tick_watermarks_follow_trades_and_quotes_and_bar_watermarks_follow_bars() {
    let world = make_world(7);
    let (mut core, _) = build(world_store(&world), AccountMode::Cash, 0, &[vec![0, 1, 2, 3, 4]]);
    let store = core.store().clone();
    let mut cursor = MergeCursor::new(&store);
    let mut bars = [0u32; INSTRUMENTS];
    while let Some(e) = core.advance(&mut cursor).unwrap() {
        let i = e.instrument.index();
        if e.kind == EventKind::Bar {
            bars[i] += 1;
        }
        for (k, &seen) in bars.iter().enumerate() {
            let id = InstrumentId::new(k as u32);
            assert_eq!(core.watermark(id), u64::from(seen));
            assert_eq!(
                core.tick_watermark(id),
                cursor.visible_tick_row(id).map_or(0, |r| u64::from(r) + 1)
            );
        }
    }
}

#[test]
fn prints_that_complete_a_primary_bar_feed_the_bank() {
    use crate::data::{builder_for_with, BuilderParams, SourceRecord};
    let world = make_world(11);
    let mut b = RuntimeBuilder::new(
        world_store(&world),
        params(0),
        config(),
        AccountMode::Cash,
        PositionPolicy::Net,
    )
    .unwrap();
    let sid = b.add_strategy(&ids(&[0, 1])).unwrap();
    let spec = BarSpec::new(4, AggregationUnit::Tick).unwrap();
    let inst0 = InstrumentId::new(0);
    b.set_primary_bars(inst0, spec, 0).unwrap();
    let reg = b
        .register_indicator(sid, IndKey::new(IndInput::Primary(inst0), IndSpec::Sma(3)), None)
        .unwrap();
    let mut core = b.build().unwrap();
    let mut private = StreamingSma::new(3);
    let mut reference = builder_for_with(spec, 0, BuilderParams { brick_size: 0.0 }).unwrap();
    let store = core.store().clone();
    let mut cursor = MergeCursor::new(&store);
    let (mut completed, mut checked) = (0, 0);
    while let Some(e) = core.advance(&mut cursor).unwrap() {
        let cur = *core.current().unwrap();
        if e.instrument == inst0 && e.kind == EventKind::Trade {
            let ScheduleData::Trade(t) = cur.data else { panic!("a trade event carries a trade") };
            let want = reference.push(&SourceRecord::trade(t.timestamp, t.price, t.size));
            assert_eq!(cur.primary.map(|b| b.close.to_bits()), want.map(|b| b.close.to_bits()));
            if let Some(bar) = want {
                completed += 1;
                let value = private.update(bar.close);
                core.bank().reveal_primary(sid, inst0);
                let got = core.bank().read(reg.slot);
                assert_eq!(got.initialized, value.is_some());
                assert_eq!(got.values[0].to_bits(), value.map_or(0, f64::to_bits));
                checked += 1;
            }
        } else {
            assert!(cur.primary.is_none());
        }
        if core.subscribers(e.instrument).contains(&sid) {
            core.apply(sid, StepInput::default()).unwrap();
        }
    }
    assert!(completed > 20 && checked == completed);
    // One bank update per completed primary bar, nothing for the other events.
    assert_eq!(core.bank().updates(), completed as u64);
}

// -- an instrument added to a running core ------------------------------------------------

/// A core over `n` instruments that hold no data, fed by pushes like a live node's.
fn live_core(n: usize, subsets: &[Vec<u32>]) -> (RuntimeCore, Vec<StrategyId>, LiveFeed) {
    let mut store = MarketStore::new();
    for i in 0..n {
        store.add_instrument(&format!("L{i}")).unwrap();
    }
    let params = vec![InstrumentParams::default(); n];
    let mut b =
        RuntimeBuilder::new(store, params, config(), AccountMode::Cash, PositionPolicy::Net)
            .unwrap();
    let sids = subsets.iter().map(|s| b.add_strategy(&ids(s)).unwrap()).collect();
    let feed = LiveFeed::new((0..n).map(|i| format!("L{i}")).collect(), 64);
    (b.build().unwrap(), sids, feed)
}

/// Push one bar of `g` at `ts` and let every subscriber apply `intent`.
fn push_bar(
    core: &mut RuntimeCore,
    feed: &mut LiveFeed,
    traces: &mut [Trace],
    g: u32,
    ts: i64,
    px: f64,
) {
    let id = InstrumentId::new(g);
    let bar = KernelBar {
        timestamp: ts,
        open: px,
        high: px + 1.0,
        low: px - 1.0,
        close: px,
        volume: 10.0,
    };
    let event = feed.push_bar(id, ts, bar).unwrap();
    core.advance_pushed(event, Pushed::Bar(bar)).unwrap();
    for sid in core.subscribers(id).to_vec() {
        let local = core.local_of(sid, id).unwrap();
        traces[sid.index()].push((local, event.local_idx as usize));
        core.apply(sid, intent(sid.raw(), local, event.local_idx as usize)).unwrap();
    }
    core.clear_current();
}

fn tape() -> Vec<(u32, i64, f64)> {
    let mut rng = Lcg(77);
    (0..240)
        .map(|k| {
            let g = (k % 3) as u32;
            (g, 10 * (k / 3) as i64, 100.0 + 5.0 * g as f64 + rng.unit() * 6.0)
        })
        .collect()
}

#[test]
fn an_added_instrument_is_traded_as_if_it_had_been_there_from_the_start() {
    let tape = tape();
    let subsets = vec![vec![0, 1, 2], vec![0], vec![1, 2]];
    // The twin: three instruments from the start. Instrument 2 has no event before the cut.
    let cut = 10 * 20;
    let (mut twin, sids, mut twin_feed) = live_core(3, &subsets);
    let mut want_traces = vec![Vec::new(); sids.len()];
    for &(g, ts, px) in tape.iter().filter(|e| e.0 != 2 || e.1 >= cut) {
        push_bar(&mut twin, &mut twin_feed, &mut want_traces, g, ts, px);
    }
    // The subject: two instruments, the third added at the cut to the strategies that trade it.
    let (mut core, sids, mut feed) = live_core(2, &[vec![0, 1], vec![0], vec![1]]);
    let mut traces = vec![Vec::new(); sids.len()];
    let mut added = false;
    for &(g, ts, px) in tape.iter().filter(|e| e.0 != 2 || e.1 >= cut) {
        if g == 2 && !added {
            let id = core
                .add_instrument(
                    "L2",
                    &InstrumentParams::default(),
                    &[sids[0], sids[2]],
                    None,
                    false,
                )
                .unwrap();
            feed.add_instrument("L2".into(), Some(8));
            assert_eq!(id, InstrumentId::new(2));
            assert_eq!(core.instruments_of(sids[0]), 3);
            assert_eq!(core.instruments_of(sids[1]), 1);
            assert_eq!(core.instruments_of(sids[2]), 2);
            assert_eq!(feed.capacity_of(id), 8);
            added = true;
        }
        push_bar(&mut core, &mut feed, &mut traces, g, ts, px);
    }
    assert!(added);
    let mut any_trades = false;
    for (k, sid) in sids.iter().enumerate() {
        assert_eq!(traces[k], want_traces[k], "strategy {k}: events seen");
        let got = digest(core.finish(*sid).unwrap(), traces[k].clone());
        let want = digest(twin.finish(*sid).unwrap(), want_traces[k].clone());
        assert_same(&got, &want, &format!("strategy {k}"));
        any_trades |= got.trades != "[]";
    }
    assert!(any_trades, "the scenario must trade");
}

#[test]
fn an_instrument_in_use_is_not_retired_and_a_retired_one_is_not_fed() {
    let (mut core, sids, mut feed) = live_core(2, &[vec![0, 1]]);
    let mut traces = vec![Vec::new()];
    let id = core.add_instrument("L2", &InstrumentParams::default(), &sids, None, false).unwrap();
    feed.add_instrument("L2".into(), None);
    let book = core.book_mut(sids[0]).unwrap();
    let order = book.kernel_mut(2).submit_order_full(
        crate::execution::orders::OrderSide::Buy,
        crate::execution::orders::QtySpec::Units(1.0),
        crate::execution::orders::OrderKind::Limit { price: 1.0 },
        crate::execution::orders::TimeInForce::Gtc,
        0,
        0,
        "w".into(),
        None,
        None,
        false,
        false,
        None,
    );
    assert_eq!(core.instrument_in_use(id), vec![(sids[0], 0, 1)]);
    assert!(core.book_mut(sids[0]).unwrap().kernel_mut(2).cancel_order(0, order));
    assert!(core.instrument_in_use(id).is_empty());
    core.retire_instrument(id).unwrap();
    assert!(core.subscribers(id).is_empty());
    // Events on the others still run, with the retired instrument's slot in every book.
    push_bar(&mut core, &mut feed, &mut traces, 0, 10, 100.0);
    assert_eq!(core.instruments_of(sids[0]), 3);
}

#[test]
fn a_retired_instrument_that_comes_back_is_traded_as_if_it_had_never_left() {
    let tape = tape();
    let subsets = vec![vec![0, 1, 2], vec![1, 2]];
    let cut = 10 * 20;
    let events: Vec<_> = tape.iter().filter(|e| e.0 != 2 || e.1 >= cut).collect();
    let (mut twin, sids, mut twin_feed) = live_core(3, &subsets);
    let mut want_traces = vec![Vec::new(); sids.len()];
    for &&(g, ts, px) in &events {
        push_bar(&mut twin, &mut twin_feed, &mut want_traces, g, ts, px);
    }
    let (mut core, sids, mut feed) = live_core(3, &subsets);
    let g2 = InstrumentId::new(2);
    core.retire_instrument(g2).unwrap();
    feed.retire_instrument(g2);
    assert!(core.subscribers(g2).is_empty());
    let mut traces = vec![Vec::new(); sids.len()];
    let mut back = false;
    for &&(g, ts, px) in &events {
        if g == 2 && !back {
            core.revive_instrument(g2, &sids, None, None).unwrap();
            feed.revive_instrument(g2, None);
            back = true;
        }
        push_bar(&mut core, &mut feed, &mut traces, g, ts, px);
    }
    assert!(back);
    for (k, sid) in sids.iter().enumerate() {
        assert_eq!(traces[k], want_traces[k], "strategy {k}: events seen");
        let got = digest(core.finish(*sid).unwrap(), traces[k].clone());
        let want = digest(twin.finish(*sid).unwrap(), want_traces[k].clone());
        assert_same(&got, &want, &format!("strategy {k}"));
    }
    assert!(
        core.revive_instrument(g2, &[], None, None).is_err(),
        "a live instrument is not revived"
    );
}
