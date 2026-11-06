//! The bank must show every strategy exactly what a private streaming core,
//! updated in the order a strategy runs today, would show at the same point.

use std::cell::{Cell, RefCell};
use std::collections::HashSet;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed, TestRunner};

use crate::indicators::streaming::{
    StreamingAtr, StreamingBollinger, StreamingDonchian, StreamingEma, StreamingIndicator,
    StreamingMacd, StreamingRoc, StreamingRsi, StreamingSma, StreamingStdDev, StreamingWma,
};
use crate::runtime::ids::{AggId, InstrumentId, StrategyId};
use crate::runtime::indicators::{
    IndInput, IndKey, IndKind, IndSpec, IndicatorBank, IndicatorRegistry, InputField,
    LocalStreamId, Ohlc, SlotId,
};
use crate::runtime::memory::HeapSize;

// ---------------------------------------------------------------------------
// Reference: a private core per registration, as strategies hold today
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum Private {
    Sma(StreamingSma),
    Ema(StreamingEma),
    Wilder(StreamingEma),
    Wma(StreamingWma),
    Roc(StreamingRoc),
    StdDev(StreamingStdDev),
    Rsi(StreamingRsi),
    Atr(StreamingAtr),
    Donchian(StreamingDonchian),
    Bollinger(StreamingBollinger),
    Macd(StreamingMacd),
}

type Bits = (bool, [u64; 3]);

fn bits1(v: Option<f64>) -> Bits {
    (v.is_some(), [v.unwrap_or(0.0).to_bits(), 0.0f64.to_bits(), 0.0f64.to_bits()])
}

fn bits2(v: Option<(f64, f64)>) -> Bits {
    let (a, b) = v.unwrap_or((0.0, 0.0));
    (v.is_some(), [a.to_bits(), b.to_bits(), 0.0f64.to_bits()])
}

fn bits3(v: Option<(f64, f64, f64)>) -> Bits {
    let (a, b, c) = v.unwrap_or((0.0, 0.0, 0.0));
    (v.is_some(), [a.to_bits(), b.to_bits(), c.to_bits()])
}

impl Private {
    fn new(spec: IndSpec) -> Self {
        match spec {
            IndSpec::Sma(p) => Private::Sma(StreamingSma::new(p)),
            IndSpec::Ema(p) => Private::Ema(StreamingEma::new(p)),
            IndSpec::WilderMa(p) => Private::Wilder(StreamingEma::wilder(p)),
            IndSpec::Wma(p) => Private::Wma(StreamingWma::new(p)),
            IndSpec::Roc(p) => Private::Roc(StreamingRoc::new(p)),
            IndSpec::StdDev(p) => Private::StdDev(StreamingStdDev::new(p)),
            IndSpec::Rsi(p) => Private::Rsi(StreamingRsi::new(p)),
            IndSpec::Atr(p) => Private::Atr(StreamingAtr::new(p)),
            IndSpec::Donchian(p) => Private::Donchian(StreamingDonchian::new(p)),
            IndSpec::Bollinger { period, k } => {
                Private::Bollinger(StreamingBollinger::new(period, k))
            }
            IndSpec::Macd { fast, slow, signal } => {
                Private::Macd(StreamingMacd::new(fast, slow, signal))
            }
            // The model is the legacy cores' reference; expressions have their own tests
            // (`indicators_expr_tests`).
            IndSpec::Expr { .. } => unreachable!("the generator makes no expression specs"),
        }
    }

    fn update(&mut self, bar: Ohlc, field: InputField) {
        let x = match field {
            InputField::Close | InputField::Bar => bar.close,
            InputField::Open => bar.open,
            InputField::High => bar.high,
            InputField::Low => bar.low,
        };
        match self {
            Private::Sma(c) => drop(c.update(x)),
            Private::Ema(c) | Private::Wilder(c) => drop(c.update(x)),
            Private::Wma(c) => drop(c.update(x)),
            Private::Roc(c) => drop(c.update(x)),
            Private::StdDev(c) => drop(c.update(x)),
            Private::Rsi(c) => drop(c.update(x)),
            Private::Atr(c) => drop(c.update_bar(bar.high, bar.low, bar.close)),
            Private::Donchian(c) => drop(c.update_bar(bar.high, bar.low)),
            Private::Bollinger(c) => drop(c.update(x)),
            Private::Macd(c) => drop(c.update(x)),
        }
    }

    /// What the strategy sees: `value()` and `initialized()`.
    fn shows(&self) -> Bits {
        match self {
            Private::Sma(c) => bits1(c.value()),
            Private::Ema(c) | Private::Wilder(c) => bits1(c.value()),
            Private::Wma(c) => bits1(c.value()),
            Private::Roc(c) => bits1(c.value()),
            Private::StdDev(c) => bits1(c.value()),
            Private::Rsi(c) => bits1(c.value()),
            Private::Atr(c) => bits1(c.value()),
            Private::Donchian(c) => bits2(c.value()),
            Private::Bollinger(c) => bits3(c.value()),
            Private::Macd(c) => bits3(c.value()),
        }
    }
}

// ---------------------------------------------------------------------------
// Scenario
// ---------------------------------------------------------------------------

/// Decodes structure from a vector of raw integers, so shrinking the vector
/// shrinks the scenario.
struct Genes {
    v: Vec<u32>,
    pos: usize,
}

impl Genes {
    fn next(&mut self, n: usize) -> usize {
        let g = self.v[self.pos % self.v.len()];
        self.pos += 1;
        g as usize % n
    }

    fn unit(&mut self) -> f64 {
        self.next(1 << 20) as f64 / (1u32 << 20) as f64
    }
}

#[derive(Debug, Clone, Copy)]
enum DefInput {
    Primary(u32),
    AllPrimary,
    Stream(u32),
}

#[derive(Debug, Clone, Copy)]
struct IndDef {
    spec: IndSpec,
    field: InputField,
    input: DefInput,
}

#[derive(Debug)]
struct Strat {
    set: Vec<u32>,
    /// The aggregation behind each local stream, in subscription order.
    streams: Vec<u32>,
    inds: Vec<IndDef>,
}

#[derive(Debug)]
struct Step {
    inst: u32,
    bar: Ohlc,
    /// Bars each aggregation emits in this event.
    emissions: Vec<Vec<Ohlc>>,
}

#[derive(Debug)]
struct Scenario {
    strats: Vec<Strat>,
    steps: Vec<Step>,
}

fn palette() -> Vec<IndSpec> {
    vec![
        IndSpec::Sma(2),
        IndSpec::Sma(3),
        IndSpec::Sma(5),
        IndSpec::Ema(3),
        IndSpec::Ema(4),
        IndSpec::WilderMa(3),
        IndSpec::Wma(3),
        IndSpec::Wma(4),
        IndSpec::Roc(2),
        IndSpec::StdDev(3),
        IndSpec::Rsi(3),
        IndSpec::Rsi(4),
        IndSpec::Atr(3),
        IndSpec::Donchian(2),
        IndSpec::Donchian(3),
        IndSpec::Bollinger { period: 3, k: 2.0 },
        IndSpec::Bollinger { period: 3, k: 1.5 },
        IndSpec::Macd { fast: 2, slow: 3, signal: 2 },
        IndSpec::Macd { fast: 2, slow: 4, signal: 3 },
    ]
}

fn is_bar_spec(spec: IndSpec) -> bool {
    matches!(spec, IndSpec::Atr(_) | IndSpec::Donchian(_))
}

fn bar_around(g: &mut Genes, level: f64) -> Ohlc {
    let open = level * (1.0 + (g.unit() - 0.5) * 0.04);
    let close = level * (1.0 + (g.unit() - 0.5) * 0.04);
    let high = open.max(close) * (1.0 + g.unit() * 0.01);
    let low = open.min(close) * (1.0 - g.unit() * 0.01);
    // Rarely a bad print, to exercise the NaN handling of the cores.
    let close = if g.next(60) == 0 { f64::NAN } else { close };
    Ohlc { open, high, low, close, volume: 1_000.0, ts: 0 }
}

fn decode(genes: Vec<u32>) -> Scenario {
    let mut g = Genes { v: genes, pos: 0 };
    let n_inst = 2 + g.next(4);
    let n_aggs = g.next(4);
    let agg_src: Vec<u32> = (0..n_aggs).map(|_| g.next(n_inst) as u32).collect();
    let fields = [
        InputField::Close,
        InputField::Close,
        InputField::Close,
        InputField::High,
        InputField::Low,
        InputField::Open,
    ];
    let palette = palette();

    let n_strats = 2 + g.next(7);
    let mut strats = Vec::new();
    for _ in 0..n_strats {
        let mask = 1 + g.next((1 << n_inst) - 1);
        let set: Vec<u32> = (0..n_inst as u32).filter(|i| mask >> i & 1 == 1).collect();
        let mut streams: Vec<u32> = Vec::new();
        for (a, &src) in agg_src.iter().enumerate() {
            if set.contains(&src) {
                let roll = g.next(10);
                if roll < 8 {
                    streams.push(a as u32);
                }
                if (6..8).contains(&roll) {
                    streams.push(a as u32);
                }
            }
        }
        for i in (1..streams.len()).rev() {
            streams.swap(i, g.next(i + 1));
        }
        let n_inds = 1 + g.next(6);
        let mut inds = Vec::new();
        for _ in 0..n_inds {
            let t = g.next(100);
            let input = if t < 40 {
                DefInput::Primary(set[g.next(set.len())])
            } else if t < 65 || streams.is_empty() {
                DefInput::AllPrimary
            } else {
                DefInput::Stream(g.next(streams.len()) as u32)
            };
            // Half the picks come from a few popular specs, so keys overlap.
            let spec =
                if g.next(2) == 0 { palette[g.next(3)] } else { palette[g.next(palette.len())] };
            let field = if is_bar_spec(spec) { InputField::Bar } else { fields[g.next(6)] };
            inds.push(IndDef { spec, field, input });
        }
        strats.push(Strat { set, streams, inds });
    }

    let mut levels: Vec<f64> = (0..n_inst).map(|i| 100.0 * (i + 1) as f64).collect();
    let agg_levels: Vec<f64> = (0..n_aggs).map(|a| 50.0 + 25.0 * a as f64).collect();
    let n_steps = 25 + g.next(50);
    let mut steps = Vec::new();
    for _ in 0..n_steps {
        let inst = g.next(n_inst);
        let bar = bar_around(&mut g, levels[inst]);
        if bar.close.is_finite() {
            levels[inst] = bar.close;
        }
        let mut emissions = vec![Vec::new(); n_aggs];
        for (a, &src) in agg_src.iter().enumerate() {
            if src as usize == inst {
                let m = match g.next(10) {
                    0..=2 => 0,
                    3..=6 => 1,
                    7 | 8 => 2,
                    _ => 3,
                };
                emissions[a] = (0..m).map(|_| bar_around(&mut g, agg_levels[a])).collect();
            }
        }
        steps.push(Step { inst: inst as u32, bar, emissions });
    }
    Scenario { strats, steps }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Order {
    /// Composites, then primary: today's order.
    Correct,
    /// Mutation: the primary is revealed before the composites.
    PrimaryFirst,
}

#[derive(Default)]
struct Coverage {
    cases: Cell<usize>,
    shared_cases: Cell<usize>,
    multi_emission_cases: Cell<usize>,
    all_primary_cases: Cell<usize>,
    partial_reveals: Cell<usize>,
    checks: Cell<usize>,
    kinds: RefCell<HashSet<IndKind>>,
}

fn key_of(
    def: &IndDef,
    order: crate::runtime::indicators::PrimaryOrder,
    agg: Option<u32>,
) -> IndKey {
    let input = match def.input {
        DefInput::Primary(i) => IndInput::Primary(InstrumentId::new(i)),
        DefInput::AllPrimary => IndInput::AllPrimary(order),
        DefInput::Stream(_) => IndInput::Composite(AggId::new(agg.unwrap())),
    };
    IndKey::new(input, def.spec).with_field(def.field)
}

struct RefSlot {
    slot: SlotId,
    def: IndDef,
    core: Private,
}

fn run(sc: &Scenario, order: Order, cov: &Coverage) -> Result<(), String> {
    // Register, in strategy order.
    let mut reg = IndicatorRegistry::new();
    let mut seen: HashSet<IndKey> = HashSet::new();
    let mut refs: Vec<Vec<RefSlot>> = Vec::new();
    let mut registrations = 0;
    for (s, strat) in sc.strats.iter().enumerate() {
        let insts: Vec<InstrumentId> = strat.set.iter().map(|&i| InstrumentId::new(i)).collect();
        let prim_order = reg.primary_order(&insts).map_err(|e| e.to_string())?;
        let mut slots = Vec::new();
        for def in &strat.inds {
            let (agg, stream) = match def.input {
                DefInput::Stream(l) => (Some(strat.streams[l as usize]), Some(LocalStreamId(l))),
                _ => (None, None),
            };
            let key = key_of(def, prim_order, agg);
            let first_cell = reg.cells();
            let r =
                reg.register(StrategyId::new(s as u32), key, stream).map_err(|e| e.to_string())?;
            registrations += 1;
            if r.shared != seen.contains(&key) {
                return Err(format!("shared flag wrong for {key:?}"));
            }
            if !r.shared && r.ind.index() != first_cell {
                return Err("cells are not numbered in first-registration order".into());
            }
            seen.insert(key);
            cov.kinds.borrow_mut().insert(key.kind);
            slots.push(RefSlot { slot: r.slot, def: *def, core: Private::new(def.spec) });
        }
        refs.push(slots);
    }
    if reg.cells() != seen.len() || reg.registrations() != registrations {
        return Err("distinct cells differ from distinct keys".into());
    }
    let bank = IndicatorBank::new(&reg);
    if bank.cells() != seen.len() || bank.registrations() != registrations {
        return Err("bank counts differ from the registry".into());
    }
    cov.cases.set(cov.cases.get() + 1);
    if seen.len() < registrations {
        cov.shared_cases.set(cov.shared_cases.get() + 1);
    }
    if sc.strats.iter().flat_map(|s| &s.inds).any(|d| matches!(d.input, DefInput::AllPrimary)) {
        cov.all_primary_cases.set(cov.all_primary_cases.get() + 1);
    }

    let check = |s: usize, refs: &[Vec<RefSlot>], when: &str, step: usize| -> Result<(), String> {
        for r in &refs[s] {
            cov.checks.set(cov.checks.get() + 1);
            let got = bank.read(r.slot).bits();
            let want = r.core.shows();
            if got != want {
                return Err(format!(
                    "step {step}, strategy {s}, {when}: {:?} {:?} reads {got:?}, private core shows {want:?}",
                    r.def.spec, r.def.input
                ));
            }
        }
        Ok(())
    };

    let mut multi = false;
    for (n, step) in sc.steps.iter().enumerate() {
        let sid = |s: usize| StrategyId::new(s as u32);
        let inst = InstrumentId::new(step.inst);
        bank.begin_event();
        for (a, bars) in step.emissions.iter().enumerate() {
            for bar in bars {
                bank.emit_composite(AggId::new(a as u32), *bar).map_err(|e| e.to_string())?;
            }
        }
        bank.emit_primary(inst, step.bar).map_err(|e| e.to_string())?;

        for (s, strat) in sc.strats.iter().enumerate() {
            if !strat.set.contains(&step.inst) {
                continue;
            }
            check(s, &refs, "before any reveal", n)?;
            if order == Order::PrimaryFirst {
                bank.reveal_primary(sid(s), inst);
            }
            for (l, &agg) in strat.streams.iter().enumerate() {
                let bars = &step.emissions[agg as usize];
                let consumed = strat
                    .inds
                    .iter()
                    .any(|d| matches!(d.input, DefInput::Stream(x) if x as usize == l));
                multi |= bars.len() > 1 && consumed;
                for (j, bar) in bars.iter().enumerate() {
                    for r in &mut refs[s] {
                        if matches!(r.def.input, DefInput::Stream(x) if x as usize == l) {
                            r.core.update(*bar, r.def.field);
                        }
                    }
                    bank.reveal_stream(sid(s), LocalStreamId(l as u32), j as u32 + 1);
                    if j + 1 < bars.len() {
                        cov.partial_reveals.set(cov.partial_reveals.get() + 1);
                    }
                    check(s, &refs, "after a composite emission", n)?;
                }
            }
            for r in &mut refs[s] {
                match r.def.input {
                    DefInput::Primary(i) if i == step.inst => r.core.update(step.bar, r.def.field),
                    DefInput::AllPrimary => r.core.update(step.bar, r.def.field),
                    _ => {}
                }
            }
            if order == Order::Correct {
                bank.reveal_primary(sid(s), inst);
            }
            check(s, &refs, "after the primary", n)?;
        }
        for s in 0..sc.strats.len() {
            check(s, &refs, "end of event", n)?;
        }
    }
    if multi {
        cov.multi_emission_cases.set(cov.multi_emission_cases.get() + 1);
    }
    // One update per cell per emission, whatever the number of readers.
    let mut expected = 0u64;
    for step in &sc.steps {
        for key in reg.keys() {
            expected += match key.input {
                IndInput::Primary(i) => u64::from(i.raw() == step.inst),
                IndInput::AllPrimary(o) => u64::from(
                    reg.order_instruments(o).unwrap().iter().any(|i| i.raw() == step.inst),
                ),
                IndInput::Composite(a) => step.emissions[a.index()].len() as u64,
                IndInput::Universe(_) => 0,
            };
        }
    }
    if bank.updates() != expected {
        return Err(format!("{} core updates, expected {expected}", bank.updates()));
    }
    Ok(())
}

fn runner(seed: u64) -> TestRunner {
    TestRunner::new(Config {
        cases: 400,
        failure_persistence: None,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(seed),
        max_shrink_iters: 2048,
        ..Config::default()
    })
}

fn genes() -> impl Strategy<Value = Vec<u32>> {
    prop::collection::vec(any::<u32>(), 700)
}

#[test]
fn readers_see_what_private_cores_show_at_every_reveal_point() {
    let cov = Coverage::default();
    runner(0xB4A4_0001)
        .run(&genes(), |g| {
            let sc = decode(g);
            run(&sc, Order::Correct, &cov).map_err(TestCaseError::fail)
        })
        .unwrap();
    // Guard against a generator that degenerates into trivial cases.
    assert_eq!(cov.cases.get(), 400);
    assert!(cov.shared_cases.get() > 200, "shared in {}", cov.shared_cases.get());
    assert!(cov.multi_emission_cases.get() > 100, "multi {}", cov.multi_emission_cases.get());
    assert!(cov.all_primary_cases.get() > 200, "all-primary {}", cov.all_primary_cases.get());
    assert!(cov.partial_reveals.get() > 500, "partial {}", cov.partial_reveals.get());
    assert!(cov.checks.get() > 100_000, "checks {}", cov.checks.get());
    assert_eq!(cov.kinds.borrow().len(), 11, "kinds {:?}", cov.kinds.borrow());
}

#[test]
fn revealing_the_primary_before_the_composites_is_caught() {
    let cov = Coverage::default();
    let result = runner(0xB4A4_0001).run(&genes(), |g| {
        let sc = decode(g);
        run(&sc, Order::PrimaryFirst, &cov).map_err(TestCaseError::fail)
    });
    let err = result.expect_err("the mutated order must fail the equivalence test");
    assert!(err.to_string().contains("private core shows"), "{err}");
}

// ---------------------------------------------------------------------------
// Hand-written protocol cases
// ---------------------------------------------------------------------------

fn flat(x: f64) -> Ohlc {
    Ohlc { open: x, high: x, low: x, close: x, volume: 1_000.0, ts: 0 }
}

fn value(bank: &IndicatorBank, slot: SlotId) -> Option<f64> {
    let r = bank.read(slot);
    r.initialized.then_some(r.values[0])
}

#[test]
fn reveal_states_for_a_multi_emission_event() {
    // Two strategies on one SMA(2) over a composite aggregation; the event
    // emits 10, 20 and 30 after earlier bars 2 and 4.
    let mut reg = IndicatorRegistry::new();
    let key = IndKey::new(IndInput::Composite(AggId::new(0)), IndSpec::Sma(2));
    let a = reg.register(StrategyId::new(0), key, Some(LocalStreamId(0))).unwrap();
    let b = reg.register(StrategyId::new(1), key, Some(LocalStreamId(5))).unwrap();
    let bank = IndicatorBank::new(&reg);
    let (a, b) = (a.slot, b.slot);
    assert_eq!(bank.cells(), 1);

    bank.begin_event();
    bank.emit_composite(AggId::new(0), flat(2.0)).unwrap();
    bank.emit_composite(AggId::new(0), flat(4.0)).unwrap();
    bank.reveal_stream(StrategyId::new(0), LocalStreamId(0), 2);
    bank.reveal_stream(StrategyId::new(1), LocalStreamId(5), 2);
    assert_eq!((value(&bank, a), value(&bank, b)), (Some(3.0), Some(3.0)));

    bank.begin_event();
    for x in [10.0, 20.0, 30.0] {
        bank.emit_composite(AggId::new(0), flat(x)).unwrap();
    }
    // Nobody revealed yet: the value before the event.
    assert_eq!((value(&bank, a), value(&bank, b)), (Some(3.0), Some(3.0)));
    bank.reveal_stream(StrategyId::new(0), LocalStreamId(0), 1);
    assert_eq!(value(&bank, a), Some(7.0)); // (4 + 10) / 2
    assert_eq!(value(&bank, b), Some(3.0)); // the other reader is unaffected
    bank.reveal_stream(StrategyId::new(0), LocalStreamId(0), 2);
    assert_eq!(value(&bank, a), Some(15.0));
    bank.reveal_stream(StrategyId::new(0), LocalStreamId(0), 3);
    assert_eq!(value(&bank, a), Some(25.0));
    assert_eq!(value(&bank, b), Some(3.0));
    bank.reveal_stream(StrategyId::new(1), LocalStreamId(5), 3);
    assert_eq!(value(&bank, b), Some(25.0));
    assert_eq!(bank.updates(), 5);

    // A reader that never revealed sees the cell as of the end of the next
    // event in which the cell is not fed.
    bank.begin_event();
    assert_eq!((value(&bank, a), value(&bank, b)), (Some(25.0), Some(25.0)));
}

#[test]
fn shared_cells_update_once_per_emission() {
    let mut reg = IndicatorRegistry::new();
    let key = IndKey::new(IndInput::Primary(InstrumentId::new(0)), IndSpec::Rsi(14));
    let slots: Vec<SlotId> =
        (0..50).map(|s| reg.register(StrategyId::new(s), key, None).unwrap().slot).collect();
    let bank = IndicatorBank::new(&reg);
    let handles: Vec<_> = (0..50)
        .map(|s| bank.primary_handle(StrategyId::new(s), InstrumentId::new(0)).unwrap())
        .collect();
    assert!(bank.primary_handle(StrategyId::new(0), InstrumentId::new(1)).is_none());
    assert!(bank.stream_handle(StrategyId::new(0), LocalStreamId(0)).is_none());
    for i in 0..100 {
        bank.begin_event();
        bank.emit_primary(InstrumentId::new(0), flat(100.0 + i as f64)).unwrap();
        for (s, h) in handles.iter().enumerate() {
            // Half through a handle, half through the lookup.
            if s % 2 == 0 {
                bank.reveal(*h, 1);
            } else {
                bank.reveal_primary(StrategyId::new(s as u32), InstrumentId::new(0));
            }
        }
    }
    assert_eq!(bank.updates(), 100);
    assert!(slots.iter().all(|&s| bank.read(s).bits() == bank.read(slots[0]).bits()));
}

#[test]
fn bank_is_send_sync_and_accounts_for_its_memory() {
    fn assert_sync<T: Send + Sync>() {}
    assert_sync::<IndicatorBank>();
    assert_sync::<std::sync::Arc<IndicatorBank>>();

    let build = |cells: u32, per_cell: u32| {
        let mut reg = IndicatorRegistry::new();
        for c in 0..cells {
            let key = IndKey::new(IndInput::Primary(InstrumentId::new(c)), IndSpec::Sma(10));
            for s in 0..per_cell {
                reg.register(StrategyId::new(s), key, None).unwrap();
            }
        }
        IndicatorBank::new(&reg).heap_bytes()
    };
    let (one, two) = (build(100, 1), build(100, 2));
    let (more_cells, _) = (build(200, 1), ());
    // Another reader of an existing cell costs a slot; another cell costs
    // more than that.
    let per_registration = (two - one) / 100;
    let per_cell = (more_cells - one) / 100;
    println!("bank: {per_cell} B per SMA(10) cell with one reader, {per_registration} B per extra reader");
    assert!(per_registration > 0 && per_registration < 64);
    assert!(per_cell > per_registration);
}
