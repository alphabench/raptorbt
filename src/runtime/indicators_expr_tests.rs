//! Expression cells in the bank: what a cell shows must be what the batch evaluator gives on
//! the same bars, with the reveal protocol of every other cell.

use crate::quant::eval::{eval, Inputs, Plan};
use crate::quant::kernel::canon_bits;
use crate::quant::lang::{compile, Env, Field, Program};
use crate::runtime::exprs::{self, ProgramId};
use crate::runtime::ids::{InstrumentId, StrategyId};
use crate::runtime::indicators::{
    CellCore, IndInput, IndKey, IndKind, IndSpec, IndicatorBank, IndicatorRegistry, Ohlc,
};
use crate::runtime::memory::HeapSize;

fn checked(src: &str) -> (Program, Plan) {
    let program = compile(src, &Env::default()).unwrap_or_else(|e| panic!("{src:?}: {e}"));
    let plan = Plan::new(&program).unwrap();
    (program, plan)
}

fn intern(src: &str) -> ProgramId {
    let (program, plan) = checked(src);
    exprs::intern(&program, &plan).unwrap()
}

/// Deterministic bars with a NaN close now and then and a varying volume.
fn bars(n: usize) -> Vec<Ohlc> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            let close =
                if i % 17 == 9 { f64::NAN } else { 100.0 + (x * 0.37).sin() * 5.0 + x * 0.05 };
            Ohlc {
                open: close - 0.4,
                high: close + 1.0,
                low: close - 1.0,
                close,
                volume: 50.0 + ((i * 7) % 40) as f64,
                ts: i as i64,
            }
        })
        .collect()
}

fn columns(bars: &[Ohlc]) -> [Vec<f64>; 5] {
    [
        bars.iter().map(|b| b.open).collect(),
        bars.iter().map(|b| b.high).collect(),
        bars.iter().map(|b| b.low).collect(),
        bars.iter().map(|b| b.close).collect(),
        bars.iter().map(|b| b.volume).collect(),
    ]
}

fn batch(plan: &Plan, bars: &[Ohlc]) -> Vec<f64> {
    let [o, h, l, c, v] = columns(bars);
    let inputs = Inputs::new()
        .field(Field::Open, &o)
        .field(Field::High, &h)
        .field(Field::Low, &l)
        .field(Field::Close, &c)
        .field(Field::Volume, &v);
    eval(plan, &inputs).unwrap().remove(0)
}

const SOURCES: [&str; 3] = [
    "ema(close, 3) - sma(close, 5) + sma(volume, 4) * 0.01",
    "sma(close, 3) > ema(close, 5) and not (volume < 60)",
    "(close - open) / (high - low) * sma(volume, 2)",
];

#[test]
fn equal_canonical_text_is_one_program() {
    let a = intern("sma(close, 3) > ema(close, 5)");
    let b = intern("sma( close,3 )  >  ema(close,5)");
    let c = intern("sma(close, 4) > ema(close, 5)");
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_eq!(exprs::info(a).canonical, exprs::info(b).canonical);
}

#[test]
fn a_cell_shows_the_batch_value_at_every_bar() {
    let feed = bars(120);
    for src in SOURCES {
        let (program, plan) = checked(src);
        let id = exprs::intern(&program, &plan).unwrap();
        let want = batch(&plan, &feed);
        let mut reg = IndicatorRegistry::new();
        let key =
            IndKey::new(IndInput::Primary(InstrumentId::new(0)), IndSpec::expr(id, 0).unwrap());
        let slot = reg.register(StrategyId::new(0), key, None).unwrap().slot;
        let bank = IndicatorBank::new(&reg);
        let handle = bank.primary_handle(StrategyId::new(0), InstrumentId::new(0)).unwrap();

        let mut before = bank.read(slot);
        assert!(!before.initialized);
        for (i, bar) in feed.iter().enumerate() {
            bank.begin_event();
            bank.emit_primary(InstrumentId::new(0), *bar).unwrap();
            // A hook sees the value before the bar until the strategy reveals it.
            assert_eq!(bank.read(slot).bits(), before.bits(), "{src}: bar {i}, before reveal");
            bank.reveal(handle, 1);
            let after = bank.read(slot);
            assert_eq!(after.initialized, i >= program.warmup(), "{src}: bar {i} readiness");
            if after.initialized {
                assert_eq!(
                    canon_bits(after.values[0]),
                    canon_bits(want[i]),
                    "{src}: bar {i}: {} vs {}",
                    after.values[0],
                    want[i]
                );
            } else {
                assert!(want[i].is_nan(), "{src}: bar {i} is known before the warm-up ends");
            }
            before = after;
        }
    }
}

#[test]
fn sma_by_expression_is_bit_equal_to_the_legacy_cell() {
    let feed: Vec<Ohlc> = bars(200).into_iter().filter(|b| b.close.is_finite()).collect();
    let id = intern("sma(close, 5)");
    let mut reg = IndicatorRegistry::new();
    let input = IndInput::Primary(InstrumentId::new(0));
    let legacy =
        reg.register(StrategyId::new(0), IndKey::new(input, IndSpec::Sma(5)), None).unwrap();
    let expr = reg
        .register(StrategyId::new(0), IndKey::new(input, IndSpec::expr(id, 0).unwrap()), None)
        .unwrap();
    assert_ne!(legacy.ind, expr.ind);
    let bank = IndicatorBank::new(&reg);
    let handle = bank.primary_handle(StrategyId::new(0), InstrumentId::new(0)).unwrap();
    for (i, bar) in feed.iter().enumerate() {
        bank.begin_event();
        bank.emit_primary(InstrumentId::new(0), *bar).unwrap();
        bank.reveal(handle, 1);
        let (a, b) = (bank.read(legacy.slot), bank.read(expr.slot));
        assert_eq!(a.initialized, b.initialized, "bar {i}");
        assert_eq!(a.values[0].to_bits(), b.values[0].to_bits(), "bar {i}");
    }
}

#[test]
fn two_strategies_with_one_canonical_expression_share_a_cell() {
    let a = intern("sma(close, 6) > ema(close, 3)");
    let b = intern("sma(close,6)>ema(close,3)");
    let mut reg = IndicatorRegistry::new();
    let input = IndInput::Primary(InstrumentId::new(0));
    let first = reg
        .register(StrategyId::new(0), IndKey::new(input, IndSpec::expr(a, 0).unwrap()), None)
        .unwrap();
    let second = reg
        .register(StrategyId::new(1), IndKey::new(input, IndSpec::expr(b, 0).unwrap()), None)
        .unwrap();
    assert!(!first.shared && second.shared);
    assert_eq!(first.ind, second.ind);
    // Another instrument is another stream, so another cell.
    let other = reg
        .register(
            StrategyId::new(1),
            IndKey::new(IndInput::Primary(InstrumentId::new(1)), IndSpec::expr(a, 0).unwrap()),
            None,
        )
        .unwrap();
    assert_ne!(other.ind, first.ind);
    let bank = IndicatorBank::new(&reg);
    for bar in bars(30) {
        bank.begin_event();
        bank.emit_primary(InstrumentId::new(0), bar).unwrap();
    }
    assert_eq!((bank.cells(), bank.registrations()), (2, 3));
    assert_eq!(bank.updates(), 30);
}

#[test]
fn cells_without_expressions_keep_their_numbers() {
    let id = intern("sma(close, 9)");
    let input = IndInput::Primary(InstrumentId::new(0));
    let legacy = [IndSpec::Sma(5), IndSpec::Ema(3), IndSpec::Rsi(14)];

    let mut plain = IndicatorRegistry::new();
    let numbers: Vec<usize> = legacy
        .iter()
        .map(|s| {
            plain.register(StrategyId::new(0), IndKey::new(input, *s), None).unwrap().ind.index()
        })
        .collect();
    assert_eq!(numbers, [0, 1, 2]);

    // An expression registered after them takes the next number and moves none.
    let mut mixed = IndicatorRegistry::new();
    for s in legacy {
        mixed.register(StrategyId::new(0), IndKey::new(input, s), None).unwrap();
    }
    let expr = mixed
        .register(StrategyId::new(0), IndKey::new(input, IndSpec::expr(id, 0).unwrap()), None)
        .unwrap();
    assert_eq!(expr.ind.index(), 3);
    assert_eq!(mixed.keys()[..3], plain.keys()[..]);
    assert_eq!(mixed.keys()[3].kind, IndKind::Expr);
}

#[test]
fn an_output_the_program_lacks_and_an_instrument_set_are_refused() {
    let single = intern("sma(close, 3)");
    assert!(IndSpec::expr(single, 0).is_ok());
    assert!(IndSpec::expr(single, 1).is_err());

    let triple = intern("macd(close)");
    assert_eq!(exprs::info(triple).outputs, 3);
    assert!(IndSpec::expr(triple, 2).is_ok());
    assert!(IndSpec::expr(triple, 3).is_err());

    let mut reg = IndicatorRegistry::new();
    let set = reg.primary_order(&[InstrumentId::new(0), InstrumentId::new(1)]).unwrap();
    let key = IndKey::new(IndInput::AllPrimary(set), IndSpec::expr(single, 0).unwrap());
    let err = reg.register(StrategyId::new(0), key, None).unwrap_err();
    assert!(err.to_string().contains("one instrument"), "{err}");
}

#[test]
fn what_a_spec_says_about_its_program() {
    let by_close = intern("sma(close, 5)");
    let by_volume = intern("sma(volume, 5) > 10");
    assert!(!IndSpec::expr(by_close, 0).unwrap().reads_volume());
    assert!(IndSpec::expr(by_volume, 0).unwrap().reads_volume());
    assert!(!IndSpec::Sma(5).reads_volume());
    // The bars to the first known value: warm-up 4 is a first value on the fifth bar.
    assert_eq!(IndSpec::expr(by_close, 0).unwrap().lookback(), 5);
    assert!(exprs::info(by_volume).boolean && !exprs::info(by_close).boolean);
}

#[test]
fn a_private_core_reads_the_volume_it_is_fed() {
    let id = intern("sma(volume, 3)");
    let mut core = CellCore::for_spec(IndSpec::expr(id, 0).unwrap());
    let bar = |v| Ohlc { open: 1.0, high: 1.0, low: 1.0, close: 1.0, volume: v, ts: 0 };
    assert!(!core.update(bar(30.0)).initialized);
    assert!(!core.update(bar(60.0)).initialized);
    let r = core.update(bar(90.0));
    assert!(r.initialized);
    assert_eq!(r.values[0], 60.0);
}

#[test]
fn the_memory_report_counts_an_expression_cell_by_its_state() {
    let heap = |src: &str| {
        let mut reg = IndicatorRegistry::new();
        let key = IndKey::new(
            IndInput::Primary(InstrumentId::new(0)),
            IndSpec::expr(intern(src), 0).unwrap(),
        );
        reg.register(StrategyId::new(0), key, None).unwrap();
        IndicatorBank::new(&reg).heap_bytes()
    };
    let (short, long) = (heap("sma(close, 10)"), heap("sma(close, 1000)"));
    // The longer window keeps 990 more bars of history in its ring.
    assert!(long >= short + 990 * std::mem::size_of::<f64>(), "{short} vs {long}");
}
