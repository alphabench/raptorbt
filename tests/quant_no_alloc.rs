//! `Stream::update` allocates nothing per bar once the stream is built.
//!
//! A `#[global_allocator]` is per binary, so this lives in its own test file. It counts on the
//! calling thread only (a const thread-local, which itself never allocates), so the test
//! harness and other threads cannot add to the count. Run with `--no-default-features`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use raptorbt::quant::eval::{Bar, Plan, Stream};
use raptorbt::quant::lang::{compile, Env};
use raptorbt::quant::registry::{all, Family, ParamKind, ParamVal};
use raptorbt::runtime::exprs::{self, ExprCore};

struct Counting;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
    /// Bytes requested and not yet returned, on this thread.
    static LIVE: Cell<isize> = const { Cell::new(0) };
}

fn bump() {
    let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
}

fn track(delta: isize) {
    let _ = LIVE.try_with(|c| c.set(c.get() + delta));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        track(layout.size() as isize);
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump();
        track(layout.size() as isize);
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump();
        track(new_size as isize - layout.size() as isize);
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        track(-(layout.size() as isize));
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static A: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.with(Cell::get)
}

fn live() -> isize {
    LIVE.with(Cell::get)
}

#[test]
fn update_allocates_nothing_after_construction() {
    let sources = [
        "ema(sma(close, 5) - sma(close, 20), 9) > 0 and not (volume < 100)",
        "(close - open) / (high - low) + sma(volume, 7) * 2",
        "close ** 2 - ema(close, 3)",
    ];
    for src in sources {
        let program = compile(src, &Env::default()).unwrap();
        let plan = Plan::new(&program).unwrap();
        let mut stream = Stream::new(&plan).unwrap();

        // The counter sees this thread's allocations; prove it counts before relying on it.
        let before = allocs();
        let probe = vec![0u8; 16];
        assert!(allocs() > before, "the counting allocator is not installed");
        drop(probe);

        let bars: Vec<Bar> = (0..2000)
            .map(|i| {
                let x = i as f64;
                // Every seventh bar carries NaN, every eleventh a zero range (division by 0).
                let close = if i % 7 == 3 { f64::NAN } else { 100.0 + (x * 0.37).sin() * 5.0 };
                let (high, low) =
                    if i % 11 == 5 { (close, close) } else { (close + 1.0, close - 1.0) };
                Bar { open: close - 0.5, high, low, close, volume: 50.0 + (i % 100) as f64 }
            })
            .collect();

        let start = allocs();
        let mut known = 0usize;
        for bar in &bars {
            if !stream.update(bar, &[]).unwrap()[0].is_nan() {
                known += 1;
            }
        }
        let used = allocs() - start;
        assert_eq!(used, 0, "{src}: {used} allocations over {} bars", bars.len());
        assert!(stream.ready() && known > 0, "{src}: the stream never produced a value");
    }
}

/// A call of `def` with its inputs on the default fields and every parameter given (a default
/// where there is one, else a small legal value).
fn call_of(def: &raptorbt::quant::registry::FnDef) -> String {
    let mut args: Vec<String> = def
        .inputs
        .iter()
        .enumerate()
        .map(|(i, inp)| match inp.default_field {
            Some(f) => f.to_string(),
            None if def.name == "if_else" && i == 0 => "close > open".to_string(),
            None => "close".to_string(),
        })
        .collect();
    for p in def.params {
        args.push(match p.default {
            Some(ParamVal::Int(v)) => v.to_string(),
            Some(ParamVal::Float(v)) => format!("{v:?}"),
            Some(ParamVal::Enum(v)) => format!("\"{v}\""),
            None => match p.kind {
                ParamKind::Int { min, max } => 5i64.clamp(min.max(1), max).to_string(),
                ParamKind::Float { min, max } => format!("{:?}", 0.5f64.clamp(min, max)),
                ParamKind::Enum(v) => format!("\"{}\"", v[0]),
            },
        });
    }
    format!("{}({})", def.name, args.join(", "))
}

#[test]
fn a_cell_reports_the_heap_it_holds() {
    let mut checked = 0usize;
    let mut skipped = Vec::new();
    for def in all() {
        // `at` is not run by the evaluator yet.
        if def.family == Family::At {
            continue;
        }
        let src = call_of(def);
        let Ok(program) = compile(&src, &Env::default()) else {
            skipped.push(src);
            continue;
        };
        let Ok(plan) = Plan::new(&program) else {
            skipped.push(src);
            continue;
        };
        let id = exprs::intern(&program, &plan).unwrap();
        let outputs = exprs::info(id).outputs;

        let before = live();
        let core = Box::new(ExprCore::new(id, 0));
        let measured = (live() - before) as usize;
        let claimed = std::mem::size_of::<ExprCore>() + exprs::info(id).state_bytes;
        assert_eq!(core.state_bytes(), exprs::info(id).state_bytes, "{src}");
        assert!(
            measured <= claimed && claimed <= measured + measured / 20 + 64,
            "{src}: the cell holds {measured} bytes, the report says {claimed} ({outputs} outputs)"
        );
        checked += 1;
    }
    assert!(checked >= 150, "only {checked} functions were checked; skipped {skipped:?}");
    eprintln!("checked {checked}; skipped {skipped:?}");
}

#[test]
fn a_composite_cell_reports_the_heap_it_holds() {
    let mut compiled = 0;
    for src in [
        "macd(close).signal",
        "bbands(close, 20, 2.0, 2.0, \"sma\").upper",
        "ema(close, 12) - sma(close, 26) + stoch(14, 3, 3).k",
        "zscore(close, 20) < -2 and adx(14) < 25",
    ] {
        let Ok(program) = compile(src, &Env::default()) else {
            eprintln!("skipped {src}");
            continue;
        };
        compiled += 1;
        let plan = Plan::new(&program).unwrap();
        let id = exprs::intern(&program, &plan).unwrap();
        let before = live();
        let core = Box::new(ExprCore::new(id, 0));
        let measured = (live() - before) as usize;
        let claimed = std::mem::size_of::<ExprCore>() + exprs::info(id).state_bytes;
        assert_eq!(measured, claimed, "{src}");
        drop(core);
    }
    assert!(compiled >= 3, "only {compiled} composite expressions compiled");
}
