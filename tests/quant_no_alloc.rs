//! `Stream::update` allocates nothing per bar once the stream is built.
//!
//! A `#[global_allocator]` is per binary, so this lives in its own test file. It counts on the
//! calling thread only (a const thread-local, which itself never allocates), so the test
//! harness and other threads cannot add to the count. Run with `--no-default-features`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use raptorbt::data::BarStamp;
use raptorbt::quant::eval::{Bar, ClockCtx, Plan, Stream};
use raptorbt::quant::lang::{compile, Env};
use raptorbt::quant::registry::{all, Domain, Family, FnDef, ParamKind, ParamVal};
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

/// An `at(...)` steps its window builder and its body's stream without allocating, on emission
/// bars too (the plain time builder emits by value, so the emission bar counts zero as well),
/// when no session is bound (a bound session's builder allocates per bar, as the runtime hub's
/// does). `at` is not in the registry, so the registry-driven tests do not cover it.
#[test]
fn an_at_expression_allocates_nothing_after_construction() {
    const HOUR: i64 = 3_600_000_000_000;
    let program =
        compile("close > at(\"1d\", sma(close, 3)) and at(\"1h\", high) > 0", &Env::default())
            .unwrap();
    let plan = Plan::new(&program).unwrap();
    let before = live();
    let mut stream = Stream::bound(&plan, ClockCtx::new(0, BarStamp::Close, None)).unwrap();
    let built = (live() - before) as usize;
    // The report covers the heap the stream owns; the struct itself sits where the caller put it.
    assert!(stream.state_bytes() > 0, "an at stream reports no heap");
    assert!(
        built <= stream.state_bytes() + std::mem::size_of::<Stream>(),
        "the stream holds {built} bytes, the report says {}",
        stream.state_bytes()
    );
    let start = allocs();
    let mut known = 0usize;
    for i in 0..2000i64 {
        let x = i as f64;
        let c = 100.0 + (x * 0.37).sin() * 5.0;
        let bar = Bar { open: c - 0.5, high: c + 1.0, low: c - 1.0, close: c, volume: 10.0 };
        // Close-stamped half-hourly bars: 48 a day.
        if !stream.update_at((i + 1) * HOUR / 2, &bar, &[]).unwrap()[0].is_nan() {
            known += 1;
        }
    }
    let used = allocs() - start;
    assert_eq!(used, 0, "{used} allocations over 2000 bars");
    assert!(stream.ready() && known > 0, "the stream never produced a value");
}

/// A call of `def` with its inputs on the default fields and every parameter given: a default
/// where there is one, else `pick` clamped into the parameter's range.
fn call_of(def: &FnDef, pick: i64) -> String {
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
                ParamKind::Int { min, max } => pick.clamp(min.max(1), max).to_string(),
                ParamKind::Float { min, max } => format!("{:?}", 0.5f64.clamp(min, max)),
                ParamKind::Enum(v) => format!("\"{}\"", v[0]),
            },
        });
    }
    format!("{}({})", def.name, args.join(", "))
}

/// The first spelling of `def` that compiles and plans, or why none did.
fn plan_of(def: &FnDef) -> Result<(String, raptorbt::quant::lang::Program, Plan), String> {
    let mut last = String::new();
    for pick in [5, 10, 14, 20, 30] {
        let src = call_of(def, pick);
        match compile(&src, &Env::default()) {
            Err(e) => last = format!("{src}: {e}"),
            Ok(program) => match Plan::new(&program) {
                Ok(plan) => return Ok((src, program, plan)),
                Err(e) => last = format!("{src}: {e}"),
            },
        }
    }
    Err(last)
}

/// Bars with every kind of trouble, spaced so every window has clean bars on both sides: the
/// bad bars come in short bursts 700 bars apart, far wider than any period tried or any
/// warm-up (Hilbert's 63, the longest candles). `unit` keeps prices inside [-1, 1] for the
/// functions whose domain is that range.
fn rough_bars(n: usize, unit: bool) -> Vec<Bar> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            let (mut close, half) = if unit {
                ((x * 0.37).sin() * 0.8, 0.05)
            } else {
                (100.0 + (x * 0.37).sin() * 5.0 + x * 0.01, 1.0)
            };
            let phase = i % 700;
            if (600..603).contains(&phase) {
                close = f64::NAN;
            }
            let (mut high, mut low) = (close + half, close - half);
            if (300..310).contains(&phase) {
                (high, low) = (close, close);
            }
            if phase == 650 {
                high = f64::NAN;
            }
            let volume = match phase {
                400..=402 => f64::NAN,
                450..=452 => 0.0,
                _ => 50.0 + (i % 13) as f64 * 10.0,
            };
            Bar { open: close - half / 2.0, high, low, close, volume }
        })
        .collect()
}

#[test]
fn every_registered_function_updates_without_allocating() {
    let ordinary = rough_bars(4200, false);
    let unit = rough_bars(4200, true);
    let (mut checked, mut failures) = (0usize, Vec::new());
    for def in all() {
        // A cross-sectional function has a value only over a panel and is not a bank cell: it
        // is sealed on the bank in the next slice (3.4), and these checks join it there.
        if def.family == Family::Cross {
            continue;
        }
        let (src, _, plan) = match plan_of(def) {
            Ok(x) => x,
            Err(e) => {
                failures.push(format!("{}: cannot be built: {e}", def.name));
                continue;
            }
        };
        let bars = if def.domain == Domain::Unit { &unit } else { &ordinary };
        // Every output of a multi-output function is a root output of this plan.
        let mut stream = Stream::new(&plan).unwrap();
        let start = allocs();
        let mut known = 0usize;
        for bar in bars {
            known += stream.update(bar, &[]).unwrap().iter().filter(|v| !v.is_nan()).count();
        }
        let used = allocs() - start;
        if used != 0 {
            failures.push(format!(
                "{}: {used} allocations over {} bars ({src})",
                def.name,
                bars.len()
            ));
        }
        if known == 0 {
            failures
                .push(format!("{}: never produced a value on the rough bars ({src})", def.name));
        }
        checked += 1;
    }
    assert!(failures.is_empty(), "{checked} checked; failures:\n{}", failures.join("\n"));
}

#[test]
fn every_registered_function_reports_the_heap_it_holds() {
    let (mut checked, mut failures) = (0usize, Vec::new());
    for def in all() {
        // A cross-sectional function has a value only over a panel and is not a bank cell: it
        // is sealed on the bank in the next slice (3.4), and these checks join it there.
        if def.family == Family::Cross {
            continue;
        }
        let (src, program, plan) = match plan_of(def) {
            Ok(x) => x,
            Err(e) => {
                failures.push(format!("{}: cannot be built: {e}", def.name));
                continue;
            }
        };
        let id = exprs::intern(&program, &plan).unwrap();
        let before = live();
        let core = Box::new(ExprCore::new(id, 0));
        let measured = (live() - before) as usize;
        let claimed = std::mem::size_of::<ExprCore>() + exprs::info(id).state_bytes;
        if core.state_bytes() != exprs::info(id).state_bytes {
            failures.push(format!("{src}: a core and its program disagree"));
        }
        if !(measured <= claimed && claimed <= measured + 64) {
            failures
                .push(format!("{src}: the cell holds {measured} bytes, the report says {claimed}"));
        }
        checked += 1;
    }
    assert!(failures.is_empty(), "{checked} checked; failures:\n{}", failures.join("\n"));
}

#[test]
fn a_composite_cell_reports_the_heap_it_holds() {
    let mut failures = Vec::new();
    for src in [
        "macd(close).macdsignal",
        "bbands(close, 20, 2.0, 2.0, 0).upperband",
        "ema(close, 12) - sma(close, 26) + stoch(high, low, close, 14, 3, 0, 3, 0).slowk",
        "zscore(close, 20) < -2 and adx(14) < 25",
    ] {
        let program = match compile(src, &Env::default()) {
            Ok(p) => p,
            Err(e) => {
                failures.push(format!("{src}: {e}"));
                continue;
            }
        };
        let plan = Plan::new(&program).unwrap();
        let id = exprs::intern(&program, &plan).unwrap();
        let before = live();
        let core = Box::new(ExprCore::new(id, 0));
        let measured = (live() - before) as usize;
        let claimed = std::mem::size_of::<ExprCore>() + exprs::info(id).state_bytes;
        if measured != claimed {
            failures
                .push(format!("{src}: the cell holds {measured} bytes, the report says {claimed}"));
        }
        drop(core);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
