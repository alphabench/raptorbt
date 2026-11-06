//! Expression kernels, batch and streaming, per function kind at 1M and 10M bars.
//!
//! Every case is an expression string looked up in the registry at run time, so a function
//! that has not landed yet is skipped with a note and appears on its own once it compiles.
//! To add one, add a line to `CASES`: a label and the spellings to try, first match wins.
//! `P` in a spelling is replaced by each period.
//!
//! Throughput is bars per second. `eval_many` splits the same bar count over 500
//! instruments (so 1M is 2,000 bars each), which is what a universe scan looks like.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use raptorbt::quant::eval::sweep::{SweepLimits, SweepPlan};
use raptorbt::quant::eval::{eval, eval_many, Bar, Inputs, Plan, Stream};
use raptorbt::quant::lang::{compile, Env, Field};

const SIZES: [usize; 2] = [1_000_000, 10_000_000];
const INSTRUMENTS: usize = 500;

/// (label, periods, spellings to try in order). Empty periods means the spelling has no `P`.
const CASES: &[(&str, &[usize], &[&str])] = &[
    ("sma", &[14, 200], &["sma(close, P)"]),
    ("ema", &[14, 200], &["ema(close, P)"]),
    ("rsi", &[14], &["rsi(close, P)"]),
    (
        "rolling_max",
        &[14, 200],
        &["rolling_max(close, P)", "highest(close, P)", "max(close, P)", "ts_max(close, P)"],
    ),
    (
        "rolling_min",
        &[14, 200],
        &["rolling_min(close, P)", "lowest(close, P)", "min(close, P)", "ts_min(close, P)"],
    ),
    // One shared sma(close,20) and ema(close,20) feed every term: the DAG computes them once.
    (
        "dag_shared",
        &[],
        &[
            "(sma(close, 20) - ema(close, 20)) / sma(close, 20) + (sma(close, 20) * 2 - ema(close, 20)) \
             + (ema(close, 20) - sma(close, 20)) / ema(close, 20) * (sma(close, 20) + ema(close, 20))",
        ],
    ),
];

fn series(n: usize, seed: u64) -> Vec<f64> {
    // xorshift walk: deterministic, no dependency, stays positive and finite.
    let mut s = seed | 1;
    let mut p = 100.0f64;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            p = (p + ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5)).clamp(50.0, 150.0);
            p
        })
        .collect()
}

fn plan_for(spellings: &[&str], period: Option<usize>) -> Option<(String, Plan)> {
    for sp in spellings {
        let src = match period {
            Some(p) => sp.replace('P', &p.to_string()),
            None => (*sp).to_string(),
        };
        if let Ok(prog) = compile(&src, &Env::default()) {
            if let Ok(plan) = Plan::new(&prog) {
                return Some((src, plan));
            }
        }
    }
    None
}

fn bench_kernels(c: &mut Criterion) {
    for &n in &SIZES {
        let close = series(n, 7);
        let bars: Vec<Bar> = close
            .iter()
            .map(|&x| Bar { open: x, high: x, low: x, close: x, volume: 1.0 })
            .collect();

        for &(label, periods, spellings) in CASES {
            let variants: Vec<Option<usize>> = if periods.is_empty() {
                vec![None]
            } else {
                periods.iter().map(|&p| Some(p)).collect()
            };
            for period in variants {
                let Some((src, plan)) = plan_for(spellings, period) else {
                    eprintln!("skip {label} (none of {spellings:?} compiles yet)");
                    continue;
                };
                let name = match period {
                    Some(p) => format!("{label}/{p}"),
                    None => label.to_string(),
                };

                let mut g = c.benchmark_group(format!("batch/{name}"));
                g.throughput(Throughput::Elements(n as u64)).sample_size(10);
                g.bench_function(BenchmarkId::from_parameter(n), |b| {
                    let inputs = Inputs::new().field(Field::Close, &close);
                    b.iter(|| black_box(eval(&plan, &inputs).unwrap()));
                });
                g.finish();

                let mut g = c.benchmark_group(format!("stream/{name}"));
                g.throughput(Throughput::Elements(n as u64)).sample_size(10);
                g.bench_function(BenchmarkId::from_parameter(n), |b| {
                    b.iter(|| {
                        let mut s = Stream::new(&plan).unwrap();
                        let mut acc = 0.0;
                        for bar in &bars {
                            acc += s.update(bar, &[]).unwrap()[0];
                        }
                        black_box(acc)
                    });
                });
                g.finish();
                eprintln!("{label} period {period:?}: {src}");
            }
        }

        // eval_many over INSTRUMENTS instruments, n bars in total.
        if let Some((_, plan)) = plan_for(&["sma(close, P)"], Some(20)) {
            let per = n / INSTRUMENTS;
            let cols: Vec<Vec<f64>> =
                (0..INSTRUMENTS).map(|i| series(per, 100 + i as u64)).collect();
            let mut g = c.benchmark_group("eval_many/sma/20");
            g.throughput(Throughput::Elements((per * INSTRUMENTS) as u64)).sample_size(10);
            g.bench_function(BenchmarkId::from_parameter(n), |b| {
                let inputs: Vec<Inputs> =
                    cols.iter().map(|col| Inputs::new().field(Field::Close, col)).collect();
                b.iter(|| black_box(eval_many(&plan, &inputs)));
            });
            g.finish();
        }
    }
}

/// 100 bindings of `sma(close,$n) > sma(close,$m)` over 1M bars: the joint plan, which computes
/// each distinct moving average once, against evaluating every binding on its own.
fn bench_sweep(c: &mut Criterion) {
    const N: usize = 1_000_000;
    const SRC: &str = "sma(close, $n) > sma(close, $m)";
    let close = series(N, 7);
    let inputs = Inputs::new().field(Field::Close, &close);
    let periods: Vec<f64> = (1..=10).map(|k| (k * 10) as f64).collect();
    let programs: Vec<_> = periods
        .iter()
        .flat_map(|n| periods.iter().map(move |m| (*n, *m)))
        .map(|(n, m)| {
            let mut env = Env::default();
            env.params.insert("n".into(), n);
            env.params.insert("m".into(), m + 5.0);
            compile(SRC, &env).unwrap()
        })
        .collect();
    let joint = SweepPlan::new(&programs).unwrap();
    let plans: Vec<Plan> = programs.iter().map(|p| Plan::new(p).unwrap()).collect();
    eprintln!(
        "sweep: {} bindings, {} kernel steps joint, {} independent",
        joint.n_bindings(),
        joint.n_kernel_steps(),
        joint.independent_kernel_steps()
    );

    let mut g = c.benchmark_group("sweep/sma_cross_100");
    g.throughput(Throughput::Elements((N * programs.len()) as u64)).sample_size(10);
    g.bench_function("joint", |b| {
        b.iter(|| black_box(joint.eval(&inputs, &SweepLimits::default()).unwrap()));
    });
    g.bench_function("independent", |b| {
        b.iter(|| black_box(plans.iter().map(|p| eval(p, &inputs).unwrap()).collect::<Vec<_>>()));
    });
    g.finish();
}

criterion_group!(benches, bench_kernels, bench_sweep);
criterion_main!(benches);
