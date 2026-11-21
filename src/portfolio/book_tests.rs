//! Tests for [`PortfolioBook`], which a runtime drives over a shared tape.

use super::*;
use crate::portfolio::session::EventSession;

fn bars(start_ts: i64, closes: &[f64]) -> Vec<KernelBar> {
    closes
        .iter()
        .enumerate()
        .map(|(i, &c)| KernelBar {
            timestamp: start_ts + i as i64 * 10,
            open: c,
            high: c + 1.0,
            low: c - 1.0,
            close: c,
            volume: 1_000.0,
        })
        .collect()
}

fn config() -> BacktestConfig {
    BacktestConfig { fees: 0.0, max_drawdown_pct: Some(15.0), ..BacktestConfig::default() }
}

/// AAA collapses, BBB churns sideways.
fn load(session: &mut EventSession) {
    let a = session.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    let b = session.add_instrument("BBB".into(), Direction::Long, None, None, PositionPolicy::Net);
    session.set_bars(a, bars(0, &[100.0, 90.0, 60.0, 50.0, 50.0, 55.0])).unwrap();
    session.set_bars(b, bars(5, &[50.0, 51.0, 49.0, 50.0, 52.0, 51.0])).unwrap();
    session.seal();
}

/// What each strategy does on an event, as a pure function of the event.
type Intent = fn(&ScheduleEntry) -> StepInput;

/// Goes all-in on AAA's first bar and holds into the collapse.
fn hold_aaa(entry: &ScheduleEntry) -> StepInput {
    StepInput { entry: entry.instrument == 0 && entry.local_idx == 0, ..StepInput::default() }
}

/// Trades BBB in and out; never touches AAA.
fn churn_bbb(entry: &ScheduleEntry) -> StepInput {
    if entry.instrument != 1 {
        return StepInput::default();
    }
    StepInput {
        entry: entry.local_idx & 1 == 0,
        exit: entry.local_idx & 1 == 1,
        ..StepInput::default()
    }
}

fn standalone(intent: Intent) -> (SessionOutcome, usize) {
    let mut session = EventSession::new(config());
    load(&mut session);
    let mut events = 0;
    while let Some(entry) = session.current() {
        session.apply_current(intent(&entry));
        events += 1;
    }
    (session.finish(), events)
}

fn bits(curve: &[f64]) -> Vec<u64> {
    curve.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn two_books_over_one_tape_match_standalone_sessions() {
    let intents: [Intent; 2] = [hold_aaa, churn_bbb];

    // The tape: only its schedule is read; its own book is never stepped.
    let mut tape = EventSession::new(config());
    load(&mut tape);
    let mut books: Vec<PortfolioBook> = intents
        .iter()
        .map(|_| {
            let mut book = PortfolioBook::new(config(), AccountMode::Cash);
            book.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
            book.add_instrument("BBB".into(), Direction::Long, None, None, PositionPolicy::Net);
            book
        })
        .collect();
    for entry in tape.entries() {
        for (book, intent) in books.iter_mut().zip(&intents) {
            book.apply(entry, None, intent(entry));
        }
    }

    let mut halted = Vec::new();
    for (book, intent) in books.into_iter().zip(&intents) {
        let (expected, events) = standalone(*intent);
        assert_eq!(book.applied(), events);
        let got = book.finish();
        assert_eq!(format!("{:?}", got.result.trades), format!("{:?}", expected.result.trades));
        assert_eq!(bits(&got.result.equity_curve), bits(&expected.result.equity_curve));
        assert_eq!(bits(&got.result.returns), bits(&expected.result.returns));
        assert_eq!(got.halted_at, expected.halted_at);
        assert_eq!(got.halted, expected.halted);
        assert!(!got.result.trades.is_empty());
        halted.push(got.halted_at);
    }
    // The books really diverged: the AAA holder tripped the drawdown halt,
    // the BBB churner did not.
    assert!(halted[0].is_some(), "the AAA holder must halt");
    assert_eq!(halted[1], None);
}

/// Drive a one-instrument book over `n` bars, one distinct timestamp each.
fn run_window(window: Option<usize>, n: usize) -> SessionOutcome {
    let mut book = PortfolioBook::new(config(), AccountMode::Cash);
    book.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    book.set_curve_window(window);
    let closes: Vec<f64> = (0..n).map(|i| 100.0 + (i % 7) as f64).collect();
    for (i, bar) in bars(0, &closes).into_iter().enumerate() {
        let entry = ScheduleEntry { instrument: 0, local_idx: i, data: ScheduleData::Bar(bar) };
        book.apply(&entry, None, StepInput { entry: i == 0, ..StepInput::default() });
    }
    book.finish()
}

#[test]
fn a_curve_window_keeps_the_recent_samples_and_the_full_run_keeps_all() {
    let full = run_window(None, 500);
    let windowed = run_window(Some(100), 500);
    assert_eq!(full.result.equity_curve.len(), 500);
    let kept = windowed.result.equity_curve.len();
    assert!((100..200).contains(&kept), "kept {kept}");
    // The window is the tail of the full curve, sample for sample.
    let tail = &full.result.equity_curve[500 - kept..];
    assert_eq!(bits(&windowed.result.equity_curve), bits(tail));
    // Trades are never windowed.
    assert_eq!(full.result.trades.len(), windowed.result.trades.len());
}

#[test]
fn a_windowed_book_holds_flat_memory_and_still_refuses_late_adoption() {
    let mut book = PortfolioBook::new(config(), AccountMode::Cash);
    book.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    book.set_curve_window(Some(64));
    let mut peak = 0;
    for (i, bar) in bars(0, &vec![100.0; 5_000]).into_iter().enumerate() {
        let entry = ScheduleEntry { instrument: 0, local_idx: i, data: ScheduleData::Bar(bar) };
        book.apply(&entry, None, StepInput::default());
        peak = peak.max(book.curve_heap_bytes());
    }
    // At most 2 * window samples of four curves, plus the growth slack of a Vec.
    assert!(peak <= 4 * 8 * 2 * 64 * 2, "curves grew to {peak} bytes");
    assert!(
        book.adopt_position(0, 0, 100.0, 1.0).is_err(),
        "samples were taken, however few remain"
    );
}

/// A run that fills several curve blocks, trading so the equity moves, with a
/// second print at some instants (the instant's sample is replaced), among
/// them the last print of each block.
fn run_long(reserve: bool, window: Option<usize>) -> SessionOutcome {
    use crate::portfolio::curve::BLOCK;
    let n = 3 * BLOCK + 700;
    let mut book = PortfolioBook::new(config(), AccountMode::Cash);
    book.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    if reserve {
        book.reserve_curves(n);
    }
    book.set_curve_window(window);
    let closes: Vec<f64> = (0..n).map(|i| 100.0 + ((i * 7919) % 101) as f64 * 0.05).collect();
    let mut idx = 0;
    for (i, bar) in bars(0, &closes).into_iter().enumerate() {
        let again = KernelBar { close: bar.close + 0.5, ..bar };
        let input = StepInput { entry: i % 40 == 1, exit: i % 40 == 21, ..StepInput::default() };
        let entry = ScheduleEntry { instrument: 0, local_idx: idx, data: ScheduleData::Bar(bar) };
        book.apply(&entry, None, input);
        idx += 1;
        if i % 997 == 0 || i % BLOCK == BLOCK - 1 {
            let entry =
                ScheduleEntry { instrument: 0, local_idx: idx, data: ScheduleData::Bar(again) };
            book.apply(&entry, None, StepInput::default());
            idx += 1;
        }
    }
    book.finish()
}

#[test]
fn a_long_unbounded_run_keeps_the_curves_a_reserved_run_keeps() {
    use crate::portfolio::curve::BLOCK;
    let blocks = run_long(false, None);
    let contiguous = run_long(true, None);
    let (a, b) = (&blocks.result, &contiguous.result);
    assert_eq!(a.equity_curve.len(), 3 * BLOCK + 700, "one sample per distinct instant");
    assert!(a.trades.len() > 100, "the run trades");
    assert_eq!(bits(&a.equity_curve), bits(&b.equity_curve));
    assert_eq!(bits(&a.drawdown_curve), bits(&b.drawdown_curve));
    assert_eq!(bits(&a.returns), bits(&b.returns));
    assert_eq!(format!("{:?}", a.trades), format!("{:?}", b.trades));
    assert_eq!(format!("{:?}", a.metrics), format!("{:?}", b.metrics));
    // A window wider than a block drains whole blocks and part of one, and
    // still keeps the tail of the full curve, sample for sample.
    let windowed = run_long(false, Some(BLOCK + 300));
    let w = &windowed.result;
    let kept = w.equity_curve.len();
    assert!((BLOCK + 300..2 * BLOCK + 600).contains(&kept), "kept {kept}");
    let from = a.equity_curve.len() - kept;
    assert_eq!(bits(&w.equity_curve), bits(&a.equity_curve[from..]));
    assert_eq!(bits(&w.drawdown_curve), bits(&a.drawdown_curve[from..]));
    assert_eq!(bits(&w.returns), bits(&a.returns[from..]));
    assert_eq!(format!("{:?}", w.trades), format!("{:?}", a.trades));
}

#[test]
fn an_instrument_named_in_squareoff_time_by_symbol_squares_off_at_its_own_minute() {
    let mut config = config();
    config.squareoff_time_minutes = Some(23 * 60 + 5);
    config.squareoff_time_by_symbol.insert("AGRI".into(), 16 * 60 + 35);
    let mut book = PortfolioBook::new(config.clone(), AccountMode::Cash);
    book.add_instrument("CRUDE".into(), Direction::Long, None, None, PositionPolicy::Net);
    book.add_instrument("AGRI".into(), Direction::Long, None, None, PositionPolicy::Net);
    let minutes: Vec<Option<u32>> =
        book.kernels.iter().map(|k| k.config.squareoff_time_minutes).collect();
    assert_eq!(minutes, [Some(23 * 60 + 5), Some(16 * 60 + 35)]);
    // without a run-wide time only the named symbol is squared off
    config.squareoff_time_minutes = None;
    assert_eq!(config.squareoff_minutes_for("CRUDE"), None);
    assert_eq!(config.squareoff_minutes_for("AGRI"), Some(16 * 60 + 35));
}
