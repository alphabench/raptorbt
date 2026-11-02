//! Account groups against isolated books.

use super::*;
use crate::core::types::{BacktestConfig, Direction};
use crate::portfolio::kernel::StepInput;
use crate::portfolio::ledger::PositionPolicy;
use crate::portfolio::risk::RejectReason;
use crate::portfolio::session::SessionOutcome;
use crate::runtime::cursor::MergeCursor;
use crate::runtime::ids::{InstrumentId, StrategyId};
use crate::runtime::runner::{InstrumentParams, RuntimeBuilder, RuntimeCore};
use crate::runtime::store::{BarColumns, MarketStore};

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

const DAY: i64 = 86_400_000_000_000;

/// `symbols` instruments of `bars` bars each on one shared grid. `drift` is
/// the mean per-bar move, so a negative one makes a falling market.
fn store(seed: u64, symbols: usize, bars: usize, drift: f64) -> MarketStore {
    let mut rng = Lcg(seed);
    let mut store = MarketStore::new();
    for i in 0..symbols {
        let id = store.add_instrument(&format!("S{i}")).unwrap();
        let (mut ts, mut o, mut h, mut l, mut c, mut v) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        let mut price = 100.0 + 5.0 * i as f64;
        for k in 0..bars {
            let open = price;
            price = (price * (1.0 + drift + (rng.unit() - 0.5) * 0.06)).max(1.0);
            ts.push(k as i64 * DAY / 4);
            o.push(open);
            h.push(open.max(price) + rng.unit());
            l.push((open.min(price) - rng.unit()).max(0.5));
            c.push(price);
            v.push(1_000.0);
        }
        store.set_bars(id, BarColumns::from_vecs("t", ts, o, h, l, c, v).unwrap()).unwrap();
    }
    store
}

fn params(symbols: usize) -> Vec<InstrumentParams> {
    (0..symbols).map(|_| InstrumentParams::default()).collect()
}

fn spec(capital: f64, mode: AccountMode) -> GroupSpec {
    GroupSpec {
        initial_capital: capital,
        mode,
        max_drawdown_pct: None,
        max_daily_loss: None,
        max_daily_loss_pct: None,
        max_gross_exposure: None,
        max_positions: None,
    }
}

fn ids(list: &[u32]) -> Vec<InstrumentId> {
    list.iter().map(|&i| InstrumentId::new(i)).collect()
}

/// What a strategy does on an event: a pure function of the strategy's
/// subscription and the event's ordinal.
fn intent(local: usize, idx: usize, salt: u64) -> StepInput {
    let h = (local as u64 * 104_729 + idx as u64 * 1_299_709 + salt * 7_919)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        >> 40;
    StepInput {
        entry: h.is_multiple_of(5),
        exit: h % 7 == 1,
        size_mult: h.is_multiple_of(3).then_some(0.5),
        ..StepInput::default()
    }
}

type Log = Vec<(usize, Vec<EngineEvent>)>;

/// Walk the tape; each subscribed strategy is stepped in registration order.
fn drive(core: &mut RuntimeCore, mut input: impl FnMut(usize, usize, usize) -> StepInput) -> Log {
    let store = core.store().clone();
    let mut cursor = MergeCursor::new(&store);
    let mut log = Vec::new();
    while let Some(event) = core.advance(&mut cursor).unwrap() {
        for sid in core.subscribers(event.instrument).to_vec() {
            let local = core.local_of(sid, event.instrument).unwrap();
            let step = input(sid.index(), local, event.local_idx as usize);
            log.push((sid.index(), core.apply(sid, step).unwrap()));
        }
    }
    log
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn rejected(log: &Log, strategy: usize, reason: RejectReason) -> usize {
    log.iter()
        .filter(|(s, _)| *s == strategy)
        .flat_map(|(_, events)| events)
        .filter(|e| matches!(e, EngineEvent::EntryRejected { reason: r, .. } if *r == reason))
        .count()
}

/// The kinds of account-level event a run produced: margin calls and
/// daily-loss halts.
type Counts = (usize, usize);

fn count(log: &Log) -> Counts {
    let events = log.iter().flat_map(|(_, e)| e);
    (
        events.clone().filter(|e| matches!(e, EngineEvent::MarginCall { .. })).count(),
        events
            .filter(
                |e| matches!(e, EngineEvent::RiskHalted { reason, .. } if reason == "daily_loss"),
            )
            .count(),
    )
}

fn run_pair(
    cfg: BacktestConfig,
    mode: AccountMode,
    limits: GroupSpec,
) -> (SessionOutcome, SessionOutcome, GroupOutcome, Counts, Counts) {
    let make = || store(11, 3, 400, -0.002);
    // Isolated.
    let mut iso =
        RuntimeBuilder::new(make(), params(3), cfg.clone(), mode, PositionPolicy::Net).unwrap();
    let s = iso.add_strategy(&ids(&[0, 1, 2])).unwrap();
    let mut iso = iso.build().unwrap();
    let iso_log = drive(&mut iso, |_, local, idx| intent(local, idx, 0));
    let iso_out = iso.finish(s).unwrap();
    // A group of one.
    let mut grp = RuntimeBuilder::new(make(), params(3), cfg, mode, PositionPolicy::Net).unwrap();
    let g = grp.add_group(limits, 1).unwrap();
    let s = grp.add_strategy_in(&ids(&[0, 1, 2]), Some(g)).unwrap();
    let mut grp = grp.build().unwrap();
    let log = drive(&mut grp, |_, local, idx| intent(local, idx, 0));
    let out = grp.finish(s).unwrap();
    let account = grp.finish_group(g).unwrap();
    (iso_out, out, account, count(&iso_log), count(&log))
}

fn assert_identical(
    a: &crate::portfolio::session::SessionOutcome,
    b: &crate::portfolio::session::SessionOutcome,
) {
    assert_eq!(format!("{:?}", a.result.trades), format!("{:?}", b.result.trades));
    assert_eq!(a.result.orders.len(), b.result.orders.len());
    assert_eq!(format!("{:?}", a.result.metrics), format!("{:?}", b.result.metrics));
    assert_eq!(bits(&a.result.equity_curve), bits(&b.result.equity_curve));
    assert_eq!(bits(&a.result.drawdown_curve), bits(&b.result.drawdown_curve));
    assert_eq!(bits(&a.result.returns), bits(&b.result.returns));
    assert_eq!(a.halted, b.halted);
    assert_eq!(a.halted_at, b.halted_at);
    assert_eq!(a.rejected_entries, b.rejected_entries);
    assert!(!a.result.trades.is_empty(), "the fixture must trade");
}

#[test]
fn a_group_of_one_is_an_isolated_account_in_cash_mode() {
    let cfg = BacktestConfig { fees: 0.001, max_drawdown_pct: Some(25.0), ..Default::default() };
    let limits = GroupSpec { max_drawdown_pct: Some(25.0), ..spec(100_000.0, AccountMode::Cash) };
    let (iso, grp, account, ..) = run_pair(cfg, AccountMode::Cash, limits);
    assert_identical(&iso, &grp);
    assert!(iso.halted, "the drawdown limit should trip");
    assert_eq!(bits(&iso.result.equity_curve), bits(&account.equity_curve));
    assert_eq!(account.halted_at, iso.halted_at);
}

#[test]
fn a_group_of_one_is_an_isolated_account_through_a_margin_call() {
    let mode = AccountMode::Margin { leverage: 20.0 };
    let cfg = BacktestConfig {
        fees: 0.001,
        liquidate_on_margin_call: true,
        max_positions: Some(2),
        ..Default::default()
    };
    let limits = GroupSpec { max_positions: Some(2), ..spec(100_000.0, mode) };
    let (iso, grp, account, iso_counts, counts) = run_pair(cfg, mode, limits);
    assert_identical(&iso, &grp);
    assert_eq!(bits(&iso.result.equity_curve), bits(&account.equity_curve));
    assert_eq!(iso_counts, counts);
    assert!(counts.0 >= 1, "the fixture must reach a margin call");
}

#[test]
fn a_group_of_one_is_an_isolated_account_through_a_daily_loss() {
    let mode = AccountMode::Margin { leverage: 2.0 };
    let cfg = BacktestConfig {
        fees: 0.001,
        max_daily_loss_pct: Some(1.0),
        flatten_on_daily_loss: true,
        ..Default::default()
    };
    let limits = GroupSpec { max_daily_loss_pct: Some(1.0), ..spec(100_000.0, mode) };
    let (iso, grp, account, iso_counts, counts) = run_pair(cfg, mode, limits);
    assert_identical(&iso, &grp);
    assert_eq!(bits(&iso.result.equity_curve), bits(&account.equity_curve));
    assert_eq!(iso_counts, counts);
    assert!(counts.1 >= 1, "the fixture must reach a daily loss");
}

#[test]
fn members_draw_on_one_pool_of_cash() {
    // Both enter with all the cash they can see on the first event. The
    // second finds none, in the same event and at later ones.
    let mut b = RuntimeBuilder::new(
        store(3, 1, 30, 0.0),
        params(1),
        BacktestConfig::default(),
        AccountMode::Cash,
        PositionPolicy::Net,
    )
    .unwrap();
    let g = b.add_group(spec(10_000.0, AccountMode::Cash), 2).unwrap();
    let a = b.add_strategy_in(&ids(&[0]), Some(g)).unwrap();
    let c = b.add_strategy_in(&ids(&[0]), Some(g)).unwrap();
    let mut core = b.build().unwrap();
    let log = drive(&mut core, |_, _, idx| StepInput { entry: idx < 5, ..Default::default() });
    assert!(core.book(a).unwrap().cash() < 10_000.0, "A spent cash");
    assert!(core.book(c).unwrap().free_capital() < 100.0, "B sees what is left");
    assert!(
        rejected(&log, c.index(), RejectReason::ZeroSize)
            + rejected(&log, c.index(), RejectReason::InsufficientCapital)
            >= 5,
        "B is refused on the same event and at later ones"
    );
    let out_a = core.finish(a).unwrap();
    let out_c = core.finish(c).unwrap();
    assert_eq!(out_a.result.trades.len(), 1);
    assert!(out_c.result.trades.is_empty());
    let account = core.finish_group(g).unwrap();
    let pnl: f64 = out_a.result.trades.iter().map(|t| t.pnl).sum();
    assert!((account.balance - (10_000.0 + pnl)).abs() < 1e-6);
}

#[test]
fn a_margin_call_halts_every_member() {
    let mode = AccountMode::Margin { leverage: 20.0 };
    let mut b = RuntimeBuilder::new(
        store(5, 2, 200, -0.01),
        params(2),
        BacktestConfig::default(),
        mode,
        PositionPolicy::Net,
    )
    .unwrap();
    let g = b.add_group(spec(50_000.0, mode), 2).unwrap();
    // The first strategy trades; the second never does.
    let a = b.add_strategy_in(&ids(&[0, 1]), Some(g)).unwrap();
    let c = b.add_strategy_in(&ids(&[0]), Some(g)).unwrap();
    let mut core = b.build().unwrap();
    drive(&mut core, |sid, _, idx| StepInput { entry: sid == 0 && idx == 2, ..Default::default() });
    assert!(core.book(a).unwrap().is_halted());
    assert!(core.book(c).unwrap().is_halted(), "the idle member is halted too");
    let out_a = core.finish(a).unwrap();
    let out_c = core.finish(c).unwrap();
    assert!(out_a.halted && out_c.halted);
    assert_eq!(out_a.halted_at, out_c.halted_at);
    let account = core.finish_group(g).unwrap();
    assert!(account.halted);
    assert_eq!(account.halted_at, out_a.halted_at);
}

#[test]
fn the_drawdown_limit_is_the_accounts() {
    let mut b = RuntimeBuilder::new(
        store(7, 2, 300, -0.006),
        params(2),
        BacktestConfig::default(),
        AccountMode::Cash,
        PositionPolicy::Net,
    )
    .unwrap();
    let limits = GroupSpec { max_drawdown_pct: Some(10.0), ..spec(100_000.0, AccountMode::Cash) };
    let g = b.add_group(limits, 2).unwrap();
    let a = b.add_strategy_in(&ids(&[0]), Some(g)).unwrap();
    let c = b.add_strategy_in(&ids(&[1]), Some(g)).unwrap();
    let mut core = b.build().unwrap();
    // Only the first strategy ever holds anything.
    drive(&mut core, |sid, _, idx| StepInput {
        entry: sid == 0 && idx % 40 == 0,
        exit: sid == 0 && idx % 40 == 30,
        size_mult: Some(0.5),
        ..Default::default()
    });
    assert!(core.book(c).unwrap().is_halted());
    let out_a = core.finish(a).unwrap();
    let out_c = core.finish(c).unwrap();
    assert!(out_a.halted && out_c.halted);
    assert_eq!(out_a.halted_at, out_c.halted_at);
}

#[test]
fn max_positions_counts_the_whole_account() {
    let mut b = RuntimeBuilder::new(
        store(9, 3, 40, 0.0),
        params(3),
        BacktestConfig::default(),
        AccountMode::Cash,
        PositionPolicy::Net,
    )
    .unwrap();
    let limits = GroupSpec { max_positions: Some(2), ..spec(300_000.0, AccountMode::Cash) };
    let g = b.add_group(limits, 3).unwrap();
    let sids: Vec<StrategyId> =
        (0..3).map(|i| b.add_strategy_in(&ids(&[i]), Some(g)).unwrap()).collect();
    let mut core = b.build().unwrap();
    let log = drive(&mut core, |_, _, idx| StepInput {
        entry: idx == 1,
        size_mult: Some(0.2),
        ..Default::default()
    });
    assert_eq!(rejected(&log, 0, RejectReason::MaxPositions), 0);
    assert_eq!(rejected(&log, 1, RejectReason::MaxPositions), 0);
    assert!(rejected(&log, 2, RejectReason::MaxPositions) >= 1, "the third is over the limit");
    let trades: Vec<usize> =
        sids.iter().map(|&s| core.finish(s).unwrap().result.trades.len()).collect();
    assert_eq!(trades, vec![1, 1, 0]);
}

#[test]
fn halting_an_account_stops_every_member() {
    let mut b = RuntimeBuilder::new(
        store(13, 2, 60, 0.0),
        params(2),
        BacktestConfig::default(),
        AccountMode::Cash,
        PositionPolicy::Net,
    )
    .unwrap();
    let g = b.add_group(spec(100_000.0, AccountMode::Cash), 2).unwrap();
    let a = b.add_strategy_in(&ids(&[0]), Some(g)).unwrap();
    let c = b.add_strategy_in(&ids(&[1]), Some(g)).unwrap();
    let other = b.add_strategy(&ids(&[0])).unwrap();
    let mut core = b.build().unwrap();
    let store = core.store().clone();
    let mut cursor = MergeCursor::new(&store);
    let mut n = 0;
    while let Some(event) = core.advance(&mut cursor).unwrap() {
        if n == 20 {
            core.halt_group(g, "manual", true).unwrap();
        }
        n += 1;
        for sid in core.subscribers(event.instrument).to_vec() {
            let idx = event.local_idx as usize;
            core.apply(sid, StepInput { entry: idx.is_multiple_of(10), ..Default::default() })
                .unwrap();
        }
    }
    let (a, c, other) =
        (core.finish(a).unwrap(), core.finish(c).unwrap(), core.finish(other).unwrap());
    assert!(a.halted && c.halted && !other.halted, "an isolated strategy is not halted");
    assert_eq!(a.halted_at, c.halted_at);
    // Nothing opened after the halt, and what was open was flattened.
    let account = core.finish_group(g).unwrap();
    assert!(account.halted);
    let last_entry = a.result.trades.iter().chain(&c.result.trades).map(|t| t.entry_idx).max();
    assert!(last_entry.is_none_or(|i| i <= 20));
}

#[test]
fn a_property_the_balance_is_the_capital_plus_the_members_pnl() {
    for seed in 0..40u64 {
        let mut rng = Lcg(seed + 1);
        let symbols = 1 + (rng.next() % 3) as usize;
        let margin = rng.next().is_multiple_of(2);
        let mode = if margin {
            AccountMode::Margin { leverage: 1.0 + (rng.next() % 4) as f64 }
        } else {
            AccountMode::Cash
        };
        let cfg = BacktestConfig { fees: 0.001, ..Default::default() };
        let mut b = RuntimeBuilder::new(
            store(seed, symbols, 120, (rng.unit() - 0.5) * 0.004),
            params(symbols),
            cfg,
            mode,
            PositionPolicy::Net,
        )
        .unwrap();
        let capital = 20_000.0 + 10_000.0 * (rng.next() % 5) as f64;
        let groups = 1 + (rng.next() % 2) as usize;
        let mut members: Vec<Vec<StrategyId>> = Vec::new();
        for _ in 0..groups {
            let n = 1 + (rng.next() % 3) as usize;
            let g = b.add_group(spec(capital, mode), n).unwrap();
            members.push(
                (0..n)
                    .map(|_| {
                        let first = (rng.next() % symbols as u64) as u32;
                        b.add_strategy_in(&ids(&[first]), Some(g)).unwrap()
                    })
                    .collect(),
            );
        }
        let mut core = b.build().unwrap();
        let mut min_free = f64::MAX;
        let store = core.store().clone();
        let mut cursor = MergeCursor::new(&store);
        while let Some(event) = core.advance(&mut cursor).unwrap() {
            for sid in core.subscribers(event.instrument).to_vec() {
                let local = core.local_of(sid, event.instrument).unwrap();
                let step = intent(local, event.local_idx as usize, sid.index() as u64 + seed);
                core.apply(sid, step).unwrap();
                min_free = min_free.min(core.book(sid).unwrap().cash());
            }
        }
        if !margin {
            assert!(min_free >= -1e-6, "seed {seed}: cash went negative ({min_free})");
        }
        for (g, group) in members.iter().enumerate() {
            let mut pnl = 0.0;
            for &s in group {
                pnl += core.finish(s).unwrap().result.trades.iter().map(|t| t.pnl).sum::<f64>();
            }
            let account = core.finish_group(g as u32).unwrap();
            // Trade P&L is net of fees, so the balance needs nothing more.
            assert!(
                (account.balance - (capital + pnl)).abs() <= 1e-6 * capital,
                "seed {seed} group {g}: balance {} vs {}",
                account.balance,
                capital + pnl
            );
        }
    }
}

#[test]
fn a_direction_is_not_changed_by_grouping() {
    // A short instrument in a group sells, as it does alone.
    let mut b = RuntimeBuilder::new(
        store(2, 1, 50, 0.0),
        vec![InstrumentParams { direction: Direction::Short, ..Default::default() }],
        BacktestConfig::default(),
        AccountMode::Margin { leverage: 2.0 },
        PositionPolicy::Net,
    )
    .unwrap();
    let mode = AccountMode::Margin { leverage: 2.0 };
    let g = b.add_group(spec(100_000.0, mode), 1).unwrap();
    let s = b.add_strategy_in(&ids(&[0]), Some(g)).unwrap();
    let mut core = b.build().unwrap();
    drive(&mut core, |_, _, idx| StepInput {
        entry: idx == 3,
        exit: idx == 20,
        ..Default::default()
    });
    let out = core.finish(s).unwrap();
    assert_eq!(out.result.trades[0].direction, Direction::Short);
}
