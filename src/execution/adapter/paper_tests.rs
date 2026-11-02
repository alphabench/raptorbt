//! The paper venue against the in-process matcher: one script of orders, run
//! both ways over the same bars, must leave the same fills, positions, cash
//! and equity.

use super::super::client_order_id;
use super::*;
use crate::accounts::AccountMode;
use crate::core::types::{BacktestConfig, Direction};
use crate::execution::orders::{OrderKind, OrderSide, OrderStatus, TimeInForce};
use crate::portfolio::kernel::KernelBar;
use crate::portfolio::ledger::PositionPolicy;
use crate::portfolio::session::{ScheduleData, ScheduleEntry, SessionOutcome};
use crate::portfolio::PortfolioBook;

const MIN: i64 = 60_000_000_000;

/// (step, side, units, kind, tif)
type Script = Vec<(usize, OrderSide, f64, OrderKind, TimeInForce)>;

fn bars(n: usize) -> Vec<KernelBar> {
    (0..n)
        .map(|i| {
            let mid = 100.0 + 8.0 * (i as f64 / 4.0).sin() + 0.05 * i as f64;
            KernelBar {
                timestamp: (i as i64 + 1) * MIN,
                open: mid - 0.3,
                high: mid + 1.2,
                low: mid - 1.4,
                close: mid + 0.2,
                volume: 1_000.0,
            }
        })
        .collect()
}

fn book(config: &BacktestConfig, mode: AccountMode, policy: PositionPolicy) -> PortfolioBook {
    let mut book = PortfolioBook::new(config.clone(), mode);
    book.add_instrument("A".to_string(), Direction::Long, None, None, policy);
    book
}

fn entry(i: usize, bar: KernelBar) -> ScheduleEntry {
    ScheduleEntry { instrument: 0, local_idx: i, data: ScheduleData::Bar(bar) }
}

fn run_sim(
    config: &BacktestConfig,
    mode: AccountMode,
    policy: PositionPolicy,
    script: &Script,
    n: usize,
) -> SessionOutcome {
    let mut b = book(config, mode, policy);
    for (i, bar) in bars(n).into_iter().enumerate() {
        for (at, side, units, kind, tif) in script {
            if *at == i {
                b.kernel_mut(0).submit_order_full(
                    *side,
                    QtySpec::Units(*units),
                    *kind,
                    *tif,
                    i,
                    bar.timestamp,
                    format!("o{at}-{units}"),
                    None,
                    None,
                    false,
                    false,
                    None,
                );
            }
        }
        b.apply(&entry(i, bar), None, StepInput::default());
    }
    b.finish()
}

/// The venue way: the strategy's book holds orders the venue works, and
/// learns of every fill from a report.
fn run_venue(
    config: &BacktestConfig,
    mode: AccountMode,
    policy: PositionPolicy,
    script: &Script,
    n: usize,
) -> (SessionOutcome, Vec<Report>) {
    let mut local = book(config, mode, policy);
    let mut venue = PaperVenue::new(vec![(vec!["A".to_string()], book(config, mode, policy))]);
    let mut seq = 0u64;
    let mut ids: HashMap<String, u64> = HashMap::new();
    let mut log = Vec::new();
    for (i, bar) in bars(n).into_iter().enumerate() {
        for (at, side, units, kind, tif) in script {
            if *at == i {
                let id = client_order_id(0, seq);
                seq += 1;
                let order_id = local.kernel_mut(0).submit_venue_order(
                    *side,
                    *units,
                    *kind,
                    *tif,
                    i,
                    bar.timestamp,
                    id.clone(),
                    false,
                    false,
                    None,
                    None,
                    None,
                );
                ids.insert(id.clone(), order_id);
                venue
                    .submit(&OrderRequest {
                        client_order_id: id,
                        symbol: "A".into(),
                        side: *side,
                        quantity: *units,
                        kind: *kind,
                        tif: *tif,
                        strategy_id: 0,
                        account_id: None,
                        reduce_only: false,
                        post_only: false,
                        parent_client_order_id: None,
                        tags: Default::default(),
                        submitted_ns: bar.timestamp,
                        submitted_idx: i,
                        size_frac: None,
                    })
                    .unwrap();
            }
        }
        let e = entry(i, bar);
        let mut out = Vec::new();
        venue.on_market_event(0, &e, None, 0.0, &mut out).unwrap();
        local.apply(&e, None, StepInput::default());
        for (id, report) in out {
            local.apply_venue_report(0, i, bar.timestamp, ids[&id], &report).unwrap();
            log.push((id, report));
        }
    }
    (local.finish(), log)
}

fn assert_same(sim: &SessionOutcome, venue: &SessionOutcome) {
    let s = serde_json::to_string(&sim.result.trades).unwrap();
    let v = serde_json::to_string(&venue.result.trades).unwrap();
    assert_eq!(s, v, "trades differ");
    assert_eq!(sim.result.equity_curve.len(), venue.result.equity_curve.len());
    for (a, b) in sim.result.equity_curve.iter().zip(&venue.result.equity_curve) {
        assert_eq!(a.to_bits(), b.to_bits(), "equity differs: {a} vs {b}");
    }
    assert_eq!(
        sim.result.metrics.total_return_pct.to_bits(),
        venue.result.metrics.total_return_pct.to_bits()
    );
}

fn fees(rate: f64) -> BacktestConfig {
    BacktestConfig { fees: rate, ..BacktestConfig::default() }
}

fn limit(price: f64) -> OrderKind {
    OrderKind::Limit { price }
}

#[test]
fn market_round_trip_matches_the_matcher_with_fees() {
    let script: Script = vec![
        (2, OrderSide::Buy, 100.0, OrderKind::Market, TimeInForce::Gtc),
        (9, OrderSide::Sell, 100.0, OrderKind::Market, TimeInForce::Gtc),
    ];
    let config = fees(0.001);
    let sim = run_sim(&config, AccountMode::Cash, PositionPolicy::Net, &script, 30);
    let (venue, log) = run_venue(&config, AccountMode::Cash, PositionPolicy::Net, &script, 30);
    assert_same(&sim, &venue);
    assert_eq!(sim.result.trades.len(), 1);
    // Every fill the venue reported carried the fee the book charged.
    let fills: Vec<_> = log
        .iter()
        .filter_map(|(_, r)| match r {
            ExecutionReport::Fill { fee, qty, price, .. } => Some((*fee, *qty, *price)),
            _ => None,
        })
        .collect();
    assert_eq!(fills.len(), 2);
    assert!(fills.iter().all(|(fee, ..)| *fee > 0.0));
    let charged: f64 = fills.iter().map(|(fee, ..)| fee).sum();
    assert!((charged - sim.result.trades[0].fees).abs() < 1e-9);
}

#[test]
fn resting_orders_and_slippage_match_the_matcher() {
    let script: Script = vec![
        (1, OrderSide::Buy, 50.0, limit(97.0), TimeInForce::Gtc),
        (4, OrderSide::Buy, 20.0, OrderKind::Market, TimeInForce::Gtc),
        (12, OrderSide::Sell, 70.0, OrderKind::StopMarket { trigger: 96.0 }, TimeInForce::Gtc),
        (18, OrderSide::Buy, 30.0, limit(95.0), TimeInForce::Day),
        (20, OrderSide::Sell, 30.0, OrderKind::Market, TimeInForce::Ioc),
    ];
    let config = BacktestConfig { fees: 0.0005, slippage: 0.0004, ..BacktestConfig::default() };
    let sim = run_sim(&config, AccountMode::Cash, PositionPolicy::Net, &script, 40);
    let (venue, _) = run_venue(&config, AccountMode::Cash, PositionPolicy::Net, &script, 40);
    assert_same(&sim, &venue);
}

#[test]
fn shorts_on_a_margin_account_match_the_matcher() {
    let script: Script = vec![
        (3, OrderSide::Sell, 80.0, OrderKind::Market, TimeInForce::Gtc),
        (10, OrderSide::Buy, 80.0, limit(98.0), TimeInForce::Gtc),
        (22, OrderSide::Sell, 40.0, OrderKind::Market, TimeInForce::Gtc),
        (30, OrderSide::Buy, 40.0, OrderKind::Market, TimeInForce::Gtc),
    ];
    let config = fees(0.0003);
    let mode = AccountMode::Margin { leverage: 3.0 };
    let sim = run_sim(&config, mode, PositionPolicy::Net, &script, 40);
    let (venue, _) = run_venue(&config, mode, PositionPolicy::Net, &script, 40);
    assert_same(&sim, &venue);
}

#[test]
fn an_order_the_book_refuses_is_refused_by_the_venue_too() {
    // More than the account can pay for: the shadow book rejects it, and the
    // strategy's book hears of the rejection, not of a fill.
    let script: Script =
        vec![(2, OrderSide::Buy, 1_000_000.0, OrderKind::Market, TimeInForce::Gtc)];
    let config = fees(0.0);
    let sim = run_sim(&config, AccountMode::Cash, PositionPolicy::Net, &script, 10);
    let (venue, log) = run_venue(&config, AccountMode::Cash, PositionPolicy::Net, &script, 10);
    assert_same(&sim, &venue);
    assert!(log.iter().any(|(_, r)| matches!(r, ExecutionReport::Rejected { .. })));
    assert!(!log.iter().any(|(_, r)| matches!(r, ExecutionReport::Fill { .. })));
    assert_eq!(venue.result.orders.len(), 1);
}

#[test]
fn cancelling_and_changing_an_order_go_through_the_venue() {
    let config = fees(0.0);
    let mut local = book(&config, AccountMode::Cash, PositionPolicy::Net);
    let mut venue = PaperVenue::new(vec![(
        vec!["A".to_string()],
        book(&config, AccountMode::Cash, PositionPolicy::Net),
    )]);
    let id = client_order_id(0, 0);
    let order = local.kernel_mut(0).submit_venue_order(
        OrderSide::Buy,
        10.0,
        limit(50.0),
        TimeInForce::Gtc,
        0,
        MIN,
        id.clone(),
        false,
        false,
        None,
        None,
        None,
    );
    venue
        .submit(&OrderRequest {
            client_order_id: id.clone(),
            symbol: "A".into(),
            side: OrderSide::Buy,
            quantity: 10.0,
            kind: limit(50.0),
            tif: TimeInForce::Gtc,
            strategy_id: 0,
            account_id: None,
            reduce_only: false,
            post_only: false,
            parent_client_order_id: None,
            tags: Default::default(),
            submitted_ns: MIN,
            submitted_idx: 0,
            size_frac: None,
        })
        .unwrap();
    // The same id sent twice is one order.
    let again = OrderRequest { client_order_id: id.clone(), ..sample_request() };
    venue.submit(&again).unwrap();

    let bar = bars(2);
    let feed = |i: usize, local: &mut PortfolioBook, venue: &mut PaperVenue| {
        let e = entry(i, bar[i]);
        let mut out = Vec::new();
        venue.on_market_event(0, &e, None, 0.0, &mut out).unwrap();
        local.apply(&e, None, StepInput::default());
        let mut seen = Vec::new();
        for (rid, report) in out {
            assert_eq!(rid, id);
            local.apply_venue_report(0, i, bar[i].timestamp, order, &report).unwrap();
            seen.push(report);
        }
        seen
    };
    let first = feed(0, &mut local, &mut venue);
    assert!(matches!(first.as_slice(), [ExecutionReport::Accepted { .. }]));

    local.request_venue_replace(0, order).unwrap();
    venue.modify(&id, &OrderChange { limit_price: Some(49.0), ..OrderChange::default() }).unwrap();
    assert_eq!(local.kernel(0).order(order).unwrap().status, OrderStatus::PendingReplace);
    let second = feed(1, &mut local, &mut venue);
    assert!(
        matches!(second.as_slice(), [ExecutionReport::ReplaceAck { new_limit: Some(l), .. }] if *l == 49.0)
    );
    assert_eq!(local.kernel(0).order(order).unwrap().status, OrderStatus::Accepted);

    local.request_venue_cancel(0, order).unwrap();
    assert_eq!(local.kernel(0).order(order).unwrap().status, OrderStatus::PendingCancel);
    venue.cancel(&id).unwrap();
    let mut out = Vec::new();
    venue.on_market_event(0, &entry(2, bar[1]), None, 0.0, &mut out).unwrap();
    assert!(matches!(out.as_slice(), [(_, ExecutionReport::CancelAck { .. })]));
    local.apply_venue_report(0, 2, bar[1].timestamp, order, &out[0].1).unwrap();
    assert_eq!(local.kernel(0).order(order).unwrap().status, OrderStatus::Canceled);
}

fn sample_request() -> OrderRequest {
    OrderRequest {
        client_order_id: String::new(),
        symbol: "A".into(),
        side: OrderSide::Buy,
        quantity: 10.0,
        kind: limit(50.0),
        tif: TimeInForce::Gtc,
        strategy_id: 0,
        account_id: None,
        reduce_only: false,
        post_only: false,
        parent_client_order_id: None,
        tags: Default::default(),
        submitted_ns: MIN,
        submitted_idx: 0,
        size_frac: None,
    }
}

#[test]
fn a_configured_stop_is_left_to_the_runtime_not_acted_on_by_the_shadow() {
    // The runtime turns a stop into an order; the shadow book never closes a
    // position on its own, so the venue has nothing it cannot say in reports.
    let config = BacktestConfig {
        stop: crate::core::types::StopConfig::Fixed { percent: 0.001 },
        ..fees(0.0)
    };
    let mut venue = PaperVenue::new(vec![(
        vec!["A".to_string()],
        book(&config, AccountMode::Cash, PositionPolicy::Net),
    )]);
    venue
        .submit(&OrderRequest {
            client_order_id: "s0-0".into(),
            kind: OrderKind::Market,
            ..sample_request()
        })
        .unwrap();
    let mut fills = 0;
    for (i, bar) in bars(30).into_iter().enumerate() {
        let mut out = Vec::new();
        venue.on_market_event(0, &entry(i, bar), None, 0.0, &mut out).unwrap();
        fills += out.iter().filter(|(_, r)| matches!(r, ExecutionReport::Fill { .. })).count();
    }
    assert_eq!(fills, 1, "only the entry traded");
}
