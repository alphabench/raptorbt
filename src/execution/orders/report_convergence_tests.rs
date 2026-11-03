//! Report folding is independent of delivery order for every mix of reports
//! a venue can send, and a contradictory set is refused whichever way it is
//! delivered. Rule table: `docs/architecture/order-lifecycle.md`.

use proptest::prelude::*;

use super::*;
use crate::execution::orders::order::{OrderSide, TimeInForce};

fn order(qty: f64) -> Order {
    let mut o = Order::plain(
        OrderSide::Buy,
        QtySpec::Units(qty),
        OrderKind::Limit { price: 100.0 },
        TimeInForce::Gtc,
    );
    o.id = 7;
    o
}

fn fill(id: &str, qty: f64, price: f64) -> ExecutionReport {
    ExecutionReport::Fill { exec_id: id.into(), qty, price, fee: 0.5, ts_ns: 10 }
}

fn accepted() -> ExecutionReport {
    ExecutionReport::Accepted { venue_order_id: "V".into(), ts_ns: 1 }
}

fn permutations(items: &[ExecutionReport]) -> Vec<Vec<ExecutionReport>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for i in 0..items.len() {
        let mut rest = items.to_vec();
        let head = rest.remove(i);
        for mut tail in permutations(&rest) {
            tail.insert(0, head.clone());
            out.push(tail);
        }
    }
    out
}

fn fold(qty: f64, pending: &[&str], seq: &[ExecutionReport]) -> Result<Order, OrderEventError> {
    let mut o = order(qty);
    for r in seq.iter().filter(|r| matches!(r, ExecutionReport::Accepted { .. })) {
        o.apply_report(r)?;
    }
    for p in pending {
        match *p {
            "cancel" => o.request_cancel().unwrap(),
            _ => o.request_replace().unwrap(),
        }
    }
    for r in seq {
        o.apply_report(r)?;
    }
    Ok(o)
}

fn same(a: &Order, b: &Order, seq: &[ExecutionReport]) {
    assert_eq!(a.status, b.status, "{seq:?}");
    assert_eq!(a.filled_qty.to_bits(), b.filled_qty.to_bits(), "{seq:?}");
    assert_eq!(a.qty, b.qty, "{seq:?}");
    assert_eq!(a.kind, b.kind, "{seq:?}");
    assert_eq!(a.avg_fill_price().map(f64::to_bits), b.avg_fill_price().map(f64::to_bits));
}

/// Every delivery order of `reports` must give the same order.
fn all_orders_converge(qty: f64, pending: &[&str], reports: &[ExecutionReport]) {
    let perms = permutations(reports);
    let first = fold(qty, pending, &perms[0]).unwrap_or_else(|e| panic!("{e}: {:?}", perms[0]));
    for p in &perms[1..] {
        let o = fold(qty, pending, p).unwrap_or_else(|e| panic!("{e}: {p:?}"));
        same(&first, &o, p);
    }
}

#[test]
fn a_duplicate_exec_id_with_other_terms_is_an_error_not_a_repeat() {
    let mut o = order(10.0);
    o.apply_report(&fill("E1", 4.0, 100.0)).unwrap();
    assert_eq!(o.apply_report(&fill("E1", 4.0, 100.0)).unwrap(), ReportOutcome::Duplicate);
    for other in [fill("E1", 5.0, 100.0), fill("E1", 4.0, 101.0)] {
        let err = o.apply_report(&other).unwrap_err();
        assert!(matches!(err, OrderEventError::ConflictingDuplicate { .. }), "{err}");
    }
    assert_eq!(o.filled_qty, 4.0);
}

#[test]
fn a_cancel_and_an_expiry_end_the_order_as_expired_in_either_order() {
    let reports = [
        accepted(),
        ExecutionReport::CancelAck { ts_ns: 5 },
        ExecutionReport::Expired { ts_ns: 6 },
    ];
    all_orders_converge(10.0, &[], &reports);
    let o = fold(10.0, &[], &reports).unwrap();
    assert_eq!(o.status, OrderStatus::Expired);
}

#[test]
fn a_fill_a_bust_and_a_reject_are_refused_in_every_order() {
    let reports = [
        fill("E1", 4.0, 100.0),
        ExecutionReport::Bust { exec_id: "E1".into(), ts_ns: 20 },
        ExecutionReport::Rejected { reason: "x".into(), ts_ns: 30 },
    ];
    for p in permutations(&reports) {
        assert!(fold(10.0, &[], &p).is_err(), "{p:?} was accepted");
    }
}

#[test]
fn a_reject_with_a_cancel_or_expiry_is_a_reject_in_every_order() {
    let reports = [
        ExecutionReport::Rejected { reason: "x".into(), ts_ns: 3 },
        ExecutionReport::CancelAck { ts_ns: 5 },
        ExecutionReport::Expired { ts_ns: 6 },
    ];
    all_orders_converge(10.0, &[], &reports);
    assert_eq!(fold(10.0, &[], &reports).unwrap().status, OrderStatus::Rejected);
}

#[test]
fn a_replace_ack_takes_effect_before_or_after_a_fill_or_a_cancel() {
    let replace = ExecutionReport::ReplaceAck {
        new_qty: Some(20.0),
        new_limit: Some(101.0),
        new_trigger: None,
        ts_ns: 8,
    };
    for rest in [
        vec![accepted(), fill("E1", 10.0, 100.0)],
        vec![accepted(), ExecutionReport::CancelAck { ts_ns: 9 }],
        vec![accepted(), ExecutionReport::Expired { ts_ns: 9 }],
    ] {
        let mut reports = rest;
        reports.push(replace.clone());
        all_orders_converge(10.0, &["replace"], &reports);
        let o = fold(10.0, &["replace"], &reports).unwrap();
        assert_eq!(o.qty, QtySpec::Units(20.0));
    }
}

#[test]
fn corrections_and_busts_converge_with_cancel_and_expiry() {
    let reports = [
        accepted(),
        fill("E1", 4.0, 100.0),
        fill("E2", 3.0, 101.0),
        ExecutionReport::Correction { exec_id: "E1".into(), qty: 2.0, price: 100.5, ts_ns: 40 },
        ExecutionReport::Bust { exec_id: "E2".into(), ts_ns: 50 },
        ExecutionReport::CancelAck { ts_ns: 60 },
        ExecutionReport::Expired { ts_ns: 61 },
    ];
    all_orders_converge(10.0, &["cancel"], &reports);
}

fn arbitrary_mix() -> impl Strategy<Value = Vec<ExecutionReport>> {
    (
        prop::collection::vec((1u32..30, 1u32..500, any::<u8>()), 0..4),
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(|(fills, cancel, expire, replace)| {
            let mut reports = vec![accepted()];
            for (i, (q, p, flag)) in fills.iter().enumerate() {
                let id = format!("E{i}");
                let q = f64::from(*q) * 0.1;
                let p = f64::from(*p) * 0.05;
                reports.push(fill(&id, q, p));
                match flag % 4 {
                    0 => reports.push(ExecutionReport::Correction {
                        exec_id: id,
                        qty: q * 0.5,
                        price: p + 0.1,
                        ts_ns: 100 + i as i64,
                    }),
                    1 => reports.push(ExecutionReport::Bust { exec_id: id, ts_ns: 200 + i as i64 }),
                    _ => {}
                }
            }
            if cancel {
                reports.push(ExecutionReport::CancelAck { ts_ns: 300 });
            }
            if expire {
                reports.push(ExecutionReport::Expired { ts_ns: 301 });
            }
            if replace {
                reports.push(ExecutionReport::ReplaceAck {
                    new_qty: Some(60.0),
                    new_limit: Some(99.0),
                    new_trigger: None,
                    ts_ns: 302,
                });
            }
            reports
        })
}

proptest! {
    #[test]
    fn cancel_expiry_and_replace_mixes_converge(
        reports in arbitrary_mix(),
        seed in any::<u64>(),
        dup in prop::collection::vec(any::<prop::sample::Index>(), 0..4),
    ) {
        let has_replace = reports.iter().any(|r| matches!(r, ExecutionReport::ReplaceAck { .. }));
        let pending: &[&str] = if has_replace { &["replace"] } else { &[] };
        let in_order = fold(50.0, pending, &reports).unwrap();
        let mut noisy = reports.clone();
        for pick in &dup {
            noisy.push(reports[pick.index(reports.len())].clone());
        }
        let mut x = seed | 1;
        for i in (1..noisy.len()).rev() {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            noisy.swap(i, (x >> 33) as usize % (i + 1));
        }
        let shuffled = fold(50.0, pending, &noisy).unwrap();
        same(&in_order, &shuffled, &noisy);
    }
}
