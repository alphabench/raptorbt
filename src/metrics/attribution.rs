//! Who made and lost the money: closed trades and filled orders folded by the rule (or
//! strategy) that decided them and by instrument, plus one row per closed trade.
//!
//! The `attribution` / `trades` blocks of the full report, on raptorbt's results (trades
//! and orders rather than a ledger).
//!
//! By design:
//! - an order's origin is its `source` (the rule id or strategy name recorded with the
//!   decision). A trade belongs to the origin of the order that closed it: the filled order
//!   on the trade's instrument, on the closing side, whose last fill is the trade's exit
//!   bar. A trade no order closed (a stop, a target, a settlement the kernel fired) has no
//!   origin and is counted by instrument only.
//! - a trade's fees all go to its closing origin (raptorbt keeps fees per trade, not per
//!   fill); fills and turnover go to each order's own origin.
//! - slippage is inside raptorbt's fill price and is not split out: `slippage` is `None`.
//! - money is `f64`, as in the rest of the report.
//!
//! Nothing here enters a digest.

use std::collections::BTreeMap;

/// One closed trade (or settlement), as the attribution and the trade rows read it.
#[derive(Debug, Clone, PartialEq)]
pub struct AttrTrade {
    pub id: i64,
    pub instrument: String,
    /// 1 long, -1 short.
    pub direction: i32,
    pub size: f64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub opened_at: Option<i64>,
    pub closed_at: i64,
    /// The instrument's own bar or print index of the exit.
    pub exit_idx: Option<i64>,
    /// Realised P&L before fees.
    pub realised: f64,
    pub fees: f64,
    pub entry_fees: f64,
    pub exit_fees: f64,
    /// Fees by component, when the fee model itemises.
    pub fee_parts: BTreeMap<String, f64>,
    pub tag: String,
    pub settlement: bool,
    /// The position was still open when the data ended and the run closed it at the last
    /// mark (`ExitReason::EndOfData`): a round trip, not a closed trade.
    pub open_at_end: bool,
}

/// One order, as the attribution reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct AttrOrder {
    pub instrument: String,
    /// "buy" or "sell" (any case).
    pub side: String,
    /// The rule id or strategy name that decided it; `None` for an order no decision named.
    pub source: Option<String>,
    pub filled_qty: f64,
    pub avg_fill_price: Option<f64>,
    pub multiplier: f64,
    pub fill_slices: u32,
    pub last_fill_idx: Option<i64>,
}

/// A rule run's counters: fires by rule id, no-ops by rule id then reason.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RuleMetrics {
    pub rule_fires: BTreeMap<String, i64>,
    pub noops_by_source: BTreeMap<String, BTreeMap<String, i64>>,
}

/// One attribution row.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AttrRow {
    pub fills: i64,
    pub turnover: f64,
    pub realised: f64,
    pub fees: BTreeMap<String, f64>,
    pub wins: i64,
    pub losses: i64,
    pub gross_profit: f64,
    pub gross_loss: f64,
}

impl AttrRow {
    fn close(&mut self, realised: f64) {
        self.realised += realised;
        if realised > 0.0 {
            self.wins += 1;
            self.gross_profit += realised;
        } else if realised < 0.0 {
            self.losses += 1;
            self.gross_loss -= realised;
        }
    }

    pub fn trades(&self) -> i64 {
        self.wins + self.losses
    }
}

/// A rule's row: its money, how often it fired and why it did nothing.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RuleRow {
    pub row: AttrRow,
    pub fires: i64,
    pub noops: BTreeMap<String, i64>,
}

/// The `attribution` block.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Attribution {
    /// `None` for a run without rule counters (a plain strategy).
    pub by_rule: Option<BTreeMap<String, RuleRow>>,
    pub by_instrument: BTreeMap<String, AttrRow>,
    /// The closing origin of each trade, in the order given.
    pub trade_origins: Vec<Option<String>>,
}

/// The order that closed `t`: on its instrument, on the closing side, filled, with its last
/// fill on the trade's exit index. The first such order wins.
pub fn closing_order<'a>(t: &AttrTrade, orders: &'a [AttrOrder]) -> Option<&'a AttrOrder> {
    let exit_idx = t.exit_idx?;
    let closing = if t.direction >= 0 { "sell" } else { "buy" };
    orders.iter().find(|o| {
        o.instrument == t.instrument
            && o.side.eq_ignore_ascii_case(closing)
            && o.filled_qty > 0.0
            && o.last_fill_idx == Some(exit_idx)
    })
}

/// Fold trades and orders into rows by origin and by instrument.
pub fn attribution(
    trades: &[AttrTrade],
    orders: &[AttrOrder],
    rules: Option<&RuleMetrics>,
) -> Attribution {
    let mut by_origin: BTreeMap<String, AttrRow> = BTreeMap::new();
    let mut by_instrument: BTreeMap<String, AttrRow> = BTreeMap::new();
    for o in orders.iter().filter(|o| o.fill_slices > 0 && o.filled_qty > 0.0) {
        let turnover = o.filled_qty.abs() * o.avg_fill_price.unwrap_or(0.0) * o.multiplier;
        let add = |row: &mut AttrRow| {
            row.fills += i64::from(o.fill_slices);
            row.turnover += turnover;
        };
        add(by_instrument.entry(o.instrument.clone()).or_default());
        if let Some(s) = &o.source {
            add(by_origin.entry(s.clone()).or_default());
        }
    }
    let mut trade_origins = Vec::with_capacity(trades.len());
    for t in trades {
        let origin = if t.settlement {
            Some("lifecycle".to_string())
        } else {
            closing_order(t, orders).and_then(|o| o.source.clone())
        };
        let add = |row: &mut AttrRow| {
            row.close(t.realised);
            for (k, v) in &t.fee_parts {
                if k != "total" {
                    *row.fees.entry(k.clone()).or_insert(0.0) += v;
                }
            }
        };
        add(by_instrument.entry(t.instrument.clone()).or_default());
        if let Some(s) = &origin {
            add(by_origin.entry(s.clone()).or_default());
        }
        trade_origins.push(origin);
    }
    let by_rule = rules.map(|m| {
        let mut ids: Vec<&String> = m.rule_fires.keys().collect();
        ids.extend(m.noops_by_source.keys());
        ids.extend(by_origin.keys());
        ids.sort();
        ids.dedup();
        ids.into_iter()
            .map(|id| {
                let row = RuleRow {
                    row: by_origin.get(id).cloned().unwrap_or_default(),
                    fires: m.rule_fires.get(id).copied().unwrap_or(0),
                    noops: m.noops_by_source.get(id).cloned().unwrap_or_default(),
                };
                (id.clone(), row)
            })
            .collect()
    });
    Attribution { by_rule, by_instrument, trade_origins }
}

/// One closed trade as the report lists it.
#[derive(Debug, Clone, PartialEq)]
pub struct TradeRow {
    pub n: i64,
    pub entry_ts: Option<i64>,
    pub exit_ts: i64,
    pub instrument: String,
    /// "LONG" or "SHORT": the position the trade closed.
    pub side: &'static str,
    pub quantity: f64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub realised: f64,
    pub fees: f64,
    pub entry_fees: f64,
    pub exit_fees: f64,
    pub net: f64,
    pub tag: String,
    pub rule: Option<String>,
    /// "fill" or "settlement".
    pub kind: &'static str,
    pub holding_minutes: Option<f64>,
    /// The run closed it at the end of the data, not a decision: its `net` is the open
    /// position's result at the last mark, and the trade statistics leave it out.
    pub open_at_end: bool,
}

/// The `trades` rows, with each trade's closing origin as its `rule`.
pub fn trade_rows(trades: &[AttrTrade], origins: &[Option<String>]) -> Vec<TradeRow> {
    trades
        .iter()
        .enumerate()
        .map(|(i, t)| TradeRow {
            n: t.id,
            entry_ts: t.opened_at,
            exit_ts: t.closed_at,
            instrument: t.instrument.clone(),
            side: if t.direction >= 0 { "LONG" } else { "SHORT" },
            quantity: t.size.abs(),
            entry_price: t.entry_price,
            exit_price: t.exit_price,
            realised: t.realised,
            fees: t.fees,
            entry_fees: t.entry_fees,
            exit_fees: t.exit_fees,
            net: t.realised - t.fees,
            tag: t.tag.clone(),
            rule: origins.get(i).cloned().flatten(),
            kind: if t.settlement { "settlement" } else { "fill" },
            // Three places: round((closed - opened) / 60e9, 3)
            holding_minutes: t
                .opened_at
                .map(|o| ((t.closed_at - o) as f64 / 60e9 * 1000.0).round() / 1000.0),
            open_at_end: t.open_at_end,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trade(id: i64, inst: &str, dir: i32, realised: f64, exit_idx: i64) -> AttrTrade {
        AttrTrade {
            id,
            instrument: inst.into(),
            direction: dir,
            size: 10.0,
            entry_price: 100.0,
            exit_price: 100.0 + realised / 10.0,
            opened_at: Some(0),
            closed_at: 90_000_000_000,
            exit_idx: Some(exit_idx),
            realised,
            fees: 2.0,
            entry_fees: 1.0,
            exit_fees: 1.0,
            fee_parts: BTreeMap::from([("brokerage".into(), 1.5), ("total".into(), 2.0)]),
            tag: "a".into(),
            settlement: false,
            open_at_end: false,
        }
    }

    fn order(inst: &str, side: &str, source: Option<&str>, last: i64) -> AttrOrder {
        AttrOrder {
            instrument: inst.into(),
            side: side.into(),
            source: source.map(Into::into),
            filled_qty: 10.0,
            avg_fill_price: Some(100.0),
            multiplier: 1.0,
            fill_slices: 1,
            last_fill_idx: Some(last),
        }
    }

    // The closing origin comes from the order matched to the trade's exit.
    #[test]
    fn realised_belongs_to_the_closing_origin() {
        let trades = [trade(1, "X", 1, 50.0, 3), trade(2, "X", -1, -20.0, 5)];
        let orders = [
            order("X", "BUY", Some("entry"), 1),
            order("X", "SELL", Some("exit"), 3),
            order("X", "SELL", Some("entry"), 4),
            order("X", "BUY", None, 5),
        ];
        let m = RuleMetrics {
            rule_fires: BTreeMap::from([("entry".into(), 2), ("exit".into(), 1)]),
            noops_by_source: BTreeMap::from([(
                "quiet".into(),
                BTreeMap::from([("FALSE".into(), 4)]),
            )]),
        };
        let a = attribution(&trades, &orders, Some(&m));
        assert_eq!(a.trade_origins, vec![Some("exit".into()), None]);
        let by_rule = a.by_rule.unwrap();
        assert_eq!(by_rule.keys().collect::<Vec<_>>(), ["entry", "exit", "quiet"]);
        assert_eq!((by_rule["exit"].row.realised, by_rule["exit"].row.wins), (50.0, 1));
        assert_eq!(by_rule["exit"].row.fees["brokerage"], 1.5);
        assert!(!by_rule["exit"].row.fees.contains_key("total"));
        assert_eq!((by_rule["entry"].row.fills, by_rule["entry"].fires), (2, 2));
        assert_eq!(by_rule["quiet"].noops["FALSE"], 4);
        let x = &a.by_instrument["X"];
        assert_eq!((x.fills, x.wins, x.losses, x.realised), (4, 1, 1, 30.0));
        assert_eq!((x.gross_profit, x.gross_loss, x.turnover), (50.0, 20.0, 4000.0));
    }

    #[test]
    fn a_plain_strategy_has_no_rule_rows_and_settlements_are_lifecycle() {
        let mut s = trade(7, "Y", 1, -5.0, 9);
        s.settlement = true;
        let a = attribution(&[s.clone()], &[], None);
        assert!(a.by_rule.is_none());
        assert_eq!(a.trade_origins, vec![Some("lifecycle".into())]);
        let rows = trade_rows(&[s], &a.trade_origins);
        assert_eq!((rows[0].kind, rows[0].side, rows[0].net), ("settlement", "LONG", -7.0));
        assert_eq!(rows[0].holding_minutes, Some(1.5));
        assert_eq!(rows[0].rule.as_deref(), Some("lifecycle"));
    }

    // The line the run closed at the end of the data is marked on its row, so a reader can
    // tell the open position's result from the closed trades' (the report's trade statistics
    // already leave it out).
    #[test]
    fn a_line_closed_at_the_end_of_the_data_is_marked_on_its_row() {
        let closed = trade(1, "X", 1, 50.0, 3);
        let mut open = trade(2, "X", -1, -20.0, 9);
        open.open_at_end = true;
        let trades = [closed, open];
        let a = attribution(&trades, &[], None);
        let rows = trade_rows(&trades, &a.trade_origins);
        assert_eq!(rows.iter().map(|r| r.open_at_end).collect::<Vec<_>>(), [false, true]);
        assert_eq!((rows[1].kind, rows[1].side, rows[1].net), ("fill", "SHORT", -22.0));
    }
}
