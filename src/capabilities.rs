//! What the engine supports, as data.
//!
//! The manifest an embedding platform reads to know what to offer: order
//! types, times in force, instrument kinds, settlement, margin models, fee
//! rounding, data types and the rest. Each list is made from the engine's own
//! types, never typed out a second time as prose: a name is the variant's
//! own, and a parameter is read off a value of the variant. The tests scan
//! each enum's source and fail when a variant exists that a list here does
//! not carry, so a new order type cannot be added without reaching the
//! manifest.
//!
//! The Python layer adds what only it can see (strategy hooks, indicator
//! signatures, adapters, error codes) in `raptorbt.capabilities()`.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::accounts::MarginModel;
use crate::core::types::{ExitReason, FillTiming};
use crate::execution::costs::rates::Rounding;
use crate::execution::orders::{OrderKind, TimeInForce, TrailOffset};
use crate::instruments::lifecycle::{SettlementPrice, SettlementStyle};
use crate::instruments::{InstrumentKind, OptionRight};

/// The leading identifier of a value's `Debug` form: its variant's name.
fn ident(value: &impl std::fmt::Debug) -> String {
    let text = format!("{value:?}");
    text.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect()
}

/// `CamelCase` as `snake_case`.
fn snake(ident: &str) -> String {
    let mut out = String::new();
    for (i, c) in ident.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.extend(c.to_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

fn named<T: std::fmt::Debug>(values: &[T]) -> Vec<String> {
    values.iter().map(|v| snake(&ident(v))).collect()
}

/// One value of every order kind.
pub(crate) fn order_kind_prototypes() -> Vec<OrderKind> {
    let offset = TrailOffset::Price(0.0);
    vec![
        OrderKind::Market,
        OrderKind::Limit { price: 0.0 },
        OrderKind::StopMarket { trigger: 0.0 },
        OrderKind::StopLimit { trigger: 0.0, price: 0.0 },
        OrderKind::MarketIfTouched { trigger: 0.0 },
        OrderKind::LimitIfTouched { trigger: 0.0, price: 0.0 },
        OrderKind::MarketToLimit,
        OrderKind::TrailingStopMarket { offset },
        OrderKind::TrailingStopLimit { offset, limit_offset: 0.0 },
    ]
}

pub(crate) fn time_in_force_prototypes() -> Vec<TimeInForce> {
    vec![
        TimeInForce::Gtc,
        TimeInForce::Day,
        TimeInForce::Gtd { expire_ns: 0 },
        TimeInForce::Ioc,
        TimeInForce::Fok,
        TimeInForce::AtOpen,
        TimeInForce::AtClose,
    ]
}

pub(crate) fn instrument_kind_prototypes() -> Vec<InstrumentKind> {
    vec![
        InstrumentKind::Cash,
        InstrumentKind::CurrencyPair,
        InstrumentKind::Contract { underlying: None },
        InstrumentKind::Option {
            strike: 0.0,
            right: OptionRight::Call,
            underlying: None,
            binary: false,
        },
        InstrumentKind::Index,
    ]
}

pub(crate) fn margin_model_prototypes() -> Vec<MarginModel> {
    vec![
        MarginModel::NoMargin,
        MarginModel::CashOnly,
        MarginModel::Leverage(BTreeMap::new()),
        MarginModel::PercentOfNotional(0.0),
        MarginModel::OptionSimple { seller_pct: 0.0 },
        MarginModel::Table { per_unit: BTreeMap::new(), default: None },
        MarginModel::ByType(BTreeMap::new()),
        MarginModel::ByProduct {
            intraday: Box::new(MarginModel::NoMargin),
            carry: Box::new(MarginModel::NoMargin),
        },
        MarginModel::DefinedRisk,
        MarginModel::SpreadCredit { seller_pct: 0.0, spread_pct: 0.0 },
        MarginModel::FromSpec { leverage: 1.0 },
    ]
}

pub(crate) fn settlement_style_prototypes() -> Vec<SettlementStyle> {
    vec![SettlementStyle::Cash, SettlementStyle::Physical]
}

pub(crate) fn settlement_price_prototypes() -> Vec<SettlementPrice> {
    vec![
        SettlementPrice::Auto,
        SettlementPrice::LastPrice,
        SettlementPrice::Underlying,
        SettlementPrice::Fixed(0.0),
    ]
}

pub(crate) fn rounding_prototypes() -> Vec<Rounding> {
    vec![Rounding::None, Rounding::PaiseEach, Rounding::ContractNote]
}

pub(crate) fn fill_timing_prototypes() -> Vec<FillTiming> {
    vec![FillTiming::SameBarClose, FillTiming::NextBarOpen, FillTiming::SameBarOpenLookahead]
}

pub(crate) fn exit_reason_prototypes() -> Vec<ExitReason> {
    vec![
        ExitReason::Signal,
        ExitReason::StopLoss,
        ExitReason::TakeProfit,
        ExitReason::TrailingStop,
        ExitReason::EndOfData,
        ExitReason::Order,
        ExitReason::Settlement,
        ExitReason::Delivery,
        ExitReason::CorporateAction,
        ExitReason::Liquidation,
        ExitReason::TimeExit,
        ExitReason::Squareoff,
        ExitReason::WindowEnd,
    ]
}

/// The kinds of market data a run takes, by event payload.
pub(crate) const DATA_TYPES: [&str; 4] = ["Bar", "Quote", "Trade", "Depth"];

/// The price fields an order kind carries, read off its own accessors.
fn order_kind_fields(kind: &OrderKind) -> Vec<&'static str> {
    let mut fields = Vec::new();
    if kind.trigger_price().is_some() {
        fields.push("trigger");
    }
    if kind.limit_price().is_some() {
        fields.push("price");
    }
    if matches!(kind, OrderKind::TrailingStopMarket { .. } | OrderKind::TrailingStopLimit { .. }) {
        fields.push("offset");
    }
    if matches!(kind, OrderKind::TrailingStopLimit { .. }) {
        fields.push("limit_offset");
    }
    fields
}

/// The manifest's engine-side half.
pub fn manifest() -> Value {
    let order_kinds: Vec<Value> = order_kind_prototypes()
        .iter()
        .map(|k| json!({ "name": k.as_str(), "fields": order_kind_fields(k) }))
        .collect();
    let tifs: Vec<&str> = time_in_force_prototypes().iter().map(|t| t.as_str()).collect();
    let data_types: Vec<String> = DATA_TYPES.iter().map(|d| snake(d)).collect();
    json!({
        "order_kinds": order_kinds,
        "time_in_force": tifs,
        "instrument_kinds": named(&instrument_kind_prototypes()),
        "settlement": {
            "styles": named(&settlement_style_prototypes()),
            "prices": named(&settlement_price_prototypes()),
        },
        "margin_models": named(&margin_model_prototypes()),
        "account_modes": ["cash", "margin"],
        "fee_rounding": {
            "modes": rounding_prototypes().iter().map(|r| r.as_str()).collect::<Vec<_>>(),
        },
        "data_types": data_types,
        "fill_timings": named(&fill_timing_prototypes()),
        "exit_reasons": named(&exit_reason_prototypes()),
        "functions": crate::quant::registry::expr_registry_json()["functions"].clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The variant names of `pub enum <name>` in `source`.
    fn variants(source: &str, name: &str) -> Vec<String> {
        let start = source.find(&format!("pub enum {name} ")).expect("enum in source");
        let body = &source[start..];
        let end = body.find("\n}\n").expect("enum end");
        let mut depth = 0i32;
        let mut out = Vec::new();
        for line in body[..end].lines().skip(1) {
            let l = line.trim();
            if depth == 0 && !l.is_empty() && !l.starts_with("//") && !l.starts_with('#') {
                let id: String =
                    l.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
                if id.chars().next().is_some_and(char::is_uppercase) {
                    out.push(id);
                }
            }
            depth +=
                line.matches(['{', '(']).count() as i32 - line.matches(['}', ')']).count() as i32;
        }
        out.sort();
        out
    }

    fn idents<T: std::fmt::Debug>(values: &[T]) -> Vec<String> {
        let mut v: Vec<String> = values.iter().map(ident).collect();
        v.sort();
        v
    }

    macro_rules! covers {
        ($test:ident, $file:literal, $name:literal, $protos:expr) => {
            #[test]
            fn $test() {
                assert_eq!(
                    idents(&$protos),
                    variants(include_str!($file), $name),
                    "{} has a variant the manifest does not list (or the reverse)",
                    $name
                );
            }
        };
    }

    covers!(order_kinds, "execution/orders/order.rs", "OrderKind", order_kind_prototypes());
    covers!(tifs, "execution/orders/order.rs", "TimeInForce", time_in_force_prototypes());
    covers!(
        instrument_kinds,
        "instruments/spec.rs",
        "InstrumentKind",
        instrument_kind_prototypes()
    );
    covers!(margin_models, "accounts/margin_model.rs", "MarginModel", margin_model_prototypes());
    covers!(
        settlement_styles,
        "instruments/lifecycle.rs",
        "SettlementStyle",
        settlement_style_prototypes()
    );
    covers!(
        settlement_prices,
        "instruments/lifecycle.rs",
        "SettlementPrice",
        settlement_price_prototypes()
    );
    covers!(rounding, "execution/costs/rates.rs", "Rounding", rounding_prototypes());
    covers!(fill_timings, "core/types.rs", "FillTiming", fill_timing_prototypes());
    covers!(exit_reasons, "core/types.rs", "ExitReason", exit_reason_prototypes());

    #[test]
    fn data_types_cover_the_event_payloads() {
        let mut listed: Vec<String> = DATA_TYPES.iter().map(|s| s.to_string()).collect();
        listed.sort();
        assert_eq!(listed, variants(include_str!("data/events.rs"), "EventPayload"));
    }

    #[test]
    fn manifest_names_match_the_engines_own() {
        let m = manifest();
        let kinds: Vec<&str> = m["order_kinds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|k| k["name"].as_str().unwrap())
            .collect();
        assert!(kinds.contains(&"trailing_stop_limit"));
        let limit = m["order_kinds"].as_array().unwrap().iter().find(|k| k["name"] == "stop_limit");
        assert_eq!(limit.unwrap()["fields"], json!(["trigger", "price"]));
        assert!(m["margin_models"].as_array().unwrap().contains(&json!("spread_credit")));
    }
}
