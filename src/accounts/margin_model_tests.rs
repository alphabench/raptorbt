use super::*;
use crate::instruments::InstrumentKind;

fn eq() -> InstrumentSpec {
    InstrumentSpec::new("NSE:X", InstrumentKind::Cash)
}

fn fut() -> InstrumentSpec {
    InstrumentSpec::new("NFO:XFUT", InstrumentKind::Contract { underlying: Some("NSE:X".into()) })
}

fn option_at(id: &str, strike: f64, right: OptionRight, expiry: i64) -> InstrumentSpec {
    let mut s = InstrumentSpec::new(
        id,
        InstrumentKind::Option {
            strike,
            right,
            underlying: Some("NSE:NIFTY50".into()),
            binary: false,
        },
    );
    s.expiration_ns = Some(expiry);
    s.lot_size = 75.0;
    s
}

fn option(id: &str, strike: f64, right: OptionRight) -> InstrumentSpec {
    option_at(id, strike, right, 10)
}

#[test]
fn margin_models_are_named_and_compute_what_they_say() {
    let (e, f) = (eq(), fut());
    let call = option("NFO:NIFTY25000CE", 25000.0, OptionRight::Call);
    let line = LineView::new(&f, 400.0, 1500.0); // notional 600,000
    assert_eq!(MarginModel::NoMargin.required(&line).unwrap(), 0.0);
    assert_eq!(MarginModel::CashOnly.required(&LineView::new(&e, 10.0, 100.0)).unwrap(), 1000.0);
    assert_eq!(
        MarginModel::CashOnly.required(&LineView::new(&e, -10.0, 100.0)).unwrap(),
        f64::INFINITY
    );
    let lev = MarginModel::Leverage(BTreeMap::from([(MarginClass::Contract, 5.0)]));
    assert_eq!(lev.required(&line).unwrap(), 120_000.0);
    assert!(MarginModel::Leverage(BTreeMap::new()).required(&line).is_err());
    assert_eq!(MarginModel::PercentOfNotional(10.0).required(&line).unwrap(), 60_000.0);
    let os = MarginModel::OptionSimple { seller_pct: 15.0 };
    assert_eq!(os.required(&LineView::new(&call, 75.0, 100.0)).unwrap(), 7500.0);
    let short = os.required(&LineView::new(&call, -75.0, 100.0)).unwrap();
    assert_eq!(short, 25000.0 * 75.0 * 15.0 / 100.0 + 7500.0);
    let table = MarginModel::Table {
        per_unit: BTreeMap::from([(f.symbol.clone(), 300.0)]),
        default: Some(Box::new(MarginModel::PercentOfNotional(1.0))),
    };
    assert_eq!(table.required(&line).unwrap(), 120_000.0);
    assert_eq!(table.required(&LineView::new(&e, 10.0, 100.0)).unwrap(), 10.0);
    let bare = MarginModel::Table { per_unit: BTreeMap::new(), default: None };
    assert!(bare.required(&line).is_err());
    let routed = MarginModel::ByType(BTreeMap::from([
        (MarginClass::Contract, MarginModel::PercentOfNotional(10.0)),
        (MarginClass::Cash, MarginModel::CashOnly),
    ]));
    assert_eq!(routed.required(&line).unwrap(), 60_000.0);
    assert_eq!(
        routed.params().to_string(),
        r#"{"CASH":"Margin.cash_only","CONTRACT":"Margin.percent_of_notional"}"#
    );
    assert!(routed.required(&LineView::new(&call, 1.0, 1.0)).is_err());
}

#[test]
fn an_index_has_no_margin_and_names_read_old_class_names() {
    let idx = InstrumentSpec::new("NSE:NIFTY", InstrumentKind::Index);
    let m = MarginModel::Leverage(BTreeMap::from([(MarginClass::Cash, 5.0)]));
    assert!(m.required(&LineView::new(&idx, 1.0, 1.0)).is_err());
    assert_eq!(MarginClass::parse("equity").unwrap(), MarginClass::Cash);
    assert_eq!(MarginClass::parse("FUTURE").unwrap(), MarginClass::Contract);
    assert!(MarginClass::parse("SWAP").is_err());
}

#[test]
fn defined_risk_is_the_worst_expiry_loss_of_the_spread() {
    // short 25000 call at 100, long 25100 call at 60: width 100 - credit 40 = 60/unit, x75
    let short = option("NFO:NIFTY25000CE", 25000.0, OptionRight::Call);
    let long = option("NFO:NIFTY25100CE", 25100.0, OptionRight::Call);
    let legs = [LineView::new(&short, -75.0, 100.0), LineView::new(&long, 75.0, 60.0)];
    assert_eq!(defined_risk(&legs).unwrap(), 60.0 * 75.0);
    assert_eq!(MarginModel::DefinedRisk.required_legs(&legs).unwrap(), 4500.0);
    let e = eq();
    assert!(defined_risk(&[LineView::new(&e, 1.0, 1.0)]).is_err());
}

#[test]
fn defined_risk_has_no_bound_on_an_uncovered_short_call_and_includes_the_underlying_legs() {
    let k = 25000.0;
    let call = option("NFO:NIFTY25000CE", k, OptionRight::Call);
    let put = option("NFO:NIFTY25000PE", k, OptionRight::Put);
    let f = fut();
    // A naked short call loses without bound as the underlying rises.
    assert_eq!(defined_risk(&[LineView::new(&call, -75.0, 100.0)]).unwrap(), f64::INFINITY);
    // A naked short straddle likewise.
    let straddle = [LineView::new(&call, -75.0, 100.0), LineView::new(&put, -75.0, 90.0)];
    assert_eq!(defined_risk(&straddle).unwrap(), f64::INFINITY);
    // A naked short put is bounded by the underlying reaching zero: (K - premium) per unit.
    assert_eq!(defined_risk(&[LineView::new(&put, -75.0, 100.0)]).unwrap(), (k - 100.0) * 75.0);
    // A long option risks its premium.
    assert_eq!(defined_risk(&[LineView::new(&call, 75.0, 100.0)]).unwrap(), 100.0 * 75.0);
    // A covered call: the long future cancels the slope above the strike, so the worst
    // loss is the future falling to zero, less the premium kept.
    let covered = [LineView::new(&call, -75.0, 100.0), LineView::new(&f, 75.0, 25000.0)];
    assert_eq!(defined_risk(&covered).unwrap(), (25000.0 - 100.0) * 75.0);
    // Short futures against a short call are doubly unbounded.
    let both = [LineView::new(&call, -75.0, 100.0), LineView::new(&f, -75.0, 25000.0)];
    assert_eq!(defined_risk(&both).unwrap(), f64::INFINITY);
    // An unbounded group shares an unbounded requirement over every line.
    let shares = MarginModel::DefinedRisk.group_shares(&straddle, 0).unwrap();
    assert!(shares.iter().all(|x| x.is_infinite()));
}

#[test]
fn defined_risk_refuses_legs_of_different_expiries_and_binary_options() {
    let near = option_at("NFO:N25000CE-E1", 25000.0, OptionRight::Call, 10);
    let far = option_at("NFO:N25000CE-E2", 25000.0, OptionRight::Call, 20);
    let calendar = [LineView::new(&near, -75.0, 100.0), LineView::new(&far, 75.0, 300.0)];
    assert!(defined_risk(&calendar).unwrap_err().to_string().contains("one expiry"));
    let mut binary = option("NFO:B", 25000.0, OptionRight::Call);
    binary.kind = InstrumentKind::Option {
        strike: 25000.0,
        right: OptionRight::Call,
        underlying: None,
        binary: true,
    };
    assert!(defined_risk(&[LineView::new(&binary, 1.0, 1.0)])
        .unwrap_err()
        .to_string()
        .contains("binary"));
}

const E1: i64 = 20_000 * DAY_NS + 10 * 3_600_000_000_000; // 15:30 IST on an expiry day
const E2: i64 = E1 + 7 * DAY_NS;

fn spread_credit() -> MarginModel {
    MarginModel::SpreadCredit { seller_pct: 10.0, spread_pct: 4.0 }
}

#[test]
fn a_calendar_is_margined_below_its_legs_and_at_their_sum_on_the_short_expiry_day() {
    let near = option_at("NFO:N25000CE-E1", 25000.0, OptionRight::Call, E1);
    let far = option_at("NFO:N25000CE-E2", 25000.0, OptionRight::Call, E2);
    let legs = [LineView::new(&near, -75.0, 100.0), LineView::new(&far, 75.0, 300.0)];
    let model = spread_credit();
    let sum: f64 = legs.iter().map(|l| model.required(l).unwrap()).sum();
    assert_eq!(sum, 25000.0 * 75.0 * 0.1 + 7500.0 + 22_500.0);
    let day_before = E1 - DAY_NS;
    let calendar = model.required_legs_at(&legs, day_before).unwrap();
    assert_eq!(calendar, sum - 1500.0 * 75.0);
    // the expiry-day rule: from the short's expiry day (IST) the credit is gone
    assert_eq!(model.required_legs_at(&legs, E1 - 6 * 3_600_000_000_000).unwrap(), sum);
    assert_eq!(model.required_legs_at(&legs, E1).unwrap(), sum);
    let shares = model.group_shares(&legs, day_before).unwrap();
    assert_eq!(shares[1], 22_500.0);
    assert_eq!(shares[0], 195_000.0 - 1500.0 * 75.0);
    assert!(model.required_legs(&legs).is_err(), "the plain call without the time is refused");
}

#[test]
fn a_defined_risk_vertical_in_one_expiry_locks_its_max_loss_and_a_naked_write_is_option_simple() {
    let model = spread_credit();
    let simple = MarginModel::OptionSimple { seller_pct: 10.0 };
    let now = E1 - 3 * DAY_NS;
    let short = option_at("NFO:N25000CE-E1", 25000.0, OptionRight::Call, E1);
    let wing = option_at("NFO:N25100CE-E1", 25100.0, OptionRight::Call, E1);
    let legs = [LineView::new(&short, -75.0, 100.0), LineView::new(&wing, 75.0, 60.0)];
    assert_eq!(model.required_legs_at(&legs, now).unwrap(), 60.0 * 75.0);
    assert_eq!(model.required_legs_at(&legs, E1).unwrap(), 60.0 * 75.0);
    let lone = [LineView::new(&short, -75.0, 100.0)];
    assert_eq!(model.required_legs_at(&lone, now).unwrap(), simple.required(&lone[0]).unwrap());
    let naked2 = [LineView::new(&short, -150.0, 100.0), LineView::new(&wing, 75.0, 60.0)];
    let sum: f64 = naked2.iter().map(|l| simple.required(l).unwrap()).sum();
    assert_eq!(model.required_legs_at(&naked2, now).unwrap(), sum);
    assert_eq!(
        model.describe(),
        r#"{"name":"Margin.spread_credit","params":{"seller_pct":10.0,"spread_pct":4.0}}"#
    );
}

#[test]
fn by_product_prices_the_intraday_and_the_delivery_parts_apart() {
    let model = MarginModel::ByProduct {
        intraday: Box::new(MarginModel::ByType(BTreeMap::from([(
            MarginClass::Cash,
            MarginModel::PercentOfNotional(20.0),
        )]))),
        carry: Box::new(MarginModel::ByType(BTreeMap::from([(
            MarginClass::Cash,
            MarginModel::CashOnly,
        )]))),
    };
    let e = eq();
    let line = LineView::new(&e, 100.0, 100.0);
    assert_eq!(model.required_split(&line, 100.0).unwrap(), 2000.0);
    assert_eq!(model.required_split(&line, 0.0).unwrap(), 10_000.0);
    assert_eq!(model.required_split(&line, 40.0).unwrap(), 800.0 + 6000.0);
    assert_eq!(model.required(&line).unwrap(), 10_000.0);
}

// --- FromSpec: the engine's own rule, one path ------------------------------

fn span_option(strike: f64, right: OptionRight) -> InstrumentSpec {
    let mut s = option_at(&format!("NFO:N{strike}"), strike, right, E1);
    s.span_pct = 0.0975;
    s.exposure_pct = 0.02;
    s
}

#[test]
fn from_spec_reads_the_rate_and_the_deposit_from_the_instrument() {
    let m = MarginModel::FromSpec { leverage: 5.0 };
    // no margin_init: one over the leverage; floor of 1.0 on leverage
    let e = eq();
    assert_eq!(m.required(&LineView::new(&e, 10.0, 100.0)).unwrap(), 200.0);
    assert_eq!(
        MarginModel::FromSpec { leverage: 0.5 }.required(&LineView::new(&e, 10.0, 100.0)).unwrap(),
        1000.0
    );
    let mut f = fut();
    f.margin_init = 0.1;
    assert_eq!(m.required(&LineView::new(&f, -4.0, 1500.0)).unwrap(), 600.0);
    // a sold option with a span model: the deposit, not the premium
    let call = span_option(25000.0, OptionRight::Call);
    let deposit = call.short_option_margin_per_contract().unwrap();
    assert_eq!(m.required(&LineView::new(&call, -75.0, 100.0)).unwrap(), deposit * 75.0);
    // a bought option stays on the rate path
    assert_eq!(m.required(&LineView::new(&call, 75.0, 100.0)).unwrap(), 100.0 * 75.0 / 5.0);
    assert_eq!(initial_rate(Some(&f), 5.0), 0.1);
    assert_eq!(initial_rate(None, 4.0), 0.25);
}

#[test]
fn from_spec_groups_sold_legs_like_the_position_group_rule() {
    let m = MarginModel::FromSpec { leverage: 1.0 };
    let short = span_option(25000.0, OptionRight::Put);
    let wing = span_option(24900.0, OptionRight::Put);
    let legs = [LineView::new(&short, -75.0, 100.0), LineView::new(&wing, 75.0, 60.0)];
    // the covered spread locks the structure's worst loss net of premium, far
    // below two naked deposits
    let naked = short.short_option_margin_per_contract().unwrap() * 75.0;
    let total = m.required_legs(&legs).unwrap();
    assert!(total < naked, "{total} vs naked {naked}");
    assert!(m.groups_options());
    let lone = [LineView::new(&short, -75.0, 100.0)];
    assert_eq!(m.required_legs(&lone).unwrap(), naked);
}
