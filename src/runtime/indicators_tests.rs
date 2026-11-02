//! Key behaviour: what shares a cell and what does not.

use crate::runtime::ids::{AggId, InstrumentId, StrategyId};
use crate::runtime::indicators::{
    IndInput, IndKey, IndSpec, IndicatorRegistry, InputField, LocalStreamId,
};

fn inst(i: u32) -> InstrumentId {
    InstrumentId::new(i)
}

fn prim(i: u32, spec: IndSpec) -> IndKey {
    IndKey::new(IndInput::Primary(inst(i)), spec)
}

#[test]
fn same_full_key_shares_and_cells_follow_first_registration() {
    let mut r = IndicatorRegistry::new();
    let a = r.register(StrategyId::new(0), prim(3, IndSpec::Sma(10)), None).unwrap();
    let b = r.register(StrategyId::new(1), prim(1, IndSpec::Sma(10)), None).unwrap();
    let c = r.register(StrategyId::new(2), prim(3, IndSpec::Sma(10)), None).unwrap();
    assert!(!a.shared && !b.shared && c.shared);
    assert_eq!((a.ind.index(), b.ind.index(), c.ind.index()), (0, 1, 0));
    assert_eq!((a.slot.index(), b.slot.index(), c.slot.index()), (0, 1, 2));
    assert_eq!((r.cells(), r.registrations()), (2, 3));
}

#[test]
fn params_input_field_and_kind_never_share() {
    let mut r = IndicatorRegistry::new();
    let s = StrategyId::new(0);
    let order = r.primary_order(&[inst(0), inst(1)]).unwrap();
    let keys = [
        prim(0, IndSpec::Sma(10)),
        prim(0, IndSpec::Sma(11)),
        prim(0, IndSpec::Ema(10)),
        prim(0, IndSpec::WilderMa(10)),
        prim(1, IndSpec::Sma(10)),
        prim(0, IndSpec::Sma(10)).with_field(InputField::High),
        IndKey::new(IndInput::AllPrimary(order), IndSpec::Sma(10)),
        prim(0, IndSpec::Bollinger { period: 20, k: 2.0 }),
        prim(0, IndSpec::Bollinger { period: 20, k: 2.5 }),
        prim(0, IndSpec::Macd { fast: 12, slow: 26, signal: 9 }),
        prim(0, IndSpec::Macd { fast: 12, slow: 26, signal: 10 }),
        prim(0, IndSpec::Atr(14)),
        prim(0, IndSpec::Donchian(14)),
    ];
    for k in keys {
        assert!(!r.register(s, k, None).unwrap().shared, "{k:?}");
    }
    let comp = |agg: u32| IndKey::new(IndInput::Composite(AggId::new(agg)), IndSpec::Sma(10));
    assert!(!r.register(s, comp(0), Some(LocalStreamId(0))).unwrap().shared);
    assert!(!r.register(s, comp(1), Some(LocalStreamId(1))).unwrap().shared);
    assert!(r.register(s, comp(1), Some(LocalStreamId(2))).unwrap().shared);
    assert_eq!(r.cells(), keys.len() + 2);
    assert_eq!(r.registrations(), keys.len() + 3);
}

#[test]
fn equal_behaviour_canonicalises_to_one_key() {
    // The cores clamp a zero period to one.
    assert_eq!(prim(0, IndSpec::Sma(0)), prim(0, IndSpec::Sma(1)));
    assert_eq!(
        prim(0, IndSpec::Bollinger { period: 5, k: 0.0 }),
        prim(0, IndSpec::Bollinger { period: 5, k: -0.0 })
    );
    assert_eq!(
        prim(0, IndSpec::Bollinger { period: 5, k: f64::NAN }),
        prim(0, IndSpec::Bollinger { period: 5, k: -f64::NAN })
    );
}

#[test]
fn primary_orders_intern_by_set() {
    let mut r = IndicatorRegistry::new();
    let a = r.primary_order(&[inst(2), inst(0), inst(2)]).unwrap();
    let b = r.primary_order(&[inst(0), inst(2)]).unwrap();
    let c = r.primary_order(&[inst(0)]).unwrap();
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_eq!(r.order_instruments(a), Some(&[inst(0), inst(2)][..]));
    assert!(r.primary_order(&[]).is_err());
}

#[test]
fn malformed_registrations_are_rejected() {
    let mut r = IndicatorRegistry::new();
    let s = StrategyId::new(0);
    // A scalar kind cannot read the whole bar and a bar kind cannot read one field.
    assert!(r.register(s, prim(0, IndSpec::Sma(3)).with_field(InputField::Bar), None).is_err());
    assert!(r.register(s, prim(0, IndSpec::Atr(3)).with_field(InputField::Close), None).is_err());
    // Composite needs its stream; others must not name one.
    let comp = IndKey::new(IndInput::Composite(AggId::new(0)), IndSpec::Sma(3));
    assert!(r.register(s, comp, None).is_err());
    assert!(r.register(s, prim(0, IndSpec::Sma(3)), Some(LocalStreamId(0))).is_err());
    // An order that was never interned.
    let mut other = IndicatorRegistry::new();
    let foreign = other.primary_order(&[inst(0)]).unwrap();
    let all = IndKey::new(IndInput::AllPrimary(foreign), IndSpec::Sma(3));
    assert!(r.register(s, all, None).is_err());
    assert_eq!((r.cells(), r.registrations()), (0, 0));
}
