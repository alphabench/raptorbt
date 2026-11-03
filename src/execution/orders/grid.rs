//! The price and quantity grid an instrument trades on, and the check that an
//! order's terms sit on it.
//!
//! An order whose quantity is not a whole number of lots, or whose limit,
//! trigger or protective price is not a whole number of ticks, is refused at
//! submit, in every mode. The simulation used to floor the quantity and snap
//! the price while a venue was sent the terms as written, so the two ran
//! different orders; refusing leaves one set of terms for both.

/// The lot, size step and tick an instrument trades on. A zero field means
/// the instrument has no such grid.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OrderGrid {
    /// Lot size: a quantity is a whole number of lots.
    pub lot: f64,
    /// Size increment, for instruments that trade in fractions of a unit.
    pub size_step: f64,
    /// Tick size: a price is a whole number of ticks.
    pub tick: f64,
}

/// Which term is off its grid, and the nearest values on it.
#[derive(Debug, Clone, PartialEq)]
pub struct OffGrid {
    pub off_lot: bool,
    /// The argument at fault (`units`, `limit_price`, ...).
    pub parameter: &'static str,
    pub value: f64,
    /// The grid step the value should be a multiple of.
    pub step: f64,
    /// The nearest valid values below and above (a lower one of zero is
    /// omitted: a quantity of zero is not an order).
    pub valid: Vec<f64>,
}

/// Whether `x` is a whole number of `step`s, up to float noise.
pub fn on_grid(x: f64, step: f64) -> bool {
    if step <= 0.0 {
        return true;
    }
    let ratio = x / step;
    let nearest = ratio.round();
    (ratio - nearest).abs() < 1e-9 * nearest.abs().max(1.0)
}

fn neighbours(x: f64, step: f64, keep_zero: bool) -> Vec<f64> {
    let ratio = x / step;
    let clean = |v: f64| (v * 1e9).round() / 1e9;
    let mut out = Vec::with_capacity(2);
    let low = clean(ratio.floor() * step);
    if keep_zero || low > 0.0 {
        out.push(low);
    }
    out.push(clean(ratio.ceil() * step));
    out
}

impl OrderGrid {
    /// Check an order's terms. `units` is an explicit quantity; the prices are
    /// the ones the order carries (limit, trigger, attached stop and target).
    pub fn check(
        &self,
        units: Option<f64>,
        prices: &[(&'static str, Option<f64>)],
    ) -> Result<(), OffGrid> {
        if let Some(u) = units {
            for step in [self.lot, self.size_step] {
                if step > 0.0 && !on_grid(u, step) {
                    return Err(OffGrid {
                        off_lot: true,
                        parameter: "units",
                        value: u,
                        step,
                        valid: neighbours(u, step, false),
                    });
                }
            }
        }
        if self.tick > 0.0 {
            for (name, price) in prices {
                if let Some(p) = price {
                    if p.is_finite() && !on_grid(*p, self.tick) {
                        return Err(OffGrid {
                            off_lot: false,
                            parameter: name,
                            value: *p,
                            step: self.tick,
                            valid: neighbours(*p, self.tick, true),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRID: OrderGrid = OrderGrid { lot: 75.0, size_step: 0.0, tick: 0.05 };

    #[test]
    fn whole_lots_and_ticks_pass() {
        assert!(GRID.check(Some(150.0), &[("limit_price", Some(100.05))]).is_ok());
        assert!(GRID.check(None, &[("limit_price", None)]).is_ok());
    }

    #[test]
    fn off_lot_quantity_names_the_nearest_lots() {
        let e = GRID.check(Some(100.0), &[]).unwrap_err();
        assert!(e.off_lot);
        assert_eq!(e.valid, vec![75.0, 150.0]);
        // Below one lot there is no lower valid quantity.
        assert_eq!(GRID.check(Some(10.0), &[]).unwrap_err().valid, vec![75.0]);
    }

    #[test]
    fn off_tick_price_names_the_nearest_ticks() {
        let e = GRID.check(None, &[("limit_price", Some(100.03))]).unwrap_err();
        assert!(!e.off_lot);
        assert_eq!(e.parameter, "limit_price");
        assert_eq!(e.valid, vec![100.0, 100.05]);
    }

    #[test]
    fn no_grid_accepts_anything() {
        let none = OrderGrid::default();
        assert!(none.check(Some(1.234), &[("limit_price", Some(1.2345))]).is_ok());
    }
}
