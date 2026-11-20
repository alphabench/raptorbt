//! A book taking what a venue reports.
//!
//! [`PortfolioBook::apply`](super::PortfolioBook::apply) steps a kernel with
//! market data. A venue report arrives with no market data, but it moves the
//! same state: the kernel's cash and margin are the account's, lent to the
//! kernel for the duration of a call. This does the lending for a report, so
//! a fill booked from a venue is accounted exactly as one the matcher makes.

use super::PortfolioBook;
use crate::accounts::AccountMode;
use crate::execution::orders::{ExecutionReport, ReportOutcome};
use crate::portfolio::kernel::EngineEvent;
use crate::portfolio::VenueError;

impl PortfolioBook {
    /// Fold one venue report about an order of `instrument`'s kernel into the
    /// book. `idx` is the instrument's event ordinal the report is booked at
    /// and `ts` its time. Book a report after the event it belongs to has
    /// been applied with [`PortfolioBook::apply`].
    pub fn apply_venue_report(
        &mut self,
        instrument: usize,
        idx: usize,
        ts: i64,
        order_id: u64,
        report: &ExecutionReport,
    ) -> Result<(ReportOutcome, Vec<EngineEvent>), VenueError> {
        let (outcome, mut events) = self.with_lent_account(instrument, Some(ts), |kernel| {
            kernel.apply_venue_report(idx, ts, order_id, report)
        })?;
        // A fill books after the event's own step has sampled equity (the
        // matcher fills inside the step, so its sample includes the fill).
        // Restate that sample, and run the account's checks on it, so a
        // report booked on the event reads as the matcher's fill would have.
        let moved = events
            .iter()
            .any(|e| matches!(e, EngineEvent::Entered { .. } | EngineEvent::Exited { .. }));
        if moved && self.timestamps.last() == Some(ts) && self.grouped.is_none() {
            let equity = self.equity();
            // The earlier sample may have raised the peak above what the
            // restated one reaches.
            self.peak_equity = self.peak_before_sample;
            self.restate_last_sample(equity);
            let halt_idx = self.applied.saturating_sub(1);
            self.account_checks(instrument, ts, idx, equity, halt_idx, &mut events);
        }
        Ok((outcome, events))
    }

    /// Run `f` on an instrument's kernel with the account's cash and the
    /// portfolio-wide limits lent to it, then settle what it moved.
    fn with_lent_account<R>(
        &mut self,
        instrument: usize,
        // The instant of a booking: the sold option lines margin together with their hedge held.
        at: Option<i64>,
        f: impl FnOnce(
            &mut crate::portfolio::kernel::EngineKernel,
        ) -> Result<(R, Vec<EngineEvent>), VenueError>,
    ) -> Result<(R, Vec<EngineEvent>), VenueError> {
        let portfolio_open = self
            .config
            .max_positions
            .map(|_| self.live_open_total() + self.grouped.map_or(0, |g| g.external_open));
        let external_exposure = self.kernels[instrument].tracks_exposure().then(|| {
            let own = self
                .kernels
                .iter()
                .zip(&self.last_close)
                .enumerate()
                .filter(|(i, _)| *i != instrument)
                .map(|(_, (k, close))| close.map(|c| k.gross_notional(c)).unwrap_or(0.0))
                .sum::<f64>();
            match self.grouped {
                Some(g) => own + g.external_exposure,
                None => own,
            }
        });
        let (group_context, hedge_pending) = match at {
            Some(ts) if self.margin_groups => {
                (self.group_context_for(instrument, ts), self.hedge_pending_for(instrument))
            }
            _ => (None, false),
        };
        let kernel = &mut self.kernels[instrument];
        let locked_before = kernel.locked_margin();
        let injected = match self.account.mode() {
            AccountMode::Cash => self.account.balance(),
            AccountMode::Margin { .. } => {
                self.account.balance() - (self.account.locked() - locked_before)
            }
        };
        let injected = kernel.lend_cash(injected);
        kernel.set_external_open_count(portfolio_open);
        kernel.set_external_exposure(external_exposure);
        kernel.set_group_context(group_context);
        kernel.set_hedge_pending(hedge_pending);
        let result = f(kernel);
        let delta_cash = kernel.cash_money() - injected;
        let delta_locked = kernel.locked_margin() - locked_before;
        kernel.set_cash_money(crate::accounts::Money::ZERO);
        kernel.set_external_open_count(None);
        kernel.set_external_exposure(None);
        kernel.set_group_context(None);
        kernel.set_hedge_pending(false);
        self.settle(delta_cash, delta_locked);
        if self.grouped.is_none() {
            self.regroup_option_margin();
        }
        let (value, events) = result?;
        for event in &events {
            if let EngineEvent::Exited { trade, .. } = event {
                self.streaming.update(trade.return_pct / 100.0);
                self.trades.push((**trade).clone());
            }
            if super::is_unfunded_refusal(event) {
                self.unfunded[instrument] += 1;
            }
            self.order_logs[instrument].observe(&self.kernels[instrument], event);
        }
        Ok((value, events))
    }

    /// The book's limits, checked before an order is sent to a venue that
    /// does not apply them; see [`EngineKernel::venue_pretrade`].
    ///
    /// [`EngineKernel::venue_pretrade`]: crate::portfolio::kernel::EngineKernel::venue_pretrade
    #[allow(clippy::too_many_arguments)]
    pub fn venue_pretrade(
        &mut self,
        instrument: usize,
        own: Option<u64>,
        side: crate::execution::orders::OrderSide,
        units: f64,
        reduce_only: bool,
        price: Option<f64>,
        mark: Option<f64>,
    ) -> Result<(), crate::portfolio::risk::RejectReason> {
        // Orders the venue holds for the other instruments add to the gross
        // exposure once filled, as their positions do.
        let others_working = if self.kernels[instrument].tracks_exposure() {
            self.kernels
                .iter()
                .zip(&self.last_close)
                .enumerate()
                .filter(|(i, _)| *i != instrument)
                .map(|(_, (k, close))| k.working_notional(*close))
                .sum::<f64>()
        } else {
            0.0
        };
        let (verdict, _) = self
            .with_lent_account(instrument, None, |k| {
                k.add_external_exposure(others_working);
                Ok((k.venue_pretrade(own, side, units, reduce_only, price, mark), Vec::new()))
            })
            .expect("a check cannot fail");
        verdict
    }

    /// What an order with no size would trade, sized from the book as it
    /// stands; see [`EngineKernel::venue_units`].
    ///
    /// [`EngineKernel::venue_units`]: crate::portfolio::kernel::EngineKernel::venue_units
    pub fn venue_units(
        &mut self,
        instrument: usize,
        bar: &crate::portfolio::kernel::KernelBar,
        side: crate::execution::orders::OrderSide,
        frac: Option<f64>,
        ref_price: Option<f64>,
        limit_priced: bool,
    ) -> Result<f64, &'static str> {
        let (units, _) = self
            .with_lent_account(instrument, None, |k| {
                Ok((k.venue_units(bar, side, frac, ref_price, limit_priced), Vec::new()))
            })
            .expect("a sizing cannot fail");
        units
    }

    /// The last print this book stepped on `instrument`, for sizing an order
    /// made when no event of the instrument is in flight.
    pub fn last_bar(&self, instrument: usize) -> Option<crate::portfolio::kernel::KernelBar> {
        self.last_seen.get(instrument).copied().flatten().map(|(_, bar)| bar)
    }

    /// Record a cancel request on an order of `instrument`'s kernel.
    pub fn request_venue_cancel(
        &mut self,
        instrument: usize,
        order_id: u64,
    ) -> Result<(), VenueError> {
        self.kernels[instrument].request_venue_cancel(order_id)
    }

    /// Record a change request on an order of `instrument`'s kernel.
    pub fn request_venue_replace(
        &mut self,
        instrument: usize,
        order_id: u64,
    ) -> Result<(), VenueError> {
        self.kernels[instrument].request_venue_replace(order_id)
    }
}
