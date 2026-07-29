"""Behavior tests for the portfolio math surface (0.7.0).

These pin the Python-visible contract of estimate_covariance /
optimize_portfolio / factor panels / risk contributions / rebalance
simulation against the wheel. Golden values are closed-form or hand-computed;
error-surface tests assert refusal (ValueError), because for a financial
library a plausible wrong number is strictly worse than an exception.
"""

import numpy as np
import pytest

import raptorbt as r

IDS4 = ["A", "B", "C", "D"]


def _returns_panel(n_obs=400, n_assets=4, seed=11):
    """Heterogeneous-correlation synthetic returns (see Rust test note:
    a uniform common-factor loading makes the constant-correlation target an
    exact fit and legitimately drives shrinkage to 1)."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, size=(n_obs, 1))
    loadings = np.linspace(0.2, 1.0, n_assets)[None, :]
    return common * loadings + rng.normal(0, 0.012, size=(n_obs, n_assets))


@pytest.fixture(scope="module")
def model():
    return r.estimate_covariance(_returns_panel(), IDS4, 252.0)


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------


class TestEstimateCovariance:
    def test_carries_context_and_valid_intensity(self, model):
        assert model.asset_ids == IDS4
        assert model.periods_per_year == 252.0
        assert model.n_obs == 400
        assert 0.0 <= model.shrinkage_intensity <= 1.0

    def test_cov_is_symmetric_psd(self, model):
        cov = model.cov()
        assert cov.shape == (4, 4)
        assert np.allclose(cov, cov.T)
        eigvals = np.linalg.eigvalsh(cov)
        assert eigvals.min() > 0

    def test_diagonal_equals_sample_variance(self):
        """Target and sample share the diagonal, so shrinkage preserves it."""
        panel = _returns_panel()
        m = r.estimate_covariance(panel, IDS4, 252.0)
        sample_var = panel.var(axis=0)  # ddof=0 matches the LW 1/T convention
        assert np.allclose(np.diag(m.cov()), sample_var, rtol=1e-12)

    def test_shrinks_toward_constant_correlation_not_identity(self):
        """Off-diagonals move toward r_bar*si*sj -- never toward zero the way
        an identity target would."""
        panel = _returns_panel()
        m = r.estimate_covariance(panel, IDS4, 252.0)
        s = np.cov(panel.T, ddof=0)
        d = np.sqrt(np.diag(s))
        corr = s / np.outer(d, d)
        r_bar = corr[np.triu_indices(4, k=1)].mean()
        f = r_bar * np.outer(d, d)
        np.fill_diagonal(f, np.diag(s))
        delta = m.shrinkage_intensity
        assert np.allclose(m.cov(), delta * f + (1 - delta) * s, rtol=1e-10)

    def test_refuses_nan_and_shape_errors(self):
        panel = _returns_panel()
        bad = panel.copy()
        bad[3, 1] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            r.estimate_covariance(bad, IDS4, 252.0)
        with pytest.raises(ValueError):
            r.estimate_covariance(panel, IDS4[:3], 252.0)
        with pytest.raises(ValueError, match="periods_per_year"):
            r.estimate_covariance(panel, IDS4, 0.0)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def _cfg(**overrides):
    base = dict(
        risk_aversion=1.0,
        turnover_penalty=0.0,
        position_cap=1.0,
        sector_ids=[0, 0, 0, 0],
        sector_caps=[1.0],
        cash_max=0.0,
    )
    base.update(overrides)
    return r.PyOptimizerConfig(**base)


class TestOptimizePortfolio:
    def test_two_asset_inverse_variance_golden(self):
        """Zero alpha, uncorrelated, fully invested: minimum variance gives
        w0 = v1/(v0+v1) in closed form."""
        rets = np.zeros((300, 2))
        rng = np.random.default_rng(3)
        rets[:, 0] = rng.normal(0, 0.02, 300)
        rets[:, 1] = rng.normal(0, 0.04, 300)
        m = r.estimate_covariance(rets, ["X", "Y"], 252.0)
        cov = m.cov()  # the SHRUNK matrix -- what the optimizer actually sees
        expect_w0 = (cov[1, 1] - cov[0, 1]) / (cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
        res = r.optimize_portfolio(
            m,
            np.zeros(2),
            np.array([0.5, 0.5]),
            ["X", "Y"],
            _cfg(risk_aversion=10.0, sector_ids=[0, 0], sector_caps=[1.0]),
        )
        # Closed form for min-var with sum(w)=1 on two assets.
        w = res.weights()
        assert abs(w[0] - expect_w0) < 1e-5
        assert abs(w.sum() - 1.0) < 1e-6

    def test_huge_turnover_penalty_freezes_book(self, model):
        res = r.optimize_portfolio(
            model,
            np.array([0.5, -0.5, 0.2, 0.0]),
            np.array([0.4, 0.3, 0.2, 0.1]),
            IDS4,
            _cfg(turnover_penalty=1e6),
        )
        assert res.turnover < 1e-6
        assert np.allclose(res.weights(), [0.4, 0.3, 0.2, 0.1], atol=1e-6)

    def test_asset_order_mismatch_is_an_error(self, model):
        with pytest.raises(ValueError, match="asset ids mismatch"):
            r.optimize_portfolio(
                model,
                np.zeros(4),
                np.full(4, 0.25),
                ["D", "C", "B", "A"],
                _cfg(),
            )

    def test_infeasible_caps_refused(self, model):
        with pytest.raises(ValueError, match="[Ii]nfeasible"):
            r.optimize_portfolio(
                model,
                np.zeros(4),
                np.full(4, 0.25),
                IDS4,
                _cfg(position_cap=0.2),  # 4 x 0.2 = 0.8 < 1 required
            )

    def test_batch_equals_serial_and_preserves_order(self, model):
        cfg = _cfg(turnover_penalty=0.01)
        alphas = [np.array([0.1, 0.0, -0.1, 0.05]), np.array([-0.2, 0.1, 0.0, 0.0])]
        w_cur = np.full(4, 0.25)
        serial = [
            r.optimize_portfolio(model, a, w_cur, IDS4, cfg) for a in alphas
        ]
        batch = r.batch_optimize_portfolios(
            model,
            [r.PyOptimizeItem(f"u{i}", a, w_cur) for i, a in enumerate(alphas)],
            cfg,
        )
        assert [item_id for item_id, _ in batch] == ["u0", "u1"]
        for (_, b), s in zip(batch, serial):
            assert np.array_equal(b.weights(), s.weights())
            assert b.objective == s.objective

    def test_batch_names_the_failing_item(self, model):
        with pytest.raises(ValueError, match="item 'bad'"):
            r.batch_optimize_portfolios(
                model,
                [
                    r.PyOptimizeItem("ok", np.zeros(4), np.full(4, 0.25)),
                    r.PyOptimizeItem("bad", np.full(4, np.nan), np.full(4, 0.25)),
                ],
                _cfg(),
            )


# ---------------------------------------------------------------------------
# Risk contributions
# ---------------------------------------------------------------------------


class TestRiskContributions:
    def test_pct_contributions_sum_to_one(self, model):
        rc = r.compute_risk_contributions(model, np.full(4, 0.25), IDS4)
        assert abs(rc.pct_contribution().sum() - 1.0) < 1e-12
        assert rc.total_vol_annualized > 0

    def test_refuses_zero_book(self, model):
        with pytest.raises(ValueError, match="degenerate"):
            r.compute_risk_contributions(model, np.zeros(4), IDS4)


# ---------------------------------------------------------------------------
# Factor panels
# ---------------------------------------------------------------------------


class TestFactorPanels:
    def test_momentum_12_1_identity(self):
        prices = np.cumprod(np.full((300, 1), 1.01), axis=0)
        out = r.momentum_panel(prices, 252, 21)
        d = 280
        expect = prices[d - 21, 0] / prices[d - 252, 0] - 1
        assert abs(out[d, 0] - expect) < 1e-9
        assert np.isnan(out[251, 0])

    def test_zscore_and_rank_shapes_and_nan(self):
        panel = np.array([[1.0, 2.0, 3.0, np.nan]])
        z = r.zscore_panel(panel, 2)
        assert np.isnan(z[0, 3])
        assert abs(np.nanmean(z[0]) if not np.isnan(z[0]).all() else 0) < 1e-12
        rk = r.rank_panel(panel, 2)
        assert rk[0, 0] == 0.0 and rk[0, 2] == 1.0

    def test_composite_all_or_nothing(self):
        f1 = np.array([[1.0, 2.0]])
        f2 = np.array([[np.nan, 4.0]])
        out = r.composite_scores([f1, f2], np.array([0.5, 0.5]))
        assert np.isnan(out[0, 0])
        assert out[0, 1] == 3.0

    def test_infinity_refused_nan_allowed(self):
        with pytest.raises(ValueError, match="non-finite"):
            r.zscore_panel(np.array([[1.0, np.inf, 2.0]]), 2)


# ---------------------------------------------------------------------------
# Rebalance simulation + cost parity
# ---------------------------------------------------------------------------


class TestRebalanceSim:
    def test_dp_charged_per_sold_isin(self):
        prices = np.full((2, 3), 100.0)
        targets = np.array([[0.3, 0.3, 0.3], [0.0, 0.0, 0.9]])
        res = r.simulate_rebalance_policy(
            prices, targets, 1_000_000.0, "calendar", 1.0
        )
        dp = res.cost_dp()
        assert abs(dp[1] - 2 * 15.34) < 1e-9  # two ISINs sold on day 1
        assert dp[0] == 0.0  # buy-only day has no DP charge

    def test_small_book_dp_dominates_on_sell_day(self):
        """The Phase-1 measured fact behind the small-book refusal."""
        n = 10
        prices = np.full((2, n), 100.0)
        targets = np.vstack([np.full(n, 0.1), np.zeros(n)])
        res = r.simulate_rebalance_policy(
            prices, targets, 50_000.0, "calendar", 1.0
        )
        assert res.cost_dp()[1] > res.cost_regulatory()[1]

    def test_refuses_shape_mismatch_and_bad_policy(self):
        prices = np.full((2, 2), 100.0)
        with pytest.raises(ValueError, match="shape"):
            r.simulate_rebalance_policy(
                prices, np.full((3, 2), 0.5), 1e6, "calendar", 1.0
            )
        with pytest.raises(ValueError, match="policy"):
            r.simulate_rebalance_policy(
                prices, np.full((2, 2), 0.5), 1e6, "weekly", 1.0
            )


class TestCostScheduleExport:
    def test_equity_delivery_schedule_fields(self):
        s = r.indian_cost_schedule("equity_delivery")
        assert s["brokerage_per_order"] == 20.0
        assert s["stt_rate"] == 0.001
        assert s["exchange_txn_rate"] == 0.0000345
        assert s["sebi_turnover_rate"] == 0.000001
        assert s["stamp_duty_rate"] == 0.00015
        assert s["gst_rate"] == 0.18
        assert s["dp_sell_charge_per_isin_per_day"] == 15.34

    def test_unknown_segment_refused_with_alternatives(self):
        with pytest.raises(ValueError, match="equity_delivery"):
            r.indian_cost_schedule("delivery")
