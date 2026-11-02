"""Non-contiguous NumPy input is accepted (copied), never a panic."""

import numpy as np

import raptorbt
import raptorbt._raptorbt as _native


def test_simulate_portfolio_mc_accepts_a_transposed_matrix():
    rng = np.random.default_rng(0)
    R = rng.normal(0.0005, 0.01, size=(250, 3))
    weights = np.array([0.5, 0.3, 0.2])
    corr = np.corrcoef(R.T)
    kwargs = dict(initial_value=100_000.0, n_simulations=200, horizon_days=20, seed=1)
    fast = raptorbt.simulate_portfolio_mc(
        [np.ascontiguousarray(c) for c in R.T], weights, list(corr), **kwargs
    )
    view = raptorbt.simulate_portfolio_mc(list(R.T), weights[::1], list(corr.T), **kwargs)
    np.testing.assert_equal(view, fast)


def test_strided_vector_input_matches_contiguous():
    c = np.linspace(100.0, 120.0, 60)
    wide = np.empty(120)
    wide[::2] = c
    view = wide[::2]
    assert not view.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(raptorbt.sma(view, 5), raptorbt.sma(c, 5))


def test_array_pricing_accepts_strided_inputs():
    n = 6
    base = {
        "underlying": np.linspace(95.0, 105.0, n),
        "strike": np.full(n, 100.0),
        "tau": np.full(n, 0.25),
        "vol": np.full(n, 0.2),
        "rate": np.full(n, 0.05),
        "div": np.zeros(n),
    }
    is_call = np.array([True, False] * 3)

    def strided(a):
        buf = np.empty((len(a), 2), dtype=a.dtype)
        buf[:, 0] = a
        view = buf[:, 0]
        assert not view.flags["C_CONTIGUOUS"]
        return view

    want = _native.pricing_price_array("black_scholes", is_call, *base.values())
    got = _native.pricing_price_array(
        "black_scholes", strided(is_call), *(strided(a) for a in base.values())
    )
    np.testing.assert_array_equal(got, want)
