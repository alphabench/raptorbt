"""``simulate_portfolio_mc(n_simulations=0)`` must refuse with a validation error.

Found by the ST6 stress workload (benches/stress/workloads/st6_monte_carlo.py). It panics at
src/portfolio/monte_carlo.rs:272 (``final_values`` is empty and ``n - 1`` wraps), which reaches
Python as ``PanicException``: a ``BaseException`` that ``except Exception`` in a strategy hook
does not catch. Every other bad argument of this function raises ``ValueError``.
"""

import numpy as np
import pytest

import raptorbt


def test_zero_simulations_is_a_value_error_not_a_panic():
    rets = [np.random.default_rng(1).normal(0.0, 0.001, 50)]
    with pytest.raises(ValueError):
        raptorbt.simulate_portfolio_mc(
            rets, np.array([1.0]), [np.array([1.0])], 1e6, n_simulations=0, horizon_days=5, seed=1
        )


def test_zero_horizon_is_a_value_error_naming_the_argument():
    rets = [np.random.default_rng(1).normal(0.0, 0.001, 50)]
    with pytest.raises(ValueError) as e:
        raptorbt.simulate_portfolio_mc(
            rets, np.array([1.0]), [np.array([1.0])], 1e6, n_simulations=10, horizon_days=0, seed=1
        )
    assert e.value.parameter == "horizon_days"
