"""Option pricing, Greeks and implied volatility.

The differential tests compare against the numbers a reference implementation
produced for the same inputs. The fixture `golden/pricing_reference.json` holds
numbers only.

Tolerances (stated, not tuned):

* price, delta, gamma, vega, theta, rho: relative to `1 + |reference|`, 1e-12.
  Measured agreement is ~1e-16 on every row with vol > 0.
* implied vol, where the reference answered: 2e-8 absolute when vega >= 1 (the
  reference stopped at a price tolerance of 1e-8); otherwise the new answer
  must reprice the target to 1e-9 (the reference's own repricing error is up
  to 3e-7, so it is not the better yardstick there).

Documented differences from the reference (intended fixes):

1. Black-76 theta. The reference's `theta` for a forward had the carry term
   with the wrong sign (`decay - r*P`); the true derivative with the forward
   held fixed is `decay + r*P`. Checked against a finite difference below.
2. `vol == 0`, `tau > 0`. The reference returned delta from spot intrinsic and
   zero for every other Greek. The new Greeks are the exact derivatives of the
   discounted forward payoff (delta follows the forward, theta and rho carry the
   discounting). The price is unchanged.
3. Zero time value. When the price is the discounted intrinsic value the
   reference returned a made-up volatility (the first bisection midpoint);
   the new solver reports `IvStatus.BELOW_BRACKET`.
4. A Black-76 (forward) input with a nonzero `dividend_yield` is refused; the
   reference silently ignored the yield.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from raptorbt import pricing as pr

FIXTURE = Path(__file__).parent / "golden" / "pricing_reference.json"
GREEK_FIELDS = ("price", "delta", "gamma", "vega", "theta", "rho")


@pytest.fixture(scope="module")
def ref():
    d = json.loads(FIXTURE.read_text())

    def table(cols, rows):
        a = np.array([[np.nan if v is None else v for v in r] for r in rows], dtype=float)
        return {n: a[:, i] for i, n in enumerate(cols)}

    return table(d["price_columns"], d["price_rows"]), table(d["iv_columns"], d["iv_rows"])


def _rows(t, model):
    fwd = t["is_forward"] == 1
    # A forward with a dividend yield is a documented refusal (diff 4).
    return (fwd & (t["dividend_yield"] == 0)) if model == "black76" else ~fwd


def _right(t, m):
    return np.where(t["is_call"][m] == 1, "call", "put")


def _close(a, b, tol=1e-12):
    return np.abs(a - b) <= tol * (1.0 + np.abs(b))


@pytest.mark.parametrize("model", pr.MODELS)
def test_price_and_greeks_match_reference_when_vol_positive(ref, model):
    t, _ = ref
    m = _rows(t, model) & (t["vol"] > 0)
    g = pr.greeks(
        _right(t, m), t["spot"][m], t["strike"][m], t["tau"][m], t["vol"][m],
        t["rate"][m], t["dividend_yield"][m], model=model,
    )  # fmt: skip
    assert m.sum() > 500
    for f in GREEK_FIELDS:
        want = t[f][m]
        if model == "black76" and f == "theta":
            # Documented difference 1: the reference's sign error on the carry term.
            want = np.where(t["tau"][m] > 0, want + 2 * t["rate"][m] * t["price"][m], want)
        bad = ~_close(getattr(g, f), want)
        assert not bad.any(), f"{model} {f}: {bad.sum()} rows differ"


@pytest.mark.parametrize("model", pr.MODELS)
def test_price_matches_reference_everywhere_including_zero_vol_and_expiry(ref, model):
    t, _ = ref
    m = _rows(t, model)
    p = pr.price(
        _right(t, m), t["spot"][m], t["strike"][m], t["tau"][m], t["vol"][m],
        t["rate"][m], t["dividend_yield"][m], model=model,
    )  # fmt: skip
    assert _close(p, t["price"][m]).all()


@pytest.mark.parametrize("model", pr.MODELS)
def test_zero_vol_greeks_are_the_derivatives_of_the_forward_payoff(ref, model):
    t, _ = ref
    m = _rows(t, model) & (t["vol"] == 0) & (t["tau"] > 0)
    right = _right(t, m)
    s, k, tau = t["spot"][m], t["strike"][m], t["tau"][m]
    r, q = t["rate"][m], t["dividend_yield"][m]
    g = pr.greeks(right, s, k, tau, 0.0, r, q, model=model)
    fwd = s if model == "black76" else s * np.exp((r - q) * tau)
    call = right == "call"
    itm = np.where(call, fwd > k, fwd < k)
    carry = 1.0 if model == "black76" else np.exp((r - q) * tau)
    df = np.exp(-r * tau)
    want_delta = np.where(itm, np.where(call, 1.0, -1.0) * df * carry, 0.0)
    assert _close(g.delta, want_delta, 1e-14).all()
    assert (g.gamma == 0).all() and (g.vega == 0).all()
    # Where the reference's delta convention coincides (no carry), they agree.
    flat = (r == 0) & (q == 0)
    assert _close(g.delta[flat], t["delta"][m][flat], 1e-14).all()
    # Theta is -dP/dtau for the payoff, by finite difference on the (exact) price.
    h = 1e-7
    up = pr.price(right, s, k, tau + h, 0.0, r, q, model=model)
    dn = pr.price(right, s, k, tau - h, 0.0, r, q, model=model)
    fd = -(up - dn) / (2 * h)
    assert _close(g.theta, fd, 1e-5).all()
    assert not np.isnan(g.rho).any()


def test_black76_theta_is_the_derivative_with_the_forward_fixed():
    for right in ("call", "put"):
        args = (right, 100.0, 95.0, 0.5, 0.2, 0.05)
        g = pr.greeks(*args, model="black76")
        h = 1e-6
        fd = -(
            pr.price(right, 100.0, 95.0, 0.5 + h, 0.2, 0.05, model="black76")
            - pr.price(right, 100.0, 95.0, 0.5 - h, 0.2, 0.05, model="black76")
        ) / (2 * h)
        assert g.theta == pytest.approx(fd, rel=1e-6)


def test_black_scholes_theta_and_rho_match_finite_differences():
    for right in ("call", "put"):
        a = (right, 100.0, 105.0, 0.75, 0.3, 0.04, 0.02)
        g = pr.greeks(*a)
        h = 1e-6
        d = lambda i: tuple(x + (h if j == i else 0) for j, x in enumerate(a[1:]))  # noqa: E731
        dm = lambda i: tuple(x - (h if j == i else 0) for j, x in enumerate(a[1:]))  # noqa: E731
        theta = -(pr.price(right, *d(2)) - pr.price(right, *dm(2))) / (2 * h)
        rho = (pr.price(right, *d(4)) - pr.price(right, *dm(4))) / (2 * h)
        assert g.theta == pytest.approx(theta, rel=1e-6)
        assert g.rho == pytest.approx(rho, rel=1e-6)


@pytest.mark.parametrize("model", pr.MODELS)
def test_implied_vol_matches_reference(ref, model):
    _, t = ref
    m = _rows(t, model)
    right = _right(t, m)
    args = (right, t["spot"][m], t["strike"][m], t["tau"][m], t["rate"][m], t["dividend_yield"][m])
    iv, status = pr.implied_vol_status(t["target"][m], *args, model=model)
    ref_iv = t["reference_iv"][m]
    both = ~np.isnan(iv) & ~np.isnan(ref_iv)
    assert both.sum() > 400

    # Never an answer the reference lacked.
    assert not (~np.isnan(iv) & np.isnan(ref_iv)).any()

    # Where both answered: agree, or the new one reprices better (diff in docstring).
    g = pr.greeks(right[both], args[1][both], args[2][both], args[3][both], iv[both],
                  args[4][both], args[5][both], model=model)  # fmt: skip
    err = np.abs(iv[both] - ref_iv[both])
    well_conditioned = g.vega >= 1.0
    assert (err[well_conditioned] <= 2e-8).all()
    assert well_conditioned.sum() > 100
    new_back = g.price - t["target"][m][both]
    assert (np.abs(new_back) <= 1e-9).all()

    # Reference answered, new refused: only zero time value (documented diff 3).
    old_only = np.isnan(iv) & ~np.isnan(ref_iv)
    assert (status[old_only] == pr.IvStatus.BELOW_BRACKET).all()
    p_lo = pr.price(right[old_only], args[1][old_only], args[2][old_only], args[3][old_only],
                    1e-4, args[4][old_only], args[5][old_only], model=model)  # fmt: skip
    assert (p_lo >= t["target"][m][old_only] - 1e-12).all()

    # Reference refused: the new solver refused too, for a reason.
    both_none = np.isnan(ref_iv)
    assert np.isnan(iv[both_none]).all()
    assert (status[both_none] != pr.IvStatus.OK).all()


def test_implied_vol_round_trip_is_tight():
    rng = np.random.default_rng(7)
    n = 20_000
    s = rng.uniform(50, 150, n)
    k = s * rng.uniform(0.7, 1.3, n)
    tau = rng.uniform(1 / 365, 3, n)
    vol = rng.uniform(0.05, 1.5, n)
    r = rng.uniform(-0.01, 0.08, n)
    q = rng.uniform(0, 0.04, n)
    right = rng.choice(np.array(["call", "put"]), n)
    p = pr.price(right, s, k, tau, vol, r, q)
    iv, status = pr.implied_vol_status(p, right, s, k, tau, r, q)
    solved = status == pr.IvStatus.OK
    assert solved.mean() > 0.9
    back = pr.price(
        right[solved], s[solved], k[solved], tau[solved], iv[solved], r[solved], q[solved]
    )
    assert (np.abs(back - p[solved]) <= 1e-10 * (1 + p[solved])).all()
    # What failed failed for the reason it should: no time value left.
    assert set(status[~solved].tolist()) <= {
        int(pr.IvStatus.BELOW_BRACKET), int(pr.IvStatus.BELOW_INTRINSIC), int(pr.IvStatus.ABOVE_UPPER_BOUND)
    }  # fmt: skip


def test_scalar_and_array_agree_and_shapes_follow_numpy_broadcasting():
    strikes = np.array([90.0, 100.0, 110.0])
    taus = np.array([[0.1], [0.5]])
    p = pr.price("call", 100.0, strikes, taus, 0.25, 0.03, 0.01)
    assert p.shape == (2, 3)
    for i, tau in enumerate(taus[:, 0]):
        for j, k in enumerate(strikes):
            assert p[i, j] == pr.price("call", 100.0, k, tau, 0.25, 0.03, 0.01)
    g = pr.greeks(np.array(["call", "put", "c"]), 100.0, 100.0, 0.5, 0.2)
    assert isinstance(g, pr.GreeksArray) and g.delta.shape == (3,)
    assert g.delta[0] == g.delta[2] and g.delta[1] < 0
    assert isinstance(pr.greeks("put", 100.0, 100.0, 0.5, 0.2), pr.Greeks)
    assert pr.greeks("put", 100.0, 100.0, 0.5, 0.2).as_dict().keys() == set(GREEK_FIELDS)


def test_invalid_inputs_raise_for_scalars_and_are_nan_for_arrays_never_zero():
    for bad in [
        ("call", 0.0, 100.0, 1.0, 0.2),
        ("call", 100.0, -5.0, 1.0, 0.2),
        ("call", 100.0, 100.0, -0.1, 0.2),
        ("call", 100.0, 100.0, 1.0, -0.2),
        ("call", float("nan"), 100.0, 1.0, 0.2),
        ("call", 100.0, 100.0, 1.0, float("inf")),
    ]:
        with pytest.raises(ValueError):
            pr.price(*bad)
        with pytest.raises(ValueError):
            pr.greeks(*bad)
    right = np.array(["call"] * 4)
    s = np.array([100.0, 0.0, np.nan, 100.0])
    p = pr.price(right, s, 100.0, np.array([1.0, 1.0, 1.0, -1.0]), 0.2)
    assert np.isfinite(p[0]) and np.isnan(p[1:]).all()
    g = pr.greeks(right, s, 100.0, 1.0, 0.2)
    for f in GREEK_FIELDS:
        assert np.isfinite(getattr(g, f)[0]) and np.isnan(getattr(g, f)[1:3]).all()
    with pytest.raises(ValueError, match="right"):
        pr.price("straddle", 100.0, 100.0, 1.0, 0.2)
    with pytest.raises(ValueError, match="model"):
        pr.price("call", 100.0, 100.0, 1.0, 0.2, model="heston")
    with pytest.raises(ValueError, match="dividend_yield"):
        pr.price("call", 100.0, 100.0, 1.0, 0.2, 0.0, 0.01, model="black76")
    with pytest.raises(ValueError):
        pr.price("call", np.ones(3), np.ones(4), 1.0, 0.2)


def test_expiry_and_extremes():
    g = pr.greeks("call", 105.0, 100.0, 0.0, 0.3, 0.05)
    assert (g.price, g.delta, g.gamma, g.vega, g.theta, g.rho) == (5.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    assert pr.greeks("put", 105.0, 100.0, 0.0, 0.3).delta == 0.0
    assert pr.price("call", 100.0, 100.0, 1.0, 0.0) == 0.0
    # Deep tails stay finite and inside no-arbitrage bounds.
    p = pr.price(
        "call", 100.0, np.array([1.0, 1e4]), np.array([50.0, 1e-9]), np.array([20.0, 1e-6])
    )
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 100.0).all()


def test_put_call_parity_vectorised():
    rng = np.random.default_rng(1)
    n = 5000
    s, k = rng.uniform(50, 150, n), rng.uniform(50, 150, n)
    tau, vol = rng.uniform(0.01, 3, n), rng.uniform(0.05, 1.0, n)
    r, q = rng.uniform(-0.01, 0.08, n), rng.uniform(0, 0.05, n)
    c = pr.price("call", s, k, tau, vol, r, q)
    p = pr.price("put", s, k, tau, vol, r, q)
    assert np.allclose(c - p, s * np.exp(-q * tau) - k * np.exp(-r * tau), rtol=0, atol=1e-10)


def test_large_arrays_match_scalars_and_threads_do_not_change_results():
    rng = np.random.default_rng(3)
    n = 50_000  # above the parallel threshold
    s, k = rng.uniform(80, 120, n), rng.uniform(80, 120, n)
    tau, vol = rng.uniform(0.01, 2, n), rng.uniform(0.05, 0.9, n)
    big = pr.greeks("call", s, k, tau, vol, 0.03, 0.01)
    for i in (0, 4095, 4096, 25_000, n - 1):
        one = pr.greeks("call", s[i], k[i], tau[i], vol[i], 0.03, 0.01)
        for f in GREEK_FIELDS:
            assert getattr(big, f)[i] == getattr(one, f)
    again = pr.greeks("call", s, k, tau, vol, 0.03, 0.01)
    for f in GREEK_FIELDS:
        assert np.array_equal(getattr(big, f), getattr(again, f))
    # Non-contiguous input is handled.
    strided = pr.price("call", s[::2], k[::2], tau[::2], vol[::2])
    assert np.array_equal(strided, pr.price("call", s, k, tau, vol)[::2])


def test_implied_vol_failures_are_reported_not_guessed():
    # Call S=100 K=90 tau=0.25 r=0: intrinsic 10.
    base = ("call", 100.0, 90.0, 0.25)
    cases = {
        5.0: pr.IvStatus.BELOW_INTRINSIC,
        10.0: pr.IvStatus.BELOW_BRACKET,
        99.9: pr.IvStatus.ABOVE_BRACKET,
        100.0: pr.IvStatus.ABOVE_UPPER_BOUND,
        -1.0: pr.IvStatus.BELOW_INTRINSIC,
        float("nan"): pr.IvStatus.INVALID_INPUT,
    }
    for target, want in cases.items():
        iv, status = pr.implied_vol_status(target, *base)
        assert np.isnan(iv) and status == want, (target, status)
        with pytest.raises(pr.ImpliedVolError) as e:
            pr.implied_vol(target, *base)
        assert e.value.status == want and isinstance(e.value, ValueError)
    assert pr.implied_vol_status(10.0, "call", 100.0, 90.0, 0.0)[1] == pr.IvStatus.EXPIRED
    out = pr.implied_vol(np.array([5.0, 14.0]), "call", 100.0, 90.0, 0.25)
    assert np.isnan(out[0]) and np.isfinite(out[1])
    assert pr.IvStatus.BELOW_BRACKET.text == "below_bracket"
    # A bracket can be widened by the caller.
    wide = pr.implied_vol(99.9, *base, hi=500.0)
    assert 5.0 < wide < 500.0
    with pytest.raises(ValueError):
        pr.implied_vol(14.0, *base, lo=0.0)
    # Out of the money with zero price: no volatility, not 0.
    assert pr.implied_vol_status(0.0, "call", 100.0, 150.0, 0.25)[1] == pr.IvStatus.BELOW_BRACKET


def test_iv_budget_exhaustion_is_reported():
    p = pr.price("call", 100.0, 100.0, 0.25, 0.37)
    iv, status = pr.implied_vol_status(p, "call", 100.0, 100.0, 0.25, max_iter=1, tol=1e-14)
    assert np.isnan(iv) and status == pr.IvStatus.NO_CONVERGENCE


def test_helpers():
    assert pr.year_fraction(0, 365 * 86_400 * 10**9) == 1.0
    assert pr.forward_price(100.0, 0.5, 0.06, 0.02) == pytest.approx(100.0 * np.exp(0.02))
    f = pr.forward_price(np.array([100.0, 200.0]), 1.0, 0.05)
    assert f == pytest.approx(np.array([100.0, 200.0]) * np.exp(0.05))
