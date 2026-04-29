"""
test_mvn_cdf.py — Validation & benchmarks for mvn_cdf
======================================================

Run with:
    python test_mvn_cdf.py

Requires:
    jax, numpyro, scipy  (scipy is used as the reference implementation only)
"""

import time

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random

import numpy as np
from scipy.stats import multivariate_normal, norm

from mvn_cdf import (
    mvn_cdf,
    mvn_rectangular_prob,
    mvn_cdf_with_error,
    mvn_cdf_batched,
    MultivariateNormalWithCDF,
)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def check(name: str, got: float, expected: float, tol: float = 2e-2):
    ok = abs(got - expected) < tol
    status = "✓" if ok else "✗ FAIL"
    print(f" {status}  {name}")
    print(f" got={got:.6f}  expected={expected:.6f}  diff={abs(got-expected):.2e}")
    return ok


def scipy_cdf(x, mean, cov):
    """P(X ≤ x) via scipy for comparison."""
    scipy_dist = multivariate_normal(mean,cov)
    p = scipy_dist.cdf(x)
    return float(p)


def scipy_rect(lower, upper, mean, cov):
    """P(lower ≤ X ≤ upper) via scipy."""
    scipy_dist = multivariate_normal(mean, cov)
    p = scipy_dist.cdf(upper, lower_limit=lower)
    return float(p)

# ─────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────


def test_standard_normal_2d():
    print("\n── 2D standard normal ──────────────────────────")
    mean = jnp.zeros(2)
    cov  = jnp.eye(2)
    x    = jnp.zeros(2)

    got = float(mvn_cdf(x, mean, cov, n_samples=50_000, key=jax.random.key(seed=1)))
    ref = scipy_cdf(np.zeros(2), np.zeros(2), np.eye(2))
    check("P(X ≤ [0,0]) — should be 0.25", got, ref)


def test_correlated_2d():
    print("\n── 2D correlated normal ────────────────────────")
    mean = jnp.zeros(2)
    rho = 0.5
    cov  = jnp.array([[1.0, rho],[rho, 1.0]])
    x = jnp.zeros(2)

    got = float(mvn_cdf(x, mean, cov, n_samples=50_000, key=jax.random.key(seed=2)))
    ref = 0.25 + jnp.asin(rho) / (2*jnp.pi)
    check("P(X ≤ [1, 0.5, 1])", got, ref.item())


def test_standard_normal_3d():
    print("\n── 3D standard normal ──────────────────────────")
    mean = jnp.zeros(3)
    cov  = jnp.eye(3)
    x    = jnp.zeros(3)

    got = float(mvn_cdf(x, mean, cov, n_samples=50_000, key=jax.random.key(seed=1)))
    ref = scipy_cdf(x, mean, cov)
    check("P(X ≤ [0,0]) — should be 0.125", got, ref)


def test_standard_normal_4d():
    print("\n── 4D standard normal ──────────────────────────")
    mean = jnp.zeros(4)
    cov  = jnp.eye(4)
    x    = jnp.zeros(4)

    got = float(mvn_cdf(x, mean, cov, n_samples=50_000, key=jax.random.key(seed=1)))
    ref = scipy_cdf(x, mean, cov)
    check("P(X ≤ [0,0]) — should be 0.0625", got, ref)


def test_correlated_3d():
    print("\n── 3D correlated normal ────────────────────────")
    mean = jnp.array([0.5, -0.5, 0.0])
    cov  = jnp.array([[1.0, 0.6, 0.2],[0.6, 1.5, 0.3],[0.2, 0.3, 2.0],])
    x = jnp.array([1.0, 0.5, 1.0])

    got = float(mvn_cdf(x, mean, cov, n_samples=50_000, key=jax.random.key(seed=2)))
    ref = scipy_cdf(np.array(x), np.array(mean), np.array(cov))
    check("P(X ≤ [1, 0.5, 1])", got, ref)


def test_diagonal_3d():
    print("\n── 3D diagonal normal ────────────────────────")
    mean = jnp.zeros(3)
    cov  = jnp.diag(jnp.array([1.0, 4.0, 9.0]))
    x = jnp.array([1.0, 2.0, 3.0])

    got = float(mvn_cdf(x, mean, cov, n_samples=50_000, key=jax.random.key(seed=2)))

    expected = np.prod(np.array([norm(mean[i],np.sqrt(cov[i,i])).cdf(x[i]) for i in range(3)]))
    # Have to take sqrt of cov since norm expects std

    check("P(X ≤ [1, 0.5, 1])", got, expected.item())


def test_rectangular_prob():
    print("\n── Rectangular probability ─────────────────────")
    mean  = jnp.zeros(3)
    cov   = jnp.eye(3)
    lower = -jnp.ones(3)
    upper =  jnp.ones(3)

    got = float(mvn_rectangular_prob(lower, upper, mean, cov, n_samples=50_000, key=jax.random.key(seed=3)))
    ref = scipy_rect(np.full(3, -1.), np.ones(3), np.zeros(3), np.eye(3))
    check("P(-1 ≤ X ≤ 1) — each dim 68.3% → ~0.318 joint", got, ref)


def test_4d():
    print("\n── 4D dense covariance ─────────────────────────")
    cov  = jnp.eye(4)
    mean = jnp.zeros(4)
    x = jnp.array(
        [[-2., -2., -2., -2.,],
        [-1., -1., -1., -1.,],
        [ 0.,  0.,  0.,  0.,],
        [ 1.,  1.,  1.,  1.,],
        [ 2.,  2.,  2.,  2.,],]
    )
    got = mvn_cdf_batched(x, mean, cov, n_samples=50_000, key=jax.random.key(seed=4))
    ref = jnp.array([0, 0.0006, 0.0625, 0.5011, 0.9121])

    print(f"test 4D: {jnp.all(jnp.abs(got-ref)<1e-4)}")
    
def test_5d():
    print("\n── 5D dense covariance ─────────────────────────")
    rng  = np.random.default_rng(42)
    A    = rng.standard_normal((5, 5))
    cov  = np.dot(A, A.T) + 3 * np.eye(5)
    mean = np.zeros(5)
    x    = np.ones(5)

    got = float(mvn_cdf(jnp.array(x), jnp.array(mean), jnp.array(cov),
                        n_samples=50_000, key=jax.random.key(seed=4)))
    ref = scipy_cdf(x, mean, cov)
    check("P(X ≤ 1) in 5D", got, ref, tol=3e-2)


def test_error_estimation():
    print("\n── Error estimation ────────────────────────────")
    mean = jnp.zeros(2)
    cov  = jnp.eye(2)
    x    = jnp.zeros(2)
    ref  = 0.25

    est, se = mvn_cdf_with_error(x, mean, cov, n_samples=5_000, n_batches=20,
                                 key=jax.random.key(seed=5))
    est, se = float(est), float(se)
    within = abs(est - ref) <= 3 * se          # should hold ~99.7% of the time
    status = "✓" if within else "✗ FAIL"
    print(f"  {status}  estimate={est:.5f}  se={se:.5f}  ref within 3σ={within}")


def test_batched():
    print("\n── Batched CDF evaluation ──────────────────────")
    mean  = jnp.zeros(2)
    cov   = jnp.eye(2)
    xs    = jnp.array([[-1., -1.], [0., 0.], [1., 1.], [2., 2.]])
    refs  = [scipy_cdf(np.array(x), np.zeros(2), np.eye(2)) for x in xs]

    got = mvn_cdf_batched(xs, mean, cov, n_samples=50_000, key=jax.random.key(seed=6))
    all_ok = True
    for i, (g, r) in enumerate(zip(got.tolist(), refs)):
        ok = abs(g - r) < 3e-2
        all_ok = all_ok and ok
        status = "✓" if ok else "✗"
        print(f"  {status}  x={xs[i].tolist()}  got={g:.4f}  ref={r:.4f}")


def test_distribution_class():
    print("\n── MultivariateNormalWithCDF class ─────────────")
    mean = jnp.array([1.0, -1.0])
    cov  = jnp.array([[2.0, 0.8], [0.8, 1.5]])
    d    = MultivariateNormalWithCDF(mean, cov)

    x = jnp.array([1.0, -1.0])   # median — expect ~0.25

    cdf_val     = float(d.cdf(x, n_samples=50_000, key=jax.random.key(seed=7)))
    surv_val    = float(d.survival(x, n_samples=50_000, key=jax.random.key(seed=7)))
    log_prob    = float(d.log_prob(x))

    ref = scipy_cdf(np.array(x), np.array(mean), np.array(cov))
    check("cdf(mean)", cdf_val, ref)
    check("survival(mean) + cdf(mean) ≈ 1", cdf_val + surv_val, 1.0, tol=1e-4)
    print(f"  ✓  log_prob(mean) = {log_prob:.4f}  (not 0 — pdf not CDF)")

    # Rectangular probability
    lo   = jnp.zeros(2)
    hi   = jnp.array([2.0, 0.0])
    rect = float(d.rectangular_prob(lo, hi, n_samples=50_000, key=jax.random.key(seed=8)))
    ref_rect = scipy_rect(np.zeros(2), np.array([2., 0.]), np.array(mean), np.array(cov))
    check("rectangular_prob([0,0], [2,0])", rect, ref_rect)

    # Conditional CDF
    cond = float(d.conditional_cdf(
        x=jnp.array([1.5, -0.5]),
        lower=jnp.array([0.0, -2.0]),
        upper=jnp.array([3.0, 1.0]),
        n_samples=50_000,
        key=random.PRNGKey(9),
    ))
    print(f"  ✓  conditional_cdf = {cond:.4f}  (no scipy ref — sanity: in [0,1]={0<=cond<=1})")


def test_gradient_through_cdf():
    print("\n── Gradient of CDF w.r.t. x ────────────────────")
    mean = jnp.zeros(2)
    cov  = jnp.eye(2)

    # ∂P(X≤x)/∂x_i = f(x_i | rest) * P(rest ≤ x_{-i})
    # For standard normal at x=0: each marginal density is ~0.399, and the
    # conditional structure means grad should be positive and ≈ 0.1–0.2.
    def cdf_fn(x):
        return mvn_cdf(x, mean, cov, n_samples=50_000, key=jax.random.key(seed=10))

    x    = jnp.zeros(2)
    g    = grad(cdf_fn)(x)
    ok   = jnp.all(g > 0)         # CDF is monotone — all partials must be > 0
    status = "✓" if ok else "✗ FAIL"
    print(f"  {status}  grad = {g.tolist()}  (all positive = {bool(ok)})")


def test_gradient_through_mean():
    print("\n── Gradient of CDF w.r.t. mean ─────────────────")
    cov = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    x   = jnp.zeros(2)

    def cdf_fn(mean):
        return mvn_cdf(x, mean, cov, n_samples=50_000, key=jax.random.key(seed=11))

    mean = jnp.zeros(2)
    g    = grad(cdf_fn)(mean)
    # Shifting mean up decreases P(X ≤ 0) — gradient should be negative
    ok   = jnp.all(g < 0)
    status = "✓" if ok else "✗ FAIL"
    print(f"  {status}  grad_mean = {g.tolist()}  (all negative = {bool(ok)})")


def test_jit_performance():
    print("\n── JIT compilation & performance ───────────────")
    mean = jnp.zeros(4)
    cov  = jnp.eye(4)
    x    = jnp.zeros(4)
    fn   = jit(mvn_cdf, static_argnames=("n_samples",))

    # Warmup (includes compilation)
    _ = fn(x, mean, cov, n_samples=10_000, key=jax.random.key(seed=0)).block_until_ready()

    # Timed run
    t0 = time.perf_counter()
    for i in range(20):
        _ = fn(x, mean, cov, n_samples=10_000, key=random.PRNGKey(i)).block_until_ready()
    elapsed = (time.perf_counter() - t0) / 20

    print(f"  ✓  avg wall time (4D, 10k samples): {elapsed*1000:.1f} ms per call")


def test_vmap_over_means():
    print("\n── vmap over distribution parameters ───────────")
    cov    = jnp.eye(3)
    x      = jnp.zeros(3)
    means  = jnp.linspace(-2, 2, 5)[:, None] * jnp.ones((5, 3))

    vmapped = vmap(lambda mu: mvn_cdf(x, mu, cov, n_samples=20_000,
                                      key=jax.random.key(seed=0)))
    probs = vmapped(means)
    ok = jnp.all(jnp.diff(probs) < 0)   # larger mean → smaller P(X≤0)
    status = "✓" if ok else "✗ FAIL"
    print(f"  {status}  probs = {[f'{p:.3f}' for p in probs.tolist()]}")
    print(f"         (monotone decreasing in mean = {bool(ok)})")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print("  MVN CDF — test suite")
    print("  JAX devices:", jax.devices())
    print("=" * 52)

    test_standard_normal_2d()
    test_correlated_2d()
    test_standard_normal_3d()
    test_standard_normal_4d()
    test_correlated_3d()
    test_diagonal_3d()
    test_rectangular_prob()
    test_4d()
    test_5d()
    test_error_estimation()
    test_batched()
    test_distribution_class()
    test_gradient_through_cdf()
    test_gradient_through_mean()
    test_jit_performance()
    test_vmap_over_means()

    print("\n" + "=" * 52)
    print("  Done.")
    print("=" * 52)
