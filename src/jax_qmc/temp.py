import jax

from scipy.stats import multivariate_normal
from mvn_cdf import mvn_cdf_with_error
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

dim = 100
mean = jax.random.normal(jax.random.key(seed=42), (dim,))
cov = jax.random.normal(jax.random.key(seed=43), (dim, dim))
cov = jnp.dot(cov.T,cov)
x = jnp.zeros(dim)
scipy_dist = multivariate_normal(mean, cov)
ref = scipy_dist.cdf(x)
est, se = mvn_cdf_with_error(x, mean, cov, n_samples=50, n_batches=20,key=jax.random.key(seed=5))
print(f"50: {ref}, {est}, {se}")
est, se = mvn_cdf_with_error(x, mean, cov, n_samples=500, n_batches=20,key=jax.random.key(seed=5))
print(f"500: {ref}, {est}, {se}")
est, se = mvn_cdf_with_error(x, mean, cov, n_samples=5_000, n_batches=20,key=jax.random.key(seed=5))
print(f"5000: {ref}, {est}, {se}")
est, se = mvn_cdf_with_error(x, mean, cov, n_samples=50_000, n_batches=20,key=jax.random.key(seed=5))
print(f"50000: {ref}, {est}, {se}")
est, se = mvn_cdf_with_error(x, mean, cov, n_samples=500_000, n_batches=20,key=jax.random.key(seed=5))
print(f"500000: {ref}, {est}, {se}")
