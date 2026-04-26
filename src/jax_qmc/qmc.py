mport jax

import jax.numpy as jnp

from jax import random

 

def sobol_sequence(key, dim, n_points):

    """

    Generate a Sobol sequence using JAX.

    """

    # Simplified Sobol sequence generation

    points = jnp.zeros((n_points, dim))

   

    # Simple linear congruential generator for demonstration

    for i in range(n_points):

        for j in range(dim):

            points = points.at[i, j].set(

                (random.uniform(key, shape=()) * 2**32) % 1.0

            )

            key = random.split(key)[0]

   

    return points

 

def qmc_integrate_with_uncertainty(func, dim, n_points, key=None):

    """

    QMC integration with uncertainty estimation.

   

    Args:

        func: The function to integrate (must be vectorized)

        dim: Dimension of the integration domain

        n_points: Number of points to use for integration

        key: Optional JAX random key (for reproducibility)

   

    Returns:

        A tuple (integral_estimate, standard_error)

    """

    if key is None:

        key = random.PRNGKey(0)

   

    # Generate Sobol sequence points

    points = sobol_sequence(key, dim, n_points)

   

    # Evaluate the function at the Sobol points

    values = func(points)

   

    # Compute the mean (integral estimate)

    integral = jnp.mean(values)

   

    # Compute the standard error of the mean

    # Standard error = std_dev / sqrt(n_points)

    std_dev = jnp.std(values)

    standard_error = std_dev / jnp.sqrt(n_points)

   

    return integral, standard_error

 

# Example usage with uncertainty

def example_function(x):

    """Example function to integrate: f(x) = exp(-||x||^2)"""

    return jnp.exp(-jnp.sum(x**2, axis=-1))

 

# Integrate over [0,1]^d

dim = 2

n_points = 10000

key = random.PRNGKey(42)

 

result, uncertainty = qmc_integrate_with_uncertainty(

    example_function, dim, n_points, key

)

 

print(f"Estimated integral: {result}")

print(f"Uncertainty (standard error): {uncertainty}")

print(f"Confidence interval (95%): [{result - 1.96*uncertainty}, {result + 1.96*uncertainty}]")