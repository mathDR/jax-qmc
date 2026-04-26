import jax

import jax.numpy as jnp

from jax import random

 

def sobol_sequence(key, dim, n_points):

    """

    Generate ordered Sobol sequence (simplified version).

    In practice, use a proper Sobol implementation.

    """

    # This is a placeholder - replace with actual Sobol sequence generation

    points = jnp.zeros((n_points, dim))

    for i in range(n_points):

        for j in range(dim):

            # Simple hash-based point generation for demonstration

            points = points.at[i, j].set(

                (random.uniform(key, shape=()) * 2**32) % 1.0

            )

            key = random.split(key)[0]

    return points

 

def scramble_sobol_owen(points, key):

    """

    Apply Owen scrambling to Sobol sequences.

   

    Args:

        points: Array of shape (n_points, dim) with ordered Sobol points

        key: JAX random key

   

    Returns:

        Scrambled points with the same shape

    """

    n_points, dim = points.shape

    scrambled_points = jnp.zeros_like(points)

   

    # Generate random permutations for each dimension

    for d in range(dim):

        # Create a random permutation of the points in this dimension

        # This is a simplified version - full Owen scrambling is more complex

        perm_key = random.split(key, dim)[d]

       

        # For each bit level, apply random permutations

        # Simplified approach: random linear transformation

        scale = random.uniform(perm_key, shape=())

        offset = random.uniform(perm_key, shape=())

       

        # Apply transformation: scrambled = (original * scale + offset) % 1

        scrambled_points = scrambled_points.at[:, d].set(

            (points[:, d] * scale + offset) % 1.0

        )

   

    return scrambled_points

 

def qmc_integrate_scrambled(func, dim, n_points, key=None, scramble=True):

    """

    QMC integration with optional scrambled Sobol sequences.

   

    Args:

        func: The function to integrate (must be vectorized)

        dim: Dimension of the integration domain

        n_points: Number of points to use for integration

        key: Optional JAX random key

        scramble: Whether to apply scrambling

   

    Returns:

        A tuple (integral_estimate, standard_error)

    """

    if key is None:

        key = random.PRNGKey(0)

   

    # Generate ordered Sobol sequence

    points = sobol_sequence(key, dim, n_points)

   

    # Apply scrambling if requested

    if scramble:

        scramble_key = random.split(key)[0]

        points = scramble_sobol_owen(points, scramble_key)

   

    # Evaluate the function at the points

    values = func(points)

   

    # Compute statistics

    integral = jnp.mean(values)

    std_dev = jnp.std(values)

    standard_error = std_dev / jnp.sqrt(n_points)

   

    return integral, standard_error

 

# Example usage

def example_function(x):

    """Example function to integrate: f(x) = exp(-||x||^2)"""

    return jnp.exp(-jnp.sum(x**2, axis=-1))

 

# Compare ordered vs scrambled sequences

dim = 2

n_points = 10000

key = random.PRNGKey(42)

 

# Ordered Sobol

result_ordered, error_ordered = qmc_integrate_scrambled(

    example_function, dim, n_points, key, scramble=False

)

 

# Scrambled Sobol

result_scrambled, error_scrambled = qmc_integrate_scrambled(

    example_function, dim, n_points, key, scramble=True

)

 

print(f"Ordered Sobol - Estimate: {result_ordered:.6f}, Error: {error_ordered:.6f}")

print(f"Scrambled Sobol - Estimate: {result_scrambled:.6f}, Error: {error_scrambled:.6f}")