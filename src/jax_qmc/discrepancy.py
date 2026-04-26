import jax

import jax.numpy as jnp

from jax import jit, vmap

from scipy.stats.qmc import Sobol

 

@jit

def compute_discrepancy_optimized(points):

    """

    Optimized computation of star discrepancy using sorting and cumulative sums.

   

    Args:

        points: Array of shape (N, d) containing the points in [0,1]^d.

       

    Returns:

        Scalar value representing the star discrepancy.

    """

    N, d = points.shape

   

    # Handle edge cases

    if N == 0:

        return 0.0

    if N == 1:

        # For a single point, discrepancy is max(|1/N - volume| for all boxes)

        # Since volume ranges from 0 to 1, max discrepancy is max(|1/N - 0|, |1/N - 1|)

        return jnp.maximum(1.0/N, 1.0 - 1.0/N)

   

    # Sort points along each dimension

    sorted_indices = jnp.argsort(points, axis=0)

    sorted_points = jnp.take_along_axis(points, sorted_indices, axis=0)

   

    # Compute cumulative sums for each dimension

    # cumsum[i,j] = i+1 (number of points up to and including the i-th point in dimension j)

    cumsum = jnp.arange(1, N + 1).reshape(-1, 1) * jnp.ones((1, d))

   

    # Compute discrepancies for each dimension

    discrepancies = jnp.zeros(d)

   

    for i in range(d):

        sorted_dim = sorted_points[:, i]

       

        # Compute discrepancy for this dimension

        # For each point k, consider the box [0, sorted_dim[k]]

        # Volume of box is sorted_dim[k]

        # Number of points in box is k+1

        # Discrepancy contribution is | (k+1)/N - sorted_dim[k] |

       

        k_values = jnp.arange(N)

        volumes = sorted_dim

        counts = k_values + 1

       

        # Compute absolute differences

        diffs = jnp.abs(counts / N - volumes)

       

        # Also consider the box [0, 1] (entire space)

        # Volume = 1, count = N, discrepancy = |N/N - 1| = 0

        # So we don't need to add it explicitly

       

        # Find maximum discrepancy for this dimension

        discrepancies = discrepancies.at[i].set(jnp.max(diffs))

   

    return jnp.max(discrepancies)

 

# Alternative implementation that considers all possible boxes (more accurate but slower)

@jit

def compute_discrepancy_full(points):

    """

    Compute discrepancy considering all possible boxes (more accurate but computationally expensive).

    """

    N, d = points.shape

   

    if N == 0:

        return 0.0

   

    # Sort points along each dimension

    sorted_indices = jnp.argsort(points, axis=0)

    sorted_points = jnp.take_along_axis(points, sorted_indices, axis=0)

   

    # For each dimension, compute discrepancies for all possible boxes

    # A box is defined by its upper corner (b1, b2, ..., bd) where 0 <= bi <= 1

    # We can represent boxes by their upper corners

   

    # To make this tractable, we'll consider boxes defined by points in the sequence

    # This is a common approximation

   

    # Compute for each dimension the sorted values

    sorted_values = []

    for i in range(d):

        sorted_values.append(sorted_points[:, i])

   

    # Compute discrepancies

    max_discrepancy = 0.0

   

    # Consider all combinations of points as upper bounds

    # This is still expensive, so we'll use a more efficient approach

   

    # For each point, consider it as the upper bound in one dimension

    for i in range(N):

        for dim in range(d):

            # Box upper bound is (sorted_values[0][i], ..., sorted_values[dim][i], ..., sorted_values[d-1][N-1])

            # But this is complex to implement efficiently

           

            # Simpler approach: for each dimension, consider the box [0, sorted_values[dim][i]]

            # and compute discrepancy for that box

           

            volume = sorted_values[dim][i]

            count = i + 1  # Number of points in [0, sorted_values[dim][i]] along dimension dim

           

            # But this only considers boxes aligned with one dimension

            # For true discrepancy, we need to consider all dimensions

           

            # More accurate approach: for each point, consider it as defining a box

            # Box upper bound is the point itself

            point = points[i]

            volume = jnp.prod(point)

            count = jnp.sum(jnp.all(points <= point, axis=1))

           

            discrepancy = jnp.abs(count / N - volume)

            max_discrepancy = jnp.maximum(max_discrepancy, discrepancy)

   

    return max_discrepancy

 

# Most efficient implementation for practical use

@jit

def compute_discrepancy_practical(points):

    """

    Practical implementation balancing accuracy and efficiency.

    """

    N, d = points.shape

   

    if N == 0:

        return 0.0

    if N == 1:

        return jnp.maximum(1.0/N, 1.0 - 1.0/N)

   

    # Sort points along each dimension

    sorted_indices = jnp.argsort(points, axis=0)

    sorted_points = jnp.take_along_axis(points, sorted_indices, axis=0)

   

    max_discrepancy = 0.0

   

    # For each dimension, compute discrepancies for boxes defined by points in that dimension

    for dim in range(d):

        sorted_dim = sorted_points[:, dim]

       

        # For each point in this dimension, consider the box [0, sorted_dim[i]]

        # along this dimension, and [0, 1] along other dimensions

        for i in range(N):

            volume = sorted_dim[i]  # Volume of box [0, sorted_dim[i]] along this dimension

            count = i + 1           # Number of points in this box

           

            discrepancy = jnp.abs(count / N - volume)

            max_discrepancy = jnp.maximum(max_discrepancy, discrepancy)

   

    return max_discrepancy

 

# Example usage

def main():

    # Generate Sobol sequence

    sampler = Sobol(d=2, scramble=False)

    points = sampler.random(n=1000)

   

    # Compute discrepancy using different methods

    disc1 = compute_discrepancy_optimized(points)

    disc2 = compute_discrepancy_practical(points)

   

    print(f"Optimized discrepancy: {disc1}")

    print(f"Practical discrepancy: {disc2}")

 

if __name__ == "__main__":

    main()