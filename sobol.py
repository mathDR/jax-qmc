Use new-joe-kuo-6.21201 from Sobol sequence generator
""" Generate Sobol Sequences."""
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def sobol_points_jit(
    num_points: int,
    dim: int,
    direction_numbers: dict,
):
    """
    JIT compilable version of Sobol points generator.
    """

    L = int(np.ceil(np.log(num_points) / np.log(2.0)))

    # Compute C array
    C = jnp.zeros(num_points, dtype=jnp.uint32)
    C = C.at[0].set(1)

    def compute_C_scan(i: int) -> int:
        value = i
        count = 1

        while (value & 1) == 1:

            value >>= 1

            count += 1

        return count

    C = jax.vmap(compute_C_scan)(jnp.arange(1, num_points))
    C = jnp.concatenate([jnp.array([1], dtype=jnp.uint32), C])

    # Initialize points

    POINTS = jnp.zeros((num_points, dim), dtype=jnp.float64)

    # First dimension

    V = jnp.zeros(L + 1, dtype=jnp.uint32)

    for i in range(1, L + 1):

        V = V.at[i].set(1 << (32 - i))

    X = jnp.zeros(num_points, dtype=jnp.uint32)
    X = X.at[0].set(0)
    POINTS = POINTS.at[0, 0].set(0.0)


    def first_dim_loop(i: int, carry: tuple) -> tuple:
        X, POINTS = carry
        X_new = X[i-1] ^ V[C[i-1]]
        X = X.at[i].set(X_new)
        POINTS = POINTS.at[i, 0].set(float(X_new) / (2.0**32))

        return (X, POINTS)

   

    X, POINTS = jax.lax.fori_loop(1, num_points, first_dim_loop, (X, POINTS))

    # Remaining dimensions

    def remaining_dims_loop(j: int, POINTS: jax.Array) -> jax.Array:

        d, s, a = direction_numbers[j]['d'], direction_numbers[j]['s'], direction_numbers[j]['a']

        m = direction_numbers[j]['m']

        V = jnp.zeros(L + 1, dtype=jnp.uint32)

        if L <= s:

            for i in range(1, L + 1):

                V = V.at[i].set(m[i] << (32 - i))

        else:

            for i in range(1, s + 1):

                V = V.at[i].set(m[i] << (32 - i))

            for i in range(s + 1, L + 1):

                V_i = V[i - s] ^ (V[i - s] >> s)

                for k in range(1, s):

                    bit = (a >> (s - 1 - k)) & 1

                    V_i ^= (bit * V[i - k])

                V = V.at[i].set(V_i)


        X = jnp.zeros(N, dtype=jnp.uint32)

        X = X.at[0].set(0)

        def dim_j_loop(i, X):

            X_new = X[i-1] ^ V[C[i-1]]

            X = X.at[i].set(X_new)

            POINTS = POINTS.at[i, j].set(float(X_new) / (2.0**32))

            return X

       

        X = jax.lax.fori_loop(1, num_points, dim_j_loop, X)

        return POINTS

    POINTS = jax.lax.fori_loop(1, D, remaining_dims_loop, POINTS)

    return POINTS

# Example usage function

def load_direction_numbers(filename):

    """

    Load direction numbers from file (similar to C++ version).

    This is a simplified version - in practice you'd parse the actual file format.

    """

    # This is a placeholder - you would implement actual file parsing here

    # For demonstration, returning some example direction numbers

    direction_numbers = {}

   

    # Example for dimension 1 (first dimension)

    direction_numbers[0] = {

        'd': 1, 's': 0, 'a': 0,

        'm': jnp.zeros(1, dtype=jnp.uint32)

    }

   

    # Add more dimensions as needed

    # This is just a template - you'd load actual data from the file

   

    return direction_numbers
