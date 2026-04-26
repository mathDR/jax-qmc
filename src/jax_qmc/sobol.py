""" Generate Sobol Sequences."""
import csv
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


def load_direction_numbers():
    """
    Load direction numbers from file (similar to C++ version).
    """

    direction_numbers= []
    with open("data/new-joe-kuo-6.21201", "r") as fileptr:
        # Use ignore_sequences to handle multiple spaces/tabs as one delimiter
        reader = csv.reader(fileptr, delimiter='\t') 
        # Skip header
        header = next(reader)
        
        for row in fileptr:
            # Split by whitespace and grab the first 3 individual numbers
            parts = row.split()
            row_dict = {}
            row_dict["d"] = int(parts[0])
            row_dict["s"] = int(parts[1])
            row_dict["a"] = int(parts[2])
            # Join the remaining parts as the array of ints for m_i
            row_dict["m_i"] = jnp.array(parts[3:],dtype=int)
            direction_numbers.append(row_dict)

    return direction_numbers

#@jax.jit
def sobol_points_jit(
    num_points: int,
    dim: int,
    direction_numbers: dict,
):
    """
    JIT compilable version of Sobol points generator.
    """

    L = np.ceil(np.log(num_points) / np.log(2.0)).astype(int)[0]

    # Compute C array
    def compute_C_scan(i: int) -> int:
        # Should convert this to a jax.lax.while_loop...
        init_val = (i, 1)

        cond_fun = lambda x: (x[0] & 1) == 1
        def body_fun(x: tuple) -> tuple: 
            return (x[0]>>1, x[1]+1)

        val = jax.lax.while_loop(cond_fun, body_fun, init_val)

        return val[1]

    C = jax.vmap(compute_C_scan, in_axes=0)(
        jnp.arange(1, num_points).astype(jnp.uint32)
    )
    C = jnp.concatenate([jnp.array([1], dtype=jnp.uint32), C])

    # Initialize points

    POINTS = jnp.zeros((num_points, dim), dtype=jnp.float64)

    # First dimension
    def v_i(i:int)->int:
        return 1 << (32-i)
    V = jax.vmap(v_i, in_axes=0)(jnp.arange(1, L+1).astype(jnp.uint32))
    V = jnp.concatenate([jnp.array([0], dtype=jnp.uint32), V])

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

    POINTS = jax.lax.fori_loop(1, dim, remaining_dims_loop, POINTS)

    return POINTS
