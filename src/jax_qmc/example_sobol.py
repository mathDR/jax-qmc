# Complete example with sample data
# Enable 64-bit precision for better accuracy
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jax_qmc.sobol import load_direction_numbers, sobol_points_jit


def main(num_points, dim):
    """
    Main function equivalent to C++ version.
    """

    # Load direction numbers from file

    direction_numbers = load_direction_numbers()

    # Generate Sobol points

    points = sobol_points_jit(num_points, dim, direction_numbers)

    # Display points
    print(f"Generated {N} Sobol points in {D} dimensions:")

    for i in range(min(10, num_points)):  # Print first 10 points
        print(" ".join([f"{x:.15f}" for x in points[i]]))

    return points

if __name__ == "__main__":
    # Example usage
    points = main(10, 2)
