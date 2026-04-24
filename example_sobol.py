# Complete example with sample data
# Enable 64-bit precision for better accuracy
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .sobol import load_direction_numbers, sobol_points_jit


def example_sobol_points(num_points=100, dim=2):
    """
    Example usage with sample direction numbers.
    """

    # Sample direction numbers (you would load these from the actual file)

    direction_numbers = [

        {

            'd': 1, 's': 0, 'a': 0,

            'm': jnp.array([0], dtype=jnp.uint32)

        },

        {

            'd': 2, 's': 1, 'a': 1,

            'm': jnp.array([0, 1], dtype=jnp.uint32)

        }

    ]

   

    # Convert to dictionary format

    dir_dict = {}

    for i, dn in enumerate(direction_numbers):

        dir_dict[i] = dn

   

    # Generate points

    points = sobol_points_jit(num_points, dim, dir_dict)

    return points

# Main function equivalent to C++ main

def main(num_points, dim, filename):
    """
    Main function equivalent to C++ version.
    """

    # Load direction numbers from file

    direction_numbers = load_direction_numbers(filename)

    # Generate Sobol points

    points = sobol_points_jit(num_points, dim, direction_numbers)

    # Display points
    print(f"Generated {N} Sobol points in {D} dimensions:")

    for i in range(min(10, num_points)):  # Print first 10 points
        print(" ".join([f"{x:.15f}" for x in points[i]]))

    return points

if __name__ == "__main__":

    # Example usage

    points = example_sobol_points(10, 2)

    print("Sample Sobol points:")

    print(points[:5])  # Print first 5 points
