from typing import Tuple

import jax
import jax.numpy as jnp

from src.grid import Grid
from src.physics import compute_loss


def optimize_grid(
    grid: Grid,
    learning_rate: float = 0.01,
    num_steps: int = 1000,
    lambda_bal: float = 100.0,
    lambda_cap: float = 100.0,
    verbose: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, list]:
    """
    Runs gradient descent optimization to find optimal theta and generation.
    """
    num_nodes = grid.node_demand.shape[0]

    # Initialize parameters
    # Theta starts at 0
    theta = jnp.zeros(num_nodes)
    # Generation starts at demand (naive guess)
    generation = jnp.copy(grid.node_demand)

    # JIT-compile the gradient function
    # We differentiate with respect to both theta and generation
    loss_grad_fn = jax.value_and_grad(compute_loss, argnums=(0, 1))

    loss_history = []

    for i in range(num_steps):
        loss_val, (grad_theta, grad_gen) = loss_grad_fn(
            theta, generation, grid, lambda_bal, lambda_cap
        )

        # Update parameters
        theta = theta - learning_rate * grad_theta
        generation = generation - learning_rate * grad_gen

        # Clip generation to be non-negative (simple constraint)
        # Generators exist where node_is_generator is True
        # Non-generators should have generation 0
        generation = jnp.maximum(0.0, generation)
        generation = jnp.where(grid.node_is_generator, generation, 0.0)

        loss_history.append(float(loss_val))

        if verbose and i % (num_steps // 10) == 0:
            print(f"Step {i}: Loss = {loss_val:.2f}")

    return theta, generation, loss_history
