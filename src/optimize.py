from typing import Tuple

import jax
import jax.numpy as jnp

from src.grid import Grid
from src.physics import compute_loss, compute_phase_angles


def optimize_grid(
    grid: Grid,
    learning_rate: float = 0.01,
    num_steps: int = 1000,
    lambda_bal: float = 100.0,
    lambda_cap: float = 100.0,
    verbose: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, list]:
    """
    Runs gradient descent optimization to find optimal generation.
    """
    num_nodes = grid.node_demand.shape[0]

    # Initialize parameters
    total_demand = jnp.sum(grid.node_demand)
    num_gens = jnp.sum(grid.node_is_generator)
    per_gen = jnp.where(num_gens > 0, total_demand / num_gens, 0.0)
    generation = jnp.where(grid.node_is_generator, per_gen, 0.0)

    # JIT-compile the gradient function
    # We differentiate with respect to generation only
    loss_grad_fn = jax.value_and_grad(compute_loss)

    loss_history = []

    for i in range(num_steps):
        loss_val, grad_gen = loss_grad_fn(generation, grid, lambda_bal, lambda_cap)

        # Update parameters
        generation = generation - learning_rate * grad_gen

        # Clip generation to be non-negative (simple constraint)
        # Generators exist where node_is_generator is True
        # Non-generators should have generation 0
        generation = jnp.maximum(0.0, generation)
        generation = jnp.where(grid.node_is_generator, generation, 0.0)

        loss_history.append(float(loss_val))

        if verbose and i % (num_steps // 10) == 0:
            print(f"Step {i}: Loss = {loss_val:.2f}")

    theta = compute_phase_angles(grid, generation)
    return theta, generation, loss_history
