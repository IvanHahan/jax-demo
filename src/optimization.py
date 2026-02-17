from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax

from src.grid import Grid
from src.physics import compute_loss


class OptimizationState(NamedTuple):
    generation: jnp.ndarray
    opt_state: optax.OptState


def initialize_optimization(
    grid: Grid, optimizer: optax.GradientTransformation = None
) -> OptimizationState:
    """
    Initializes the optimization variables and optimizer state.
    """
    if optimizer is None:
        optimizer = optax.adam(1e-3)

    total_demand = jnp.sum(grid.node_demand)
    num_gens = jnp.sum(grid.node_is_generator)
    per_gen = jnp.where(num_gens > 0, total_demand / num_gens, 0.0)
    generation = jnp.where(grid.node_is_generator, per_gen, 0.0)

    opt_state = optimizer.init(generation)

    return OptimizationState(generation=generation, opt_state=opt_state)


def update_step(
    grid: Grid,
    state: OptimizationState,
    optimizer: optax.GradientTransformation,
    lambda_bal: float = 1000.0,
    lambda_cap: float = 1000.0,
    lambda_angle: float = 10.0,
) -> Tuple[OptimizationState, float]:
    """
    Performs one step of optimization using the provided optax optimizer.
    """
    generation = state.generation

    # Define a helper to compute loss from generation
    def loss_fn(g):
        return compute_loss(g, grid, lambda_bal, lambda_cap, lambda_angle)

    # Value and Gradient
    loss_val, grads = jax.value_and_grad(loss_fn)(generation)

    # 2. Apply Optimizer Updates
    updates, new_opt_state = optimizer.update(grads, state.opt_state, generation)
    new_gen = optax.apply_updates(generation, updates)

    # 3. Post-process / Projection (Physical Constraints)

    # Project generation to be non-negative and only at generator nodes
    new_gen = jnp.maximum(0.0, new_gen)
    new_gen = jnp.where(grid.node_is_generator, new_gen, 0.0)

    new_state = OptimizationState(generation=new_gen, opt_state=new_opt_state)
    return new_state, loss_val


def run_optimization(
    grid: Grid,
    num_steps: int = 1000,
    learning_rate: float = 0.05,  # Adaptive learning rate for Adam
    lambda_bal: float = 1000.0,
    lambda_cap: float = 1000.0,
    lambda_angle: float = 10.0,
    print_every: int = 100,
) -> Tuple[OptimizationState, jnp.ndarray]:
    """
    Runs the optimization loop using the Adam optimizer.
    """
    # Initialize Adam
    optimizer = optax.adam(learning_rate)

    state = initialize_optimization(grid, optimizer)
    loss_history = []

    # JIT the update step with the optimizer as a static argument if needed,
    # but here optimizer is a closure or passed in.
    # Since we call it in a loop, let's make a partially applied JIT function.
    @jax.jit
    def step(s):
        return update_step(grid, s, optimizer, lambda_bal, lambda_cap, lambda_angle)

    # Main Loop
    for i in range(num_steps):
        state, loss = step(state)
        loss_history.append(loss)

        if i % print_every == 0:
            print(f"Step {i}: Loss = {loss:.4f}")

    return state, jnp.array(loss_history)
