from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax

from src.grid import Grid
from src.physics import compute_loss


class OptimizationState(NamedTuple):
    params: Tuple[jnp.ndarray, jnp.ndarray]  # (theta, generation)
    opt_state: optax.OptState


def initialize_optimization(
    grid: Grid, optimizer: optax.GradientTransformation
) -> OptimizationState:
    """
    Initializes the optimization variables and optimizer state.
    """
    num_nodes = grid.node_demand.shape[0]
    theta = jnp.zeros(num_nodes)
    generation = jnp.copy(grid.node_demand)

    # Ensure only generators have non-zero generation
    generation = jnp.where(grid.node_is_generator, generation, 0.0)

    params = (theta, generation)
    opt_state = optimizer.init(params)

    return OptimizationState(params=params, opt_state=opt_state)


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
    params = state.params
    theta, generation = params

    # Define a helper to compute loss from params
    def loss_fn(p):
        t, g = p
        return compute_loss(t, g, grid, lambda_bal, lambda_cap, lambda_angle)

    # Value and Gradient
    loss_val, grads = jax.value_and_grad(loss_fn)(params)

    # Grads for theta and generation
    grads_theta, grads_gen = grads

    # 1. Handle Reference Angle
    # Fix the reference angle (theta[0] = 0) by zeroing its gradient
    grads_theta = grads_theta.at[0].set(0.0)

    # Update grads tuple
    grads = (grads_theta, grads_gen)

    # 2. Apply Optimizer Updates
    updates, new_opt_state = optimizer.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # 3. Post-process / Projection (Physical Constraints)
    new_theta, new_gen = new_params

    # Ensure theta[0] remains exactly 0 (redundant but safe)
    new_theta = new_theta.at[0].set(0.0)

    # Project generation to be non-negative and only at generator nodes
    new_gen = jnp.maximum(0.0, new_gen)
    new_gen = jnp.where(grid.node_is_generator, new_gen, 0.0)

    new_state = OptimizationState(params=(new_theta, new_gen), opt_state=new_opt_state)
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
