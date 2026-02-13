import pytest
import jax.numpy as jnp
import optax
from src.grid import create_simple_grid
from src.physics import compute_power_balance_violations
from src.optimization import run_optimization, initialize_optimization, update_step

def test_optimization_reduces_loss():
    grid = create_simple_grid()
    optimizer = optax.adam(1e-3)
    state = initialize_optimization(grid, optimizer)
    
    # initial step
    state_1, loss_1 = update_step(grid, state, optimizer)
    state_2, loss_2 = update_step(grid, state_1, optimizer)
    
    assert loss_2 < loss_1

def test_simple_grid_convergence():
    grid = create_simple_grid()
    
    final_state, losses = run_optimization(grid, num_steps=2000, learning_rate=0.1)
    
    # Check that power balance is satisfied
    from src.physics import compute_power_flows
    theta, generation = final_state.params
    flows = compute_power_flows(grid, theta)
    mismatch = compute_power_balance_violations(grid, generation, flows)
    
    # Total absolute mismatch should be small
    total_mismatch = jnp.sum(jnp.abs(mismatch))
    print(f"Final Mismatch: {total_mismatch}")
    assert total_mismatch < 2.0 # Adam with 3k steps should be quite close
    
    # Check costs 
    print(f"Gen 0: {generation[0]}")
    print(f"Gen 1: {generation[1]}")
    
    # Make sure we didn't just output zero
    assert generation[0] + generation[1] > 95.0
