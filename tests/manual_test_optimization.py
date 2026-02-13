import jax.numpy as jnp

from src.grid import create_simple_grid
from src.optimization import initialize_optimization, run_optimization, update_step
from src.physics import compute_power_balance_violations, compute_power_flows


def run_tests():
    print("Running manual optimization tests...")

    # Test 1: Loss Reduction
    print("Test 1: Check loss reduction single step...")
    grid = create_simple_grid()
    state = initialize_optimization(grid)

    # initial step
    state_1, loss_1 = update_step(grid, state)
    state_2, loss_2 = update_step(grid, state_1)

    if loss_2 < loss_1:
        print(f"PASS: Loss reduced from {loss_1:.4f} to {loss_2:.4f}")
    else:
        print(f"FAIL: Loss did not reduce! {loss_1:.4f} -> {loss_2:.4f}")
        return

    # Test 2: Convergence
    print("\nTest 2: Check convergence on simple grid...")
    final_state, losses = run_optimization(
        grid, num_steps=10000, learning_rate=2e-4, print_every=1000
    )

    # Check that power balance is satisfied
    flows = compute_power_flows(grid, final_state.theta)
    mismatch = compute_power_balance_violations(grid, final_state.generation, flows)

    total_mismatch = jnp.sum(jnp.abs(mismatch))
    print(f"Final Total Mismatch: {total_mismatch:.6f}")

    if total_mismatch < 1.0:
        print("PASS: Mismatch is low.")
    else:
        print("FAIL: Mismatch is too high!")

    print(f"Gen 0: {final_state.generation[0]:.4f}")
    print(f"Gen 1: {final_state.generation[1]:.4f}")

    if final_state.generation[0] + final_state.generation[1] > 90.0:
        print(
            "PASS: Total generation looks reasonable (> 90 MW for 100 MW load + losses)."
        )
    else:
        print("FAIL: Total generation too low!")


if __name__ == "__main__":
    run_tests()
