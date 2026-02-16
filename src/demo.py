import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from src.grid import (
    Grid,
    create_8_node_complex_grid,
    create_20_node_large_grid,
    create_simple_grid,
)
from src.optimization import initialize_optimization, run_optimization
from src.physics import compute_power_balance_violations
from src.visualization import plot_grid

jax.config.update("jax_platform_name", "cpu")
# Configure JAX to use CPU (usually sufficient for this size)


def create_5_node_grid() -> Grid:
    # (Leaving original for compatibility or specific demo)
    node_demand = jnp.array([0.0, 50.0, 0.0, 30.0, 40.0])
    node_is_generator = jnp.array([True, False, True, False, False])
    gen_cost_a = jnp.array([0.01, 0.0, 0.05, 0.0, 0.0])
    gen_cost_b = jnp.array([10.0, 0.0, 30.0, 0.0, 0.0])
    line_from = jnp.array([0, 1, 0, 1, 3])
    line_to = jnp.array([1, 2, 3, 4, 4])
    line_susceptance = jnp.array([100.0, 100.0, 100.0, 100.0, 100.0])
    line_capacity = jnp.array([100.0, 50.0, 100.0, 50.0, 50.0])

    return Grid(
        node_demand=node_demand,
        node_is_generator=node_is_generator,
        gen_cost_a=gen_cost_a,
        gen_cost_b=gen_cost_b,
        line_from=line_from,
        line_to=line_to,
        line_susceptance=line_susceptance,
        line_capacity=line_capacity,
    )


def run_demo(grid_type: str = "medium"):
    print(f"Initializing {grid_type.capitalize()} Grid Demo...")

    if grid_type == "simple":
        grid = create_simple_grid()
        num_steps = 1000  # Adam converges much faster
    elif grid_type == "complex":
        grid = create_8_node_complex_grid()
        num_steps = 2000
    elif grid_type == "large":
        grid = create_20_node_large_grid()
        num_steps = 5000
    else:  # medium
        grid = create_5_node_grid()
        num_steps = 1500

    # Initial State
    # Use a dummy optimizer just to get the initial params structure
    state_init = initialize_optimization(grid, optax.adam(1e-3))

    suffix = f"_{grid_type}"
    print(f"Saving Initial State Plot (initial_grid{suffix}.png)...")
    plot_grid(
        grid, state_init, f"Initial State ({grid_type})", f"initial_grid{suffix}.png"
    )

    # Run Optimization
    print(f"\nRunning Optimization ({num_steps} steps with Adam)...")
    final_state, loss_history = run_optimization(
        grid,
        num_steps=num_steps,
        learning_rate=0.1,
        print_every=max(1, num_steps // 10),
    )

    print("\nOptimization Complete!")
    print(f"Final Loss: {loss_history[-1]:.4f}")

    # Extract final parameters
    final_theta, final_gen = final_state.params

    from src.physics import compute_power_flows

    flows = compute_power_flows(grid, final_theta)
    balance_mismatch = compute_power_balance_violations(grid, final_gen, flows)

    print(f"Max Power Balance Mismatch: {jnp.max(jnp.abs(balance_mismatch)):.6f}")
    print(
        f"Max Line Capacity Violation: {jnp.max(jnp.maximum(0.0, jnp.abs(flows) - grid.line_capacity)):.6f}"
    )

    print("Saving Final State Plot...")
    plot_grid(
        grid, final_state, f"Optimized State ({grid_type})", f"final_grid{suffix}.png"
    )

    # Plot Loss Curve
    plt.figure()
    plt.plot(loss_history)
    plt.title(f"Optimization Loss History ({grid_type})")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.savefig(f"loss_history{suffix}.png")


def main():
    parser = argparse.ArgumentParser(description="JAX Power Grid Optimization Demo")
    parser.add_argument(
        "--mode",
        choices=["simple", "medium", "complex", "large", "all"],
        default="medium",
        help="Select the demo complexity level (default: medium)",
    )
    args = parser.parse_args()

    if args.mode == "all":
        for m in ["simple", "medium", "complex", "large"]:
            run_demo(m)
            print("-" * 40)
    else:
        run_demo(args.mode)


if __name__ == "__main__":
    main()
