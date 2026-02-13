import jax.numpy as jnp

from src.grid import create_simple_grid
from src.physics import compute_nodal_injections, compute_power_flows


def test_power_flow_logic():
    print("Initializing Grid...")
    grid = create_simple_grid()

    # Simple case: equal angles -> no flow
    theta = jnp.zeros(3)
    flows = compute_power_flows(grid, theta)
    assert jnp.allclose(flows, 0.0), f"Flows should be 0, got {flows}"
    print("Test 1 Passed: Zero angles -> Zero flows")

    # Case 2: Node 0 angle > Node 1 -> Flow 0->1
    # Line 0 is 0->1, B=10.
    theta = jnp.array([0.1, 0.0, 0.0])
    flows = compute_power_flows(grid, theta)
    # Expected flow on line 0 (0->1): 10 * (0.1 - 0) = 1.0
    # Expected flow on line 2 (2->0): 10 * (0.0 - 0.1) = -1.0
    print(f"Flows: {flows}")
    assert jnp.isclose(flows[0], 1.0), f"Expected 1.0, got {flows[0]}"
    assert jnp.isclose(flows[2], -1.0), f"Expected -1.0, got {flows[2]}"
    print("Test 2 Passed: Simple flow calculation")

    # Check injections
    injections = compute_nodal_injections(grid, flows)
    # Node 0: leaves 0->1 (1.0), enters 2->0 (-1.0 -> leaves -(-1.0)=1.0).
    # Wait, line 2 is defined as 2->0. Flow is -1.0 (meaning 0->2 is 1.0).
    # Injection @ 0 = (flow 0->1) - (flow 2->0) = 1.0 - (-1.0) = 2.0. Correct.
    print(f"Injections: {injections}")
    assert jnp.isclose(injections[0], 2.0)
    print("Test 3 Passed: Nodal injections")


if __name__ == "__main__":
    try:
        test_power_flow_logic()
        print("\nAll tests passed successfully!")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
    except Exception as e:
        print(f"\nERROR: {e}")
