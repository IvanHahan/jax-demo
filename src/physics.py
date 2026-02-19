import jax
import jax.numpy as jnp

from src.grid import Grid


def build_bus_susceptance_matrix(grid: Grid) -> jnp.ndarray:
    """
    Builds the bus susceptance matrix B for DC power flow.
    """
    num_nodes = grid.node_demand.shape[0]
    bbus = jnp.zeros((num_nodes, num_nodes))

    u = grid.line_from
    v = grid.line_to
    b = grid.line_susceptance

    # Diagonal entries: sum of connected line susceptances
    bbus = bbus.at[u, u].add(b)
    bbus = bbus.at[v, v].add(b)

    # Off-diagonal entries: -susceptance for each connection
    bbus = bbus.at[u, v].add(-b)
    bbus = bbus.at[v, u].add(-b)

    return bbus


def compute_phase_angles(
    grid: Grid, generation: jnp.ndarray, ref_bus: int = 0
) -> jnp.ndarray:
    """
    Solves DC power flow to derive phase angles from generation and demand.

    Args:
        grid: Grid PyTree
        generation: [N] Generation at each node
        ref_bus: Reference (slack) bus index

    Returns:
        theta: [N] Phase angles at each node (radians), with theta[ref_bus] = 0
    """
    num_nodes = grid.node_demand.shape[0]
    net_injection = generation - grid.node_demand

    bbus = build_bus_susceptance_matrix(grid)

    # Remove reference bus row/col to make the system nonsingular
    indices = jnp.concatenate((jnp.arange(ref_bus), jnp.arange(ref_bus + 1, num_nodes)))
    b_reduced = jnp.take(jnp.take(bbus, indices, axis=0), indices, axis=1)
    p_reduced = jnp.take(net_injection, indices)

    theta_reduced = jnp.linalg.solve(b_reduced, p_reduced)

    theta = jnp.zeros(num_nodes)
    theta = theta.at[indices].set(theta_reduced)
    return theta


def compute_power_flows(grid: Grid, theta: jnp.ndarray) -> jnp.ndarray:
    """
    Computes power flow on each line given phase angles.
    f_ij = B_ij * (theta_i - theta_j)

    Args:
        grid: Grid PyTree
        theta: [N] Phase angles at each node (radians)

    Returns:
        flows: [E] Power flow on each line (from -> to)
    """
    # Extract indices
    u = grid.line_from
    v = grid.line_to

    # Difference in phase angles
    delta_theta = theta[u] - theta[v]

    # Flow = Susceptance * delta_theta
    flows = grid.line_susceptance * delta_theta
    return flows


def compute_nodal_injections(grid: Grid, flows: jnp.ndarray) -> jnp.ndarray:
    """
    Computes net power injection at each node given line flows.
    P_i = sum(f_ij) - sum(f_ji)

    Args:
        grid: Grid PyTree
        flows: [E] Power flow on each line

    Returns:
        injections: [N] Net power injection at each node
    """
    num_nodes = grid.node_demand.shape[0]
    injections = jnp.zeros(num_nodes)

    # Add flow leaving nodes (from)
    injections = injections.at[grid.line_from].add(flows)

    # Subtract flow entering nodes (to)
    injections = injections.at[grid.line_to].add(-flows)

    return injections


def compute_power_balance_violations(
    grid: Grid, generation: jnp.ndarray, flows: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes mismatch between generation, demand, and net flow.
    Mismatch_i = Gen_i - Demand_i - NetInjection_i

    Args:
        grid: Grid PyTree
        generation: [N] Generation at each node
        flows: [E] Power flow on each line

    Returns:
        mismatch: [N] Power balance mismatch at each node (should be 0)
    """
    net_injection = compute_nodal_injections(grid, flows)
    mismatch = generation - grid.node_demand - net_injection
    return mismatch


def compute_total_cost(grid: Grid, generation: jnp.ndarray) -> jnp.ndarray:
    """
    Computes linear generation cost: sum(c*g)
    """
    cost = grid.gen_cost * generation
    return jnp.sum(cost)


@jax.jit
def compute_loss(
    generation: jnp.ndarray,
    grid: Grid,
    lambda_bal: float = 1000.0,
    lambda_cap: float = 1000.0,
    lambda_angle: float = 10.0,
) -> jnp.ndarray:
    """
    Computes the total penalized loss.

    L = Cost(g) + lambda_bal * ||Balance||^2 + lambda_cap * ||CapWait||^2 + lambda_angle * ||theta||^2
    """
    # 1. Generation Cost
    cost = compute_total_cost(grid, generation)

    # 2. Power Balance Penalty
    theta = compute_phase_angles(grid, generation)
    flows = compute_power_flows(grid, theta)
    net_injection = compute_nodal_injections(grid, flows)

    # P_gen - P_dem - P_inj = 0
    balance_mismatch = generation - grid.node_demand - net_injection
    balance_penalty = jnp.sum(jnp.square(balance_mismatch))

    # 3. Line Capacity Penalty
    # max(0, |flow| - capacity)^2
    flow_violation = jnp.maximum(0.0, jnp.abs(flows) - grid.line_capacity)
    capacity_penalty = jnp.sum(jnp.square(flow_violation))

    total_loss = cost + lambda_bal * balance_penalty + lambda_cap * capacity_penalty
    return total_loss
