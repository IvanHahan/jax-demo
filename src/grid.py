from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class Grid(NamedTuple):
    """
    JAX PyTree representing the power grid.

    Attributes:
        node_demand: [N] Active power demand at each node (positive = load).
        node_is_generator: [N] Boolean mask, true if node has a generator.
        gen_cost_a: [N] Quadratic cost coefficient (0 for non-generators).
        gen_cost_b: [N] Linear cost coefficient (0 for non-generators).
        line_from: [E] Index of source node for each line.
        line_to: [E] Index of target node for each line.
        line_susceptance: [E] Susceptance (B) of each line (usually 1/reactance).
        line_capacity: [E] Thermal capacity limit of each line.
    """

    node_demand: jnp.ndarray
    node_is_generator: jnp.ndarray
    gen_cost_a: jnp.ndarray
    gen_cost_b: jnp.ndarray
    line_from: jnp.ndarray
    line_to: jnp.ndarray
    line_susceptance: jnp.ndarray
    line_capacity: jnp.ndarray


def create_simple_grid() -> Grid:
    """
    Creates a simple 3-node test grid.
    Node 0: Generator (Cheap)
    Node 1: Generator (Expensive)
    Node 2: Load
    """
    # 3 Nodes
    node_demand = jnp.array([0.0, 0.0, 100.0])  # Node 2 needs 100 MW
    node_is_generator = jnp.array([True, True, False])

    # Cost functions:
    # Gen 0: 0.01*g^2 + 10*g
    # Gen 1: 0.02*g^2 + 20*g
    gen_cost_a = jnp.array([0.01, 0.02, 0.0])
    gen_cost_b = jnp.array([10.0, 20.0, 0.0])

    # 3 Lines connecting all nodes (triangle)
    # 0-1, 1-2, 2-0
    line_from = jnp.array([0, 1, 2])
    line_to = jnp.array([1, 2, 0])
    line_susceptance = jnp.array([10.0, 10.0, 10.0])
    line_capacity = jnp.array([50.0, 50.0, 50.0])

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


def create_8_node_complex_grid() -> Grid:
    """
    Creates an 8-node mesh grid with more complex routing and bottlenecks.
    """
    # 8 Nodes
    # 0, 1: Cheap Generators
    # 2: Mid-range Generator
    # 3, 4, 5, 6, 7: Loads
    node_demand = jnp.array([0.0, 0.0, 0.0, 40.0, 50.0, 30.0, 60.0, 20.0])
    node_is_generator = jnp.array([True, True, True, False, False, False, False, False])

    # Costs
    gen_cost_a = jnp.array([0.005, 0.01, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    gen_cost_b = jnp.array([5.0, 15.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Edges (Mesh topology)
    edges = [
        (0, 3),
        (0, 4),
        (1, 4),
        (1, 5),
        (2, 6),
        (2, 7),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 3),
    ]
    line_from = jnp.array([e[0] for e in edges])
    line_to = jnp.array([e[1] for e in edges])

    line_susceptance = jnp.full(len(edges), 10.0)
    # Varied capacities to create interesting congestion
    line_capacity = jnp.array(
        [
            80.0,
            30.0,
            60.0,
            80.0,
            40.0,
            50.0,  # Source lines
            30.0,
            30.0,
            30.0,
            30.0,
            30.0,  # Mesh connections
        ]
    )

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


def create_20_node_large_grid(seed: int = 42) -> Grid:
    """
    Creates a 20-node grid using a Watts-Strogatz small-world model.
    This simulates a more complex topology with both local and long-distance connections.
    """
    import networkx as nx

    # Generate a small-world graph (20 nodes, each connected to 4 neighbors, p=0.2 rewiring)
    G = nx.watts_strogatz_graph(n=20, k=4, p=0.2, seed=seed)

    # 20 Nodes
    # Let's say: 4 Generators (at nodes 0, 5, 10, 15)
    # The rest are nodes with random demand
    node_is_generator = (
        jnp.zeros(20, dtype=bool).at[jnp.array([0, 5, 10, 15])].set(True)
    )

    # Costs
    # Gen 0: Very cheap
    # Gen 5: Mid
    # Gen 10: Mid
    # Gen 15: Expensive
    gen_cost_a = (
        jnp.zeros(20)
        .at[jnp.array([0, 5, 10, 15])]
        .set(jnp.array([0.002, 0.01, 0.01, 0.04]))
    )
    gen_cost_b = (
        jnp.zeros(20)
        .at[jnp.array([0, 5, 10, 15])]
        .set(jnp.array([5.0, 15.0, 20.0, 50.0]))
    )

    # Random demands for non-generators (mean ~15 MW)
    np.random.seed(seed)
    demands = np.random.uniform(5.0, 25.0, 20)
    # Set generator nodes to 0 demand for simplicity
    demands[node_is_generator] = 0.0
    node_demand = jnp.array(demands)

    # Edges
    line_from = jnp.array([u for u, v in G.edges()])
    line_to = jnp.array([v for u, v in G.edges()])

    num_edges = len(G.edges())
    line_susceptance = jnp.full(num_edges, 10.0)
    line_capacity = jnp.full(num_edges, 60.0)  # Tighten capacity to force optimization

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
