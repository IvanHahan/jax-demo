import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx

from src.grid import Grid
from src.optimization import OptimizationState
from src.physics import compute_power_flows


def plot_grid(
    grid: Grid,
    state: OptimizationState,
    title: str = "Power Grid",
    filename: str = None,
):
    """
    Plots the power grid state.

    Args:
        grid: Grid PyTree
        state: OptimizationState (contains generation and theta)
        title: Plot title
        filename: If provided, saves the plot to this file.
    """
    # 1. Create NetworkX graph
    G = nx.DiGraph()

    num_nodes = grid.node_demand.shape[0]
    num_edges = grid.line_from.shape[0]

    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)

    # 2. Compute Layout (on undirected version for stability)
    undirected_G = nx.Graph()
    for i in range(num_edges):
        undirected_G.add_edge(int(grid.line_from[i]), int(grid.line_to[i]))
    pos = nx.spring_layout(undirected_G, seed=42)

    # 3. Prepare Data for Plotting

    # Node Colors: Green (Gen), Red (Load), Gray (Neutral)
    node_colors = []
    node_sizes = []
    node_labels = {}

    theta, generation = state.params

    for i in range(num_nodes):
        is_gen = grid.node_is_generator[i]
        demand = grid.node_demand[i]
        gen = generation[i]

        if is_gen:
            node_colors.append("lightgreen")
            label = f"G:{gen:.1f}"
        elif demand > 0:
            node_colors.append("salmon")
            label = f"D:{demand:.1f}"
        else:
            node_colors.append("lightgray")
            label = ""

        node_labels[i] = f"{i}\n{label}"
        node_sizes.append(1000)

    # Edge Colors/Widths based on flow
    flows = compute_power_flows(grid, theta)
    edge_labels = {}

    # Add directed edges to G based on flow direction and store visual attributes
    for i in range(num_edges):
        u_ref = int(grid.line_from[i])
        v_ref = int(grid.line_to[i])
        flow = flows[i]
        capacity = float(grid.line_capacity[i])
        util = float(jnp.abs(flow) / capacity)

        # Determine actual flow direction for arrow
        if flow >= 0:
            u, v = u_ref, v_ref
        else:
            u, v = v_ref, u_ref

        # Color based on utilization
        if util > 1.0:
            color = "red"
        elif util > 0.8:
            color = "orange"
        else:
            color = "gray"

        width = 2.0 + 3.0 * util
        G.add_edge(u, v, color=color, width=width)
        edge_labels[(u, v)] = f"{float(jnp.abs(flow)):.1f}/{capacity:.0f}"

    # 4. Draw
    plt.figure(figsize=(12, 9))

    # Extract edge properties in the exact order G.edges() returns them to avoid mismatch
    edges = G.edges()
    colors = [G[u][v]["color"] for u, v in edges]
    widths = [G[u][v]["width"] for u, v in edges]

    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes, edgecolors="black"
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        edge_color=colors,
        width=widths,
        arrowstyle="->",
        arrowsize=20,
    )
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # 5. Add Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Generator (Producer)",
            markerfacecolor="lightgreen",
            markersize=12,
            markeredgecolor="black",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Load (Consumer)",
            markerfacecolor="salmon",
            markersize=12,
            markeredgecolor="black",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Transmission Hub",
            markerfacecolor="lightgray",
            markersize=12,
            markeredgecolor="black",
        ),
        Line2D([0], [0], color="gray", lw=2, label="Line (Flow OK)"),
        Line2D([0], [0], color="orange", lw=4, label="Line (High Load >80%)"),
        Line2D([0], [0], color="red", lw=4, label="Line (Overloaded >100%)"),
    ]
    plt.legend(handles=legend_elements, loc="upper right", title="Legend")

    plt.title(title)
    plt.axis("off")

    if filename:
        plt.savefig(filename)
        print(f"Saved plot to {filename}")

    plt.close()
