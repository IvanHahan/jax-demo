# Design: Differentiable Power Grid Optimization in JAX

## Architecture

The project uses JAX for differentiable programming. Data is structured using JAX PyTrees for efficient processing.

### Data Structures (PyTrees)

**Grid**:
```python
class Grid(NamedTuple):
    node_demand: jnp.ndarray      # [N] Active power demand
    node_is_generator: jnp.ndarray # [N] Boolean mask
    gen_cost: jnp.ndarray         # [N] Linear cost (c*g)
    line_from: jnp.ndarray        # [E] Start node index
    line_to: jnp.ndarray          # [E] End node index
    line_susceptance: jnp.ndarray # [E] Line susceptance (B)
    line_capacity: jnp.ndarray    # [E] Thermal capacity limit
```

**OptimizationState**:
```python
class OptimizationState(NamedTuple):
    theta: jnp.ndarray            # [N] Phase angles
    generation: jnp.ndarray       # [N] Power generation
```

## Mathematical Model

### DC Power Flow
$$f_{ij} = B_{ij} (\theta_i - \theta_j)$$

### Penalized Objective (Loss Function)
$$ \mathcal{L} = \sum_i c_i(g_i) + \lambda_{bal} \sum_i \| g_i - d_i - \sum_j f_{ij} \|^2 + \lambda_{cap} \sum_{e} \max(0, |f_e| - C_e)^2 $$

## Implementation Details

- **Reference Angle**: $\theta_0$ is fixed to $0.0$ to remove rotational invariance in the phase angle solution.
- **Gradient Clipping**: Gradients are clipped to a maximum norm of $100.0$ to ensure stability during early optimization steps.
- **Projections**: Generation values are projected to be non-negative and zeroed out for non-generator nodes.
- **Visualization**: Integrated with `networkx` for layout and `matplotlib` for rendering grid states and flows.

## Tech Stack
- **Core**: Python 3.10+, JAX
- **Graph Ops**: NetworkX
- **Plotting**: Matplotlib
- **Testing**: Pytest
