# Handoff

## Context
Project kick-off. Goal is to build a Differentiable Power Grid Optimization demo using JAX.

## Current State
-   **Completed**: Core logic, Optimization, Visualization, Demo Script.
-   **Verified**: All tests passing (`pytest tests`).
-   **Demo**: Run `python src/demo.py` to see the results.

## Next Steps
1.  Explore JAX-Opt for more advanced solvers (e.g., L-BFGS-B) to handle hard constraints better.
2.  Add more complex grids (IEEE 14-bus, etc.).
3.  Implement AC Power Flow (non-linear).
