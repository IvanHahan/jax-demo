# Project Tracker

## Active Tasks
- [ ] **T-001: Project Setup** - Initialize repo, dependencies, and file structure.
- [ ] **T-002: Data Structures** - Implement JAX PyTrees for Grid, Nodes, Lines.
- [ ] **T-003: DC Power Flow Model** - Implement physics equations in JAX.
- [ ] **T-004: Loss Function** - Implement the penalized objective function.
- [ ] **T-005: Optimization Loop** - Implement gradient descent loop using JAX `grad` and `jit`.
- [ ] **T-006: Visualization** - Create basic grid visualization.

## Acceptance Criteria
- **T-001**: `requirements.txt` exists, `src` folder structure created.
- **T-002**: `Grid` PyTree can be instantiated and flattened/unflattened by JAX.
- **T-003**: `power_flow` function returns correct flows given theta.
- **T-004**: Loss decreases during optimization steps.
- **T-005**: Converges to optimal solution for simple test case.
