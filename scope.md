# Scope: Differentiable Power Grid Optimization in JAX

## Purpose
Demonstrate JAX as a numerical optimization and differentiable programming tool on a real engineering problem: DC Optimal Power Flow (DC-OPF) with visualization and animation. The project avoids neural networks entirely.

## Problem Definition

### Inputs
- Power grid topology (graph)
- Node demands
- Generator cost functions
- Transmission line parameters and capacity limits

### Outputs
- Optimal generator outputs
- Node phase angles
- Line power flows
- Total generation cost
- Constraint violation metrics

### Objective
Minimize total generation cost while satisfying:
1.  Power balance at each node
2.  Transmission capacity limits

## Success Metrics
-   Correct implementation of DC Power Flow equations.
-   Successful optimization of generator outputs to minimize cost.
-   Visualization of the grid and power flows.
-   Clean, differentiable JAX implementation using PyTrees.
