# JAX Power Grid Optimization

A differentiable implementation of DC Optimal Power Flow (DC-OPF) using JAX. This project demonstrates how JAX can be used for numerical engineering optimization and differentiable programming without involving neural networks.

## What is a Power Grid?

In this context, a **Power Grid** is modeled as a network (graph) of:
- **Nodes (Buses)**: Locations where power is either produced (Generators) or consumed (Loads/Demand).
- **Edges (Transmission Lines)**: Connections between nodes that transport electricity.

### Key Concepts
- **Active Power ($P$)**: Represented in **MW (Megawatts)**. This is the actual power being generated or consumed.
- **Phase Angle ($\theta$)**: Represented in **Radians**.
- **Net Injection ($P_{inj}$)**: The total amount of power **leaving** a node through transmission lines. It is the sum of flows on all lines connected to the node (Flow Out - Flow In). A positive injection means the node is "exporting" power to the grid, while a negative injection means it is "importing" power.

## Generation Cost Model

For simplicity, we model generation cost as a **linear function** for each generator $i$:

$$Cost_i(g_i) = c_i \cdot g_i$$

### What does the parameter mean? (Units in $)
- **$c \cdot g$ (Linear Term)**: Represents the **Variable Cost** (fuel, labor). The coefficient $c$ is in **$/MWh**.
- **Total System Cost**: The sum of costs across all generators, represented in **$/h (Dollars per hour)**.

By minimizing this function, the system naturally prefers cheaper generators (small $c$), but must still respect transmission constraints.

## Power Pooling: Who powers whom?

In a mesh power grid, consumers are **not** directly connected to specific generators. Instead:
- **The Mesh Effect**: Multiple generators contribute power to the grid simultaneously.
- ** Kirchhoff's Laws**: Electricity flows based on physics (phase angles and susceptance), not business contracts. 
- **Aggregate Supply**: A consumer at any given node is effectively powered by a "pool" of all generators. The specific mix of power they receive depends on the grid's topology and which lines have the least "electrical resistance" (impedance).

The optimizer's role is to find the most cost-effective "mix" of generation that keeps this pool in equilibrium while respecting the physical limits of every transmission line.

## Line Capacity: The Grid's Speed Limit

Every transmission line has a **Thermal Capacity ($C$)**, measured in **MW**. 

### Why is there a limit?
Electricity generates heat as it travels through wires. If a line carries more power than it was designed for, it will overheat. In the real world, this causes wires to physically sag (potentially causing fires or short circuits) or even melt.

### Why do capacities vary?
Not all lines are built the same:
- **Wire Thickness**: Thicker wires have less resistance and can carry more current before they start heating up.
- **Voltage Level**: High-voltage transmission lines (the "trunk" lines) are designed to move massive amounts of power over long distances, whereas local lines involve smaller equipment.
- **Age and Material**: Older lines or those made of different alloys may have lower thermal thresholds.
- **Environment**: A line in a windy, cold area can actually carry more power than the exact same line in a hot, stagnant desert because the wind helps cool the wire down.

### How it affects Optimization
Line capacity is the primary reason why we cannot always use the cheapest power plant. If the cheapest generator is far away and the lines connecting it to the city are already "full" (at capacity), the optimizer must:
1.  **Stop increasing** output from the cheap plant.
2.  **Start increasing** output from a more expensive plant that is closer to the consumer or connected via less congested lines.

In the visualization, lines that are nearing their capacity are colored **Orange**, and those exceeding it are colored **Red**.

## Input Parameters

To simulate the grid, we define:
1. **Topology**: Which nodes are connected to each other (the graph structure).
2. **Nodal Demands**: How much power each "Load" node needs (e.g., a city or factory).
- **Generator Costs**: For each generator, we define a linear cost function ( $Cost = c \cdot g$ ).
4. **Line Limits**:
   - **Susceptance ($B$)**: Physics parameter for flow calculation.
   - **Thermal Capacity ($C$)**: The maximum safe power flow a line can handle before overheating.

## What are we Optimizing?

This is a **DC Optimal Power Flow (DC-OPF)** problem. We are searching for the optimal values of two sets of variables:

1. **Generation Levels ($g_i$)**: How much power each plant should produce.
2. **Phase Angles ($\theta_i$)**: The physical state of the grid that causes power to flow from generators to loads.

### The Objective
**Minimize Total Generation Cost** while strictly adhering to:
1. **Power Balance**: Every node must have exactly enough supply (generation + incoming flow) to meet its demand (consumption + outgoing flow).
2. **Capacity Constraints**: No transmission line can carry more power than its thermal limit.

### Why JAX?
By expressing these physical constraints as **differentiable penalties** in a loss function, we can use JAX's automatic differentiation to "see" how changing a generator's output will affect line congestion and costs across the entire network, allowing us to converge on the most efficient solution.

## Mathematical Objective (The Loss Function)

The `total_loss` optimized by JAX is the sum of three distinct components. It balances financial cost against physical constraints using a penalty method.

### 1. Generation Cost
- **Purpose**: Minimize the total money spent on producing electricity.
- **Logic**: Uses the linear cost model $\sum (c_i g_i)$. The optimizer naturally tries to drive this to zero, which is countered by the penalties below.

### 2. Power Balance Penalty
- **Purpose**: Ensure the grid is stable and every consumer's demand is met.
- **Logic**: We calculate the mismatch at every node: $Mismatch = Generation - Demand - NetInjection$. 
- **Penalty**: We square the mismatch and multiply it by a large weight ($\lambda_{bal}$). This forces the optimizer to find a state where the Net Injection (export/import) perfectly balances the local Generation vs. Demand.

#### Why not just minimize $Generation - Demand$?
If we only minimized $Gen - Dem$, the optimizer would force every node to be an "island" that produces exactly what it consumes. Under that logic, a power plant node (High Gen, No Demand) and a house node (No Gen, High Demand) would both be considered "unbalanced."

Including **Net Injection** allows the nodes to trade. It says: "The power at this node is balanced if the local production minus the local consumption equals the amount of power we export into the grid."

### 3. Line Capacity Penalty
- **Purpose**: Prevent transmission lines from overheating or sagging.
- **Logic**: If the $|Flow|$ on a line is greater than its $Capacity$, we calculate the difference. If it's less, the penalty is zero.
- **Penalty**: Any exceedance is squared and multiplied by a high weight ($\lambda_{cap}$). This "soft constraint" allows the optimizer to momentarily "explore" overloaded states while searching for a valid path, but ultimately penalizes them heavily in the final solution.


## Tech Stack

- **Core**: JAX (Numerical computation, Auto-Diff, JIT)
- **Graph Logic**: NetworkX
- **Visualization**: Matplotlib
- **Testing**: Pytest

## Getting Started

### Installation

1. Ensure you have a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Demo

Run the main demo script with different complexity levels:
```bash
# Default (Medium 5-node grid)
python src/demo.py

# Simple 3-node grid
python src/demo.py --mode simple

# Complex 8-node mesh grid
python src/demo.py --mode complex

# Large 20-node random grid
python src/demo.py --mode large

# Run all scenarios sequentially
python src/demo.py --mode all
```

#### Visualization Legend
- **Green Nodes**: Producers (Generators).
- **Red/Salmon Nodes**: Consumers (Loads/Demand).
- **Gray Nodes**: Transmission hubs (no local Gen/Load).
- **Edge Thickness**: Represents power flow volume.
- **Edge Color**: Red/Orange indicates a line approaching or exceeding its thermal capacity.

This will produce three files:
- `initial_grid.png`: The cold-start state of the grid.
- `final_grid.png`: The optimized state with flows and generation adjusted.
- `loss_history.png`: A plot showing the convergence of the optimizer.

### Running Tests

To verify the physics and optimization logic:
```bash
pytest tests
```

## Project Structure

- `src/grid.py`: PyTree definitions and grid creation utilities.
- `src/physics.py`: Power flow equations and loss function.
- `src/optimization.py`: Gradient descent loop and JIT-compiled update steps.
- `src/visualization.py`: Utilities for plotting the grid state.
- `src/demo.py`: End-to-end 5-node demo.
