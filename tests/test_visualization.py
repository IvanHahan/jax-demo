import os

import matplotlib.pyplot as plt

from src.grid import create_simple_grid
from src.optimization import initialize_optimization
from src.visualization import plot_grid


def test_plot_grid_runs_and_saves():
    grid = create_simple_grid()
    state = initialize_optimization(grid)
    filename = "test_plot.png"

    # Ensure cleanup
    if os.path.exists(filename):
        os.remove(filename)

    try:
        plot_grid(grid, state, "Test Plot", filename)
        assert os.path.exists(filename), "Plot file was not created"
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_plot_grid_no_filename():
    """Test that it runs without error even if not saving"""
    grid = create_simple_grid()
    state = initialize_optimization(grid)
    # Should not raise
    plot_grid(grid, state, "Test Plot No Save")
    plt.close("all")
