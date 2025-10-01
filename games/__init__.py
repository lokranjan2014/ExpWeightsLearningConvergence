"""
2x2 Games Module

This module provides implementations for 2x2 symmetric games including
Exponential Weights algorithm simulations.
"""

from .exponential_weights import (
    ExponentialWeights,
    InitializationStrategies,
    GameCases,
    SimulationRunner
)

from .helpers import (
    compute_convergence_metrics,
    plot_convergence_comparison,
    analyze_equilibrium_convergence,
    create_summary_table
)

__all__ = [
    'ExponentialWeights',
    'InitializationStrategies', 
    'GameCases',
    'SimulationRunner',
    'compute_convergence_metrics',
    'plot_convergence_comparison',
    'analyze_equilibrium_convergence',
    'create_summary_table'
]
