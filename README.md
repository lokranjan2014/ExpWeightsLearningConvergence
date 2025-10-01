# 2x2 Games - Exponential Weights Implementation

This directory contains a clean, well-organized implementation of Algorithm 1 (Exponential Weights) for 2×2 symmetric games, computing the shorthand functional Δ(t) = p(t)ε₁ + (1-p(t))ε₂ and producing plots for different sign cases and initialization types.

## Structure

```
2x2Games/
├── games/
│   ├── __init__.py              # Module exports
│   ├── exponential_weights.py   # Main implementation classes
│   └── helpers.py               # Additional utility functions
├── cases_output/                # Directory for generated plots
├── 2x2_ExponentialWeights_Simulation.ipynb  # Jupyter notebook for running simulations
└── README.md                    # This file
```

## Features

### Core Implementation (`exponential_weights.py`)

- **ExponentialWeights**: Main class implementing Algorithm 1
  - `build_payoff_matrix()`: Create payoff matrices from parameters a, b, c, d
  - `compute_epsilons()`: Calculate ε₁ = a - b, ε₂ = c - d
  - `ew_update()`: Single Exponential Weights update step
  - `run_simulation()`: Complete simulation with history tracking
  - `plot_simulation_results()`: Generate plots for probabilities and delta trajectories

- **InitializationStrategies**: Helper class for different initialization methods
  - `init_any()`: Random non-pure initialization
  - `init_identical()`: Identical initialization for both players
  - `init_same_sign()`: Initializations with same-sign delta values
  - `init_opposite_sign()`: Initializations with opposite-sign delta values

- **GameCases**: Predefined game cases from Table 3
  - All sign combinations: (+,+), (-,-), (-,+), (+,-), (0,+), (0,-)
  - Each case includes payoff matrices and epsilon values

- **SimulationRunner**: High-level interface for running comprehensive simulations
  - `run_case_simulation()`: Run single case with all initialization strategies
  - `run_all_simulations()`: Run all predefined cases
  - `run_custom_case()`: Run custom payoff matrices

### Additional Utilities (`helpers.py`)

- **Convergence Analysis**:
  - `compute_convergence_metrics()`: Calculate convergence statistics
  - `analyze_equilibrium_convergence()`: Analyze Nash equilibrium convergence
  - `create_summary_table()`: Generate summary tables of results

- **Visualization**:
  - `plot_convergence_comparison()`: Compare convergence across initialization strategies

## Usage

### Quick Start

1. **Run the Jupyter notebook**:
   ```bash
   jupyter notebook 2x2_ExponentialWeights_Simulation.ipynb
   ```

2. **Or use Python directly**:
   ```python
   from games import SimulationRunner
   
   # Initialize runner (creates cases_output directory)
   runner = SimulationRunner(output_dir="cases_output")
   
   # Run all predefined cases
   runner.run_all_simulations(eta=1.0, T=2000)
   ```

### Advanced Usage

```python
from games import ExponentialWeights, InitializationStrategies, GameCases

# Create custom payoff matrix
ew = ExponentialWeights(seed=42)
A = ew.build_payoff_matrix(a=1.0, b=0.0, c=0.5, d=0.5)

# Get epsilon values
eps1, eps2 = ew.compute_epsilons(A)

# Choose initialization strategy
init_strategies = InitializationStrategies()
p1_init, p2_init = init_strategies.init_same_sign(eps1, eps2)

# Run simulation
history = ew.run_simulation(A, p1_init, p2_init, eta=0.8, T=1000)

# Plot results
ew.plot_simulation_results(
    history, "Custom_Case", "Same-sign", eps1, eps2, eta=0.8
)
```

## Predefined Cases

The implementation includes all cases from Table 3:

| Case | ε₁ | ε₂ | Description |
|------|----|----|-------------|
| (+,+) | 1.0 | 1.0 | Both epsilons positive |
| (-,-) | -1.0 | -1.0 | Both epsilons negative |
| (-,+) | -1.0 | 1.0 | Mixed signs |
| (+,-) | 1.0 | -1.0 | Mixed signs |
| (0,+) | 0.0 | 1.0 | One epsilon zero |
| (0,-) | 0.0 | -1.0 | One epsilon zero |

## Initialization Strategies

1. **Any (default)**: Random non-pure initialization satisfying Assumption 1
2. **Identical**: Both players start with identical distributions
3. **Same-sign**: Initializations constructed to have same-sign delta values
4. **Opposite-sign**: Initializations constructed to have opposite-sign delta values

## Output

The simulation generates:

- **Interactive plots** showing probability trajectories and delta functionals
- **Saved plots** in the `cases_output/` directory (PNG format, 300 DPI)
- **Convergence analysis** with metrics and equilibrium detection
- **Summary tables** comparing results across cases

## Parameters

- **eta (η)**: Learning rate/step size
  - For mixed cases (-,+) and (+,-), try smaller values (e.g., η < 8/(|ε₁| + |ε₂|))
- **T**: Number of simulation rounds (default: 2000)
- **record_every**: Frequency of history recording (default: 1)

## Dependencies

- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `jupyter`: For notebook interface

## Notes

- The implementation follows Algorithm 1 exactly with synchronous updates
- Numerical stability is ensured through log-space computations
- All plots include proper labels, legends, and grid lines
- The code is well-documented with type hints and docstrings
- Results are reproducible with seed control

## Example Output

Running the simulations will generate plots showing:

1. **Probability trajectories**: How P(θ₁) evolves for both players
2. **Delta trajectories**: How Δ(t) = p(t)ε₁ + (1-p(t))ε₂ evolves
3. **Convergence analysis**: Whether players converge to pure or mixed strategies
4. **Parameter sensitivity**: Effect of different learning rates

The generated plots provide comprehensive visualizations for all sign cases and initialization types discussed in the research paper.
