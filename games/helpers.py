"""
Helper utilities for 2x2 games simulations.

This module provides additional utility functions for analysis and visualization
of 2x2 symmetric games.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


def compute_convergence_metrics(history: Dict[str, List], 
                              window_size: int = 100) -> Dict[str, float]:
    """
    Compute convergence metrics from simulation history.
    
    Args:
        history: Simulation history from ExponentialWeights.run_simulation
        window_size: Window size for computing convergence metrics
        
    Returns:
        Dictionary containing convergence metrics
    """
    p1_theta1 = np.array(history['p1'])[:, 0]
    p2_theta1 = np.array(history['p2'])[:, 0]
    delta1 = np.array(history['delta1'])
    delta2 = np.array(history['delta2'])
    
    # Compute final values
    final_p1 = p1_theta1[-1]
    final_p2 = p2_theta1[-1]
    final_delta1 = delta1[-1]
    final_delta2 = delta2[-1]
    
    # Compute convergence (variance in last window)
    if len(p1_theta1) >= window_size:
        p1_var = np.var(p1_theta1[-window_size:])
        p2_var = np.var(p2_theta1[-window_size:])
        delta1_var = np.var(delta1[-window_size:])
        delta2_var = np.var(delta2[-window_size:])
    else:
        p1_var = np.var(p1_theta1)
        p2_var = np.var(p2_theta1)
        delta1_var = np.var(delta1)
        delta2_var = np.var(delta2)
    
    return {
        'final_p1': final_p1,
        'final_p2': final_p2,
        'final_delta1': final_delta1,
        'final_delta2': final_delta2,
        'p1_convergence': p1_var,
        'p2_convergence': p2_var,
        'delta1_convergence': delta1_var,
        'delta2_convergence': delta2_var
    }


def plot_convergence_comparison(histories: Dict[str, Dict[str, List]], 
                               case_name: str,
                               initializations: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
                               save_path: Optional[str] = None) -> None:
    """
    Plot convergence comparison across different initialization strategies.
    
    Args:
        histories: Dictionary mapping init names to simulation histories
        case_name: Name of the case being analyzed
        initializations: Optional dictionary mapping init names to (p1_init, p2_init) tuples
        save_path: Optional path to save the plot
    """
    n_inits = len(histories)
    fig, axes = plt.subplots(2, n_inits, figsize=(4*n_inits, 8))
    
    if n_inits == 1:
        axes = axes.reshape(2, 1)
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_inits))
    
    for i, (init_name, history) in enumerate(histories.items()):
        rounds = np.array(history['rounds'])
        p1_theta1 = np.array(history['p1'])[:, 0]
        p1_theta2 = np.array(history['p1'])[:, 1]
        p2_theta1 = np.array(history['p2'])[:, 0]
        p2_theta2 = np.array(history['p2'])[:, 1]
        delta1 = np.array(history['delta1'])
        delta2 = np.array(history['delta2'])
        
        # Plot probabilities with different colors for players and markers for actions
        # Player 1: blue, Player 2: red
        # θ₁: circles, θ₂: squares
        markevery = max(1, len(rounds)//15)  # Show markers every 15th point
        axes[0, i].plot(rounds, p1_theta1, label='Player 1 P(θ₁)', 
                        color='blue', linewidth=2, marker='o', markersize=2, 
                        markevery=markevery, alpha=0.8)
        axes[0, i].plot(rounds, p2_theta1, label='Player 2 P(θ₁)', 
                        color='red', linewidth=2, marker='o', 
                        markersize=2, markevery=markevery, alpha=0.8)
        axes[0, i].plot(rounds, p1_theta2, label='Player 1 P(θ₂)', 
                        color='blue', linewidth=2, marker='s', markersize=2, 
                        markevery=markevery, alpha=0.8)
        axes[0, i].plot(rounds, p2_theta2, label='Player 2 P(θ₂)', 
                        color='red', linewidth=2, marker='s', 
                        markersize=2, markevery=markevery, alpha=0.8)
        # Create title with initialization details if available
        title = init_name
        if initializations and init_name in initializations:
            p1_init, p2_init = initializations[init_name]
            title += f'\nP1({p1_init[0]:.2f},{p1_init[1]:.2f}) P2({p2_init[0]:.2f},{p2_init[1]:.2f})'
        axes[0, i].set_title(title, fontsize=10)
        axes[0, i].set_ylabel('Probability', fontsize=10)
        axes[0, i].legend(fontsize=8)
        axes[0, i].grid(True, alpha=0.3)
        
        # Plot deltas
        axes[1, i].plot(rounds, delta1, label='Δ₁', 
                       color=colors[i], linewidth=2)
        axes[1, i].plot(rounds, delta2, label='Δ₂', 
                       color=colors[i], linewidth=2, linestyle='--')
        axes[1, i].axhline(0, color='k', linewidth=0.5, alpha=0.7)
        axes[1, i].set_title(f'{init_name}', fontsize=12)
        axes[1, i].set_ylabel('Δ(t)', fontsize=10)
        axes[1, i].set_xlabel('Round', fontsize=10)
        axes[1, i].legend(fontsize=8)
        axes[1, i].grid(True, alpha=0.3)
    
    plt.suptitle(f'Convergence Comparison: {case_name}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence comparison plot saved to: {save_path}")
    
    plt.show()


def analyze_equilibrium_convergence(history: Dict[str, List], 
                                   payoff_matrix: np.ndarray,
                                   tolerance: float = 1e-3) -> Dict[str, any]:
    """
    Analyze convergence to Nash equilibrium.
    
    Args:
        history: Simulation history
        payoff_matrix: 2x2 payoff matrix
        tolerance: Convergence tolerance
        
    Returns:
        Dictionary containing equilibrium analysis
    """
    eps1, eps2 = ExponentialWeights.compute_epsilons(payoff_matrix)
    
    # Compute final distributions
    final_p1 = np.array(history['p1'])[-1]
    final_p2 = np.array(history['p2'])[-1]
    
    # Check if converged to pure strategies
    pure_threshold = 0.95
    p1_pure = final_p1[0] > pure_threshold or final_p1[0] < (1 - pure_threshold)
    p2_pure = final_p2[0] > pure_threshold or final_p2[0] < (1 - pure_threshold)
    
    # Check if converged to mixed strategy
    mixed_threshold = 0.1
    p1_mixed = abs(final_p1[0] - 0.5) < mixed_threshold
    p2_mixed = abs(final_p2[0] - 0.5) < mixed_threshold
    
    # Compute convergence metrics
    metrics = compute_convergence_metrics(history)
    
    return {
        'final_distributions': {
            'p1': final_p1,
            'p2': final_p2
        },
        'convergence_type': {
            'p1_pure': p1_pure,
            'p2_pure': p2_pure,
            'p1_mixed': p1_mixed,
            'p2_mixed': p2_mixed
        },
        'epsilons': {
            'eps1': eps1,
            'eps2': eps2
        },
        'convergence_metrics': metrics
    }


def create_summary_table(results: Dict[str, Dict[str, any]]) -> None:
    """
    Create a summary table of simulation results.
    
    Args:
        results: Dictionary mapping case names to analysis results
    """
    print("Simulation Results Summary")
    print("=" * 80)
    print(f"{'Case':<10} {'ε₁':<8} {'ε₂':<8} {'P1 Final':<12} {'P2 Final':<12} {'Convergence':<15}")
    print("-" * 80)
    
    for case_name, result in results.items():
        eps1 = result['epsilons']['eps1']
        eps2 = result['epsilons']['eps2']
        p1_final = result['final_distributions']['p1'][0]
        p2_final = result['final_distributions']['p2'][0]
        
        # Determine convergence type
        conv_type = "Mixed"
        if result['convergence_type']['p1_pure'] and result['convergence_type']['p2_pure']:
            conv_type = "Pure"
        elif result['convergence_type']['p1_mixed'] and result['convergence_type']['p2_mixed']:
            conv_type = "Mixed"
        else:
            conv_type = "Mixed/Pure"
        
        print(f"{case_name:<10} {eps1:<8.2f} {eps2:<8.2f} {p1_final:<12.3f} {p2_final:<12.3f} {conv_type:<15}")
    
    print("=" * 80)
