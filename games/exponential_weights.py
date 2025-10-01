"""
Exponential Weights Algorithm for 2x2 Symmetric Games

This module implements Algorithm 1 (Exponential Weights) for 2x2 symmetric games,
computing the shorthand functional Δ(t) = p(t)ε₁ + (1-p(t))ε₂ and producing
plots for different sign cases and initialization types.

Author: Research Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import os


class ExponentialWeights:
    """
    Exponential Weights algorithm implementation for 2x2 symmetric games.
    
    This class provides methods to:
    - Build payoff matrices from parameters a, b, c, d
    - Compute epsilon values ε₁ = a - b, ε₂ = c - d
    - Run EW simulations with different initialization strategies
    - Generate plots for probabilities and delta trajectories
    """
    
    def __init__(self, seed: int = 0):
        """Initialize the ExponentialWeights class with a random seed."""
        np.random.seed(seed)
        self.seed = seed
    
    @staticmethod
    def build_payoff_matrix(a: float, b: float, c: float, d: float) -> np.ndarray:
        """
        Build payoff matrix for Player 1: A = [[a, b], [c, d]]
        Player 2's payoff is A^T (symmetry).
        
        Args:
            a, b, c, d: Payoff matrix parameters
            
        Returns:
            2x2 numpy array representing the payoff matrix
        """
        return np.array([[a, b], [c, d]], dtype=float)
    
    @staticmethod
    def compute_epsilons(payoff_matrix: np.ndarray) -> Tuple[float, float]:
        """
        Compute epsilon values from payoff matrix.
        
        Args:
            payoff_matrix: 2x2 payoff matrix
            
        Returns:
            Tuple of (eps1, eps2) where eps1 = a - b, eps2 = c - d
        """
        a, b = payoff_matrix[0, 0], payoff_matrix[0, 1]
        c, d = payoff_matrix[1, 0], payoff_matrix[1, 1]
        eps1 = a - b
        eps2 = c - d
        return eps1, eps2
    
    @staticmethod
    def expected_utility_for_action(payoff_matrix: np.ndarray, 
                                  opponent_probs: np.ndarray, 
                                  action_index: int) -> float:
        """
        Compute expected utility of playing action_index against opponent distribution.
        
        Args:
            payoff_matrix: 2x2 payoff matrix where:
                          - First index: opponent's action
                          - Second index: player's action
            opponent_probs: Probability vector [p(theta1), p(theta2)] for opponent
            action_index: Action to evaluate (0 or 1)
            
        Returns:
            Expected utility of playing action_index
        """
        # Payoff matrix is indexed as [opponent_action][player_action]
        # So for player's action_index, we look at that column across all opponent actions
        return (payoff_matrix[0, action_index] * opponent_probs[0] + 
                payoff_matrix[1, action_index] * opponent_probs[1])
    
    def ew_update(self, 
                 current_probs: np.ndarray, 
                 opponent_probs: np.ndarray, 
                 payoff_matrix: np.ndarray, 
                 eta: float) -> np.ndarray:
        """
        Single Exponential Weights update for a player.
        
        Args:
            current_probs: Current probability distribution [p(theta1), p(theta2)]
            opponent_probs: Opponent's probability distribution
            payoff_matrix: 2x2 payoff matrix for the player
            eta: Learning rate (step size)
            
        Returns:
            Updated probability distribution
        """
        # Compute expected utilities for each action
        u0 = self.expected_utility_for_action(payoff_matrix, opponent_probs, 0)
        u1 = self.expected_utility_for_action(payoff_matrix, opponent_probs, 1)
        
        # Update in log-space for numerical stability
        logits = np.array([
            np.log(max(current_probs[0], 1e-18)) + eta * u0,
            np.log(max(current_probs[1], 1e-18)) + eta * u1
        ])
        
        # Normalize in log-space
        max_log = np.max(logits)
        weights = np.exp(logits - max_log)
        new_probs = weights / np.sum(weights)
        
        return new_probs
    
    def run_simulation(self, 
                     payoff_matrix: np.ndarray,
                     p1_init: np.ndarray,
                     p2_init: np.ndarray,
                     eta: float = 1.0,
                     T: int = 2000,
                     record_every: int = 1) -> Dict[str, List]:
        """
        Run Exponential Weights simulation for T rounds.
        
        Args:
            payoff_matrix: 2x2 payoff matrix
            p1_init, p2_init: Initial probability distributions (must sum to 1)
            eta: Learning rate
            T: Number of rounds
            record_every: Record history every N rounds
            
        Returns:
            Dictionary containing simulation history
        """
        p1 = np.array(p1_init, dtype=float)
        p2 = np.array(p2_init, dtype=float)
        
        eps1, eps2 = self.compute_epsilons(payoff_matrix)
        
        history = {
            'p1': [],
            'p2': [],
            'delta1': [],
            'delta2': [],
            'rounds': []
        }
        
        for t in range(T):
            if t % record_every == 0:
                history['p1'].append(p1.copy())
                history['p2'].append(p2.copy())
                history['rounds'].append(t)
                
                # Compute delta functionals: Δ(t) = p(t)ε₁ + (1-p(t))ε₂
                delta1 = p1[0] * eps1 + p1[1] * eps2
                delta2 = p2[0] * eps1 + p2[1] * eps2
                history['delta1'].append(delta1)
                history['delta2'].append(delta2)
            
            # Synchronous update
            new_p1 = self.ew_update(p1, p2, payoff_matrix, eta)
            new_p2 = self.ew_update(p2, p1, payoff_matrix, eta)
            p1, p2 = new_p1, new_p2
        
        return history
    
    def plot_simulation_results(self, 
                              history: Dict[str, List],
                              case_name: str,
                              init_name: str,
                              eps1: float,
                              eps2: float,
                              eta: float,
                              p1_init: Optional[np.ndarray] = None,
                              p2_init: Optional[np.ndarray] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Plot simulation results showing probabilities only.
        
        Args:
            history: Simulation history from run_simulation
            case_name: Name of the case (e.g., "(+,+)")
            init_name: Name of initialization strategy
            eps1, eps2: Epsilon values
            eta: Learning rate used
            p1_init, p2_init: Initial probability distributions (optional)
            save_path: Optional path to save the plot
        """
        rounds = np.array(history['rounds'])
        p1_theta1 = np.array(history['p1'])[:, 0]  # Probability of theta1 for player 1
        p1_theta2 = np.array(history['p1'])[:, 1]  # Probability of theta2 for player 1
        p2_theta1 = np.array(history['p2'])[:, 0]  # Probability of theta1 for player 2
        p2_theta2 = np.array(history['p2'])[:, 1]  # Probability of theta2 for player 2
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot probabilities with different colors for players and markers for actions
        # Player 1: blue, Player 2: red
        # θ₁: circles, θ₂: squares (with dotted line)
        markevery = max(1, len(rounds)//20)
        
        # Plot θ₁ (action 1) with solid lines
        ax.plot(rounds, p1_theta1, label='Player 1 P(θ₁)', linewidth=2, color='blue', 
                marker='o', markersize=3, markevery=markevery, alpha=0.8)
        ax.plot(rounds, p2_theta1, label='Player 2 P(θ₁)', linewidth=2, color='red',
                marker='o', markersize=3, markevery=markevery, alpha=0.8)
        
        # Plot θ₂ (action 2) with dotted lines
        ax.plot(rounds, p1_theta2, label='Player 1 P(θ₂)', linewidth=2, color='blue',
                marker='s', markersize=3, markevery=markevery, alpha=0.8, 
                linestyle=':')
        ax.plot(rounds, p2_theta2, label='Player 2 P(θ₂)', linewidth=2, color='red',
                marker='s', markersize=3, markevery=markevery, alpha=0.8,
                linestyle=':')
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_xlabel('Round', fontsize=12)
        
        # Create title with initialization details
        title = f'{case_name} — {init_name} — η={eta} — ε₁={eps1:.3f}, ε₂={eps2:.3f}'
        if p1_init is not None and p2_init is not None:
            init_details = f' | Init: P1({p1_init[0]:.2f},{p1_init[1]:.2f}), P2({p2_init[0]:.2f},{p2_init[1]:.2f})'
            title += init_details
            
        ax.set_title(title, fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()


class InitializationStrategies:
    """Helper class for different initialization strategies."""
    
    @staticmethod
    def init_any() -> Tuple[np.ndarray, np.ndarray]:
        """Random non-pure initialization satisfying Assumption 1."""
        return np.array([0.6, 0.4]), np.array([0.45, 0.55])
    
    @staticmethod
    def init_identical() -> Tuple[np.ndarray, np.ndarray]:
        """Identical initialization for both players."""
        p = np.array([0.7, 0.3])
        return p.copy(), p.copy()
    
    @staticmethod
    def init_same_sign(eps1: float, eps2: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create initializations so that delta values have the same sign.
        
        Args:
            eps1, eps2: Epsilon values to determine target sign
            
        Returns:
            Tuple of (p1_init, p2_init) with same-sign deltas
        """
        target_sign = np.sign(eps1 + eps2) if (eps1 + eps2) != 0 else np.sign(eps1)
        if target_sign == 0:
            target_sign = 1
        
        if target_sign > 0:
            p1 = np.array([0.8, 0.2])
            p2 = np.array([0.7, 0.3])
        else:
            p1 = np.array([0.3, 0.7])
            p2 = np.array([0.4, 0.6])
        
        return p1, p2
    
    @staticmethod
    def init_opposite_sign(eps1: float, eps2: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create initializations so that delta values have opposite signs.
        
        Args:
            eps1, eps2: Epsilon values (used for validation)
            
        Returns:
            Tuple of (p1_init, p2_init) with opposite-sign deltas
        """
        p1 = np.array([0.85, 0.15])
        p2 = np.array([0.15, 0.85])
        return p1, p2


class GameCases:
    """Predefined game cases from Table 3."""
    
    @staticmethod
    def get_all_cases() -> Dict[str, np.ndarray]:
        """
        Get all predefined game cases with their payoff matrices.
        
        Returns:
            Dictionary mapping case names to payoff matrices
        """
        cases = {
            "(+,+)": ExponentialWeights.build_payoff_matrix(a=1.0, b=0.0, c=1.0, d=0.0),   # eps1=1, eps2=1
            "(-,-)": ExponentialWeights.build_payoff_matrix(a=0.0, b=1.0, c=0.0, d=1.0),   # eps1=-1, eps2=-1
            "(-,+)": ExponentialWeights.build_payoff_matrix(a=0.0, b=1.0, c=1.0, d=0.0),   # eps1=-1, eps2=1
            "(+,-)": ExponentialWeights.build_payoff_matrix(a=1.0, b=0.0, c=0.0, d=1.0),   # eps1=1, eps2=-1
            "(0,+)": ExponentialWeights.build_payoff_matrix(a=0.5, b=0.5, c=1.0, d=0.0),   # eps1=0, eps2=1
            "(0,-)": ExponentialWeights.build_payoff_matrix(a=0.5, b=0.5, c=0.0, d=1.0),   # eps1=0, eps2=-1
        }
        return cases


class SimulationRunner:
    """Main class for running comprehensive simulations."""
    
    def __init__(self, output_dir: str = "cases_output"):
        """
        Initialize the simulation runner.
        
        Args:
            output_dir: Directory to save output plots
        """
        self.output_dir = output_dir
        self.ew = ExponentialWeights()
        self.init_strategies = InitializationStrategies()
        self.game_cases = GameCases()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def run_case_simulation(self, 
                           case_name: str,
                           payoff_matrix: np.ndarray,
                           eta: float = 1.0,
                           T: int = 2000,
                           save_plots: bool = True) -> None:
        """
        Run simulation for a specific case with all initialization strategies.
        
        Args:
            case_name: Name of the case
            payoff_matrix: 2x2 payoff matrix
            eta: Learning rate
            T: Number of rounds
            save_plots: Whether to save plots to files
        """
        eps1, eps2 = self.ew.compute_epsilons(payoff_matrix)
        print(f"\n{'='*60}")
        print(f"Case {case_name}: ε₁={eps1:.3f}, ε₂={eps2:.3f}")
        print(f"{'='*60}")
        
        # Prepare initialization strategies
        inits = {
            'Any (default)': self.init_strategies.init_any(),
            'Identical': self.init_strategies.init_identical(),
            'Same-sign': self.init_strategies.init_same_sign(eps1, eps2),
            'Opposite-sign': self.init_strategies.init_opposite_sign(eps1, eps2)
        }
        
        # Run simulations for each initialization
        for init_name, (p1_init, p2_init) in inits.items():
            print(f"\nRunning simulation: {init_name}")
            
            history = self.ew.run_simulation(
                payoff_matrix, p1_init, p2_init, eta=eta, T=T
            )
            
            # Determine save path
            save_path = None
            if save_plots:
                # Include eta in filename for higher step size runs
                eta_suffix = f"_eta{eta}" if eta != 1.0 else ""
                filename = f"{case_name}_{init_name.replace(' ', '_').replace('(', '').replace(')', '')}{eta_suffix}.png"
                save_path = os.path.join(self.output_dir, filename)
            
            # Plot results
            self.ew.plot_simulation_results(
                history, case_name, init_name, eps1, eps2, eta, 
                p1_init=p1_init, p2_init=p2_init, save_path=save_path
            )
    
    def run_all_simulations(self, eta: float = 1.0, T: int = 2000) -> None:
        """
        Run simulations for all predefined cases.
        
        Args:
            eta: Learning rate
            T: Number of rounds
        """
        cases = self.game_cases.get_all_cases()
        
        for case_name, payoff_matrix in cases.items():
            # For mixed cases, also try larger eta
            if case_name in ["(-,+)", "(+,-)"]:
                print(f"\n{'='*80}")
                print(f"Running {case_name} with larger eta for mixed-NE regime")
                print(f"{'='*80}")
                self.run_case_simulation(case_name, payoff_matrix, eta=8.2, T=T)
            
            # Run with standard eta
            self.run_case_simulation(case_name, payoff_matrix, eta=eta, T=T)
    
    def run_custom_case(self, 
                       case_name: str,
                       a: float, b: float, c: float, d: float,
                       eta: float = 1.0,
                       T: int = 2000) -> None:
        """
        Run simulation for a custom case.
        
        Args:
            case_name: Name for the case
            a, b, c, d: Payoff matrix parameters
            eta: Learning rate
            T: Number of rounds
        """
        payoff_matrix = self.ew.build_payoff_matrix(a, b, c, d)
        self.run_case_simulation(case_name, payoff_matrix, eta=eta, T=T)
