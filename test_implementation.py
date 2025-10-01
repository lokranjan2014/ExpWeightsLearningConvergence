#!/usr/bin/env python3
"""
Test script for 2x2 Games Exponential Weights implementation.

This script runs a quick test to verify that the implementation works correctly.
"""

import sys
import os
sys.path.append('games')

from exponential_weights import ExponentialWeights, SimulationRunner, GameCases

def test_basic_functionality():
    """Test basic functionality of the implementation."""
    print("Testing 2x2 Games Exponential Weights Implementation")
    print("=" * 60)
    
    # Test 1: Basic class instantiation
    print("1. Testing class instantiation...")
    ew = ExponentialWeights(seed=42)
    runner = SimulationRunner(output_dir="test_output")
    cases = GameCases.get_all_cases()
    print("   ✓ All classes instantiated successfully")
    
    # Test 2: Payoff matrix creation and epsilon computation
    print("2. Testing payoff matrix and epsilon computation...")
    A = ew.build_payoff_matrix(a=1.0, b=0.0, c=1.0, d=0.0)
    eps1, eps2 = ew.compute_epsilons(A)
    expected_eps1, expected_eps2 = 1.0, 1.0
    assert abs(eps1 - expected_eps1) < 1e-10, f"Expected eps1={expected_eps1}, got {eps1}"
    assert abs(eps2 - expected_eps2) < 1e-10, f"Expected eps2={expected_eps2}, got {eps2}"
    print("   ✓ Payoff matrix and epsilon computation working correctly")
    
    # Test 3: Single simulation run
    print("3. Testing single simulation run...")
    from exponential_weights import InitializationStrategies
    init_strategies = InitializationStrategies()
    p1_init, p2_init = init_strategies.init_identical()
    
    history = ew.run_simulation(A, p1_init, p2_init, eta=1.0, T=100)
    assert len(history['p1']) == 100, f"Expected 100 history entries, got {len(history['p1'])}"
    assert len(history['delta1']) == 100, f"Expected 100 delta entries, got {len(history['delta1'])}"
    print("   ✓ Single simulation run completed successfully")
    
    # Test 4: Predefined cases
    print("4. Testing predefined cases...")
    all_cases = GameCases.get_all_cases()
    expected_cases = ["(+,+)", "(-,-)", "(-,+)", "(+,-)", "(0,+)", "(0,-)"]
    for case_name in expected_cases:
        assert case_name in all_cases, f"Missing case: {case_name}"
    print("   ✓ All predefined cases available")
    
    # Test 5: Output directory creation
    print("5. Testing output directory creation...")
    assert os.path.exists("test_output"), "Output directory not created"
    print("   ✓ Output directory created successfully")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("The implementation is working correctly.")
    print("=" * 60)

if __name__ == "__main__":
    test_basic_functionality()
