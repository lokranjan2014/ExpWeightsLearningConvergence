import numpy as np
from games.exponential_weights import ExponentialWeights

def test_expected_utility():
    """Test expected_utility_for_action function"""
    ew = ExponentialWeights()
    
    # Test case 1: Simple payoff matrix
    # Payoff matrix structure: [opponent_action][player_action]
    # For player 1:
    payoff_p1 = np.array([
        # Player 1's action 0    Player 1's action 1
        [3, 0],  # Opponent plays action 0
        [0, 2]   # Opponent plays action 1
    ])
    # For player 2 (symmetric game):
    payoff_p2 = payoff_p1.T  # Same payoff structure as player 1
    
    # Test 1: Pure strategies
    print("=== Test 1: Pure Strategies ===")
    # Player 1's strategy: [1.0, 0.0] means 100% action 0
    # Player 2's strategy: [0.0, 1.0] means 100% action 1
    p1_probs = np.array([1.0, 0.0])
    p2_probs = np.array([0.0, 1.0])
    
    # Player 1's expected utilities when facing p2_probs [0,1]
    # payoff_p1[opponent_action=0][player_action] = [3, 0]
    # payoff_p1[opponent_action=1][player_action] = [0, 2]
    p1_util0 = ew.expected_utility_for_action(payoff_p1, p2_probs, 0)  # Should be 0*3 + 1*0 = 0
    p1_util1 = ew.expected_utility_for_action(payoff_p1, p2_probs, 1)  # Should be 0*0 + 1*2 = 2
    
    # Player 2's expected utilities when facing p1_probs [1,0]
    # payoff_p2[opponent_action=0][player_action] = [3, 0]
    # payoff_p2[opponent_action=1][player_action] = [0, 2]
    p2_util0 = ew.expected_utility_for_action(payoff_p2, p1_probs, 0)  # Should be 1*3 + 0*0 = 3
    p2_util1 = ew.expected_utility_for_action(payoff_p2, p1_probs, 1)  # Should be 1*0 + 0*2 = 0
    
    print("Player 1's expected utilities (vs opponent [0,1]):")
    print(f"Action 0: {p1_util0:.2f} (should be 0.0)")
    print(f"Action 1: {p1_util1:.2f} (should be 2.0)")
    
    print("\nPlayer 2's expected utilities (vs opponent [1,0]):")
    print(f"Action 0: {p2_util0:.2f} (should be 3.0)")
    print(f"Action 1: {p2_util1:.2f} (should be 0.0)")
    
    # Test 2: Mixed strategies
    print("\n=== Test 2: Mixed Strategies ===")
    p1_probs = np.array([0.6, 0.4])  # Player 1's strategy
    p2_probs = np.array([0.3, 0.7])  # Player 2's strategy
    
    # Player 1's expected utilities when facing p2_probs [0.3, 0.7]
    # For action 0: 0.3*3 + 0.7*0 = 0.9
    # For action 1: 0.3*0 + 0.7*2 = 1.4
    p1_util0 = ew.expected_utility_for_action(payoff_p1, p2_probs, 0)
    p1_util1 = ew.expected_utility_for_action(payoff_p1, p2_probs, 1)
    
    # Player 2's expected utilities when facing p1_probs [0.6, 0.4]
    # For action 0: 0.6*3 + 0.4*0 = 1.8
    # For action 1: 0.6*0 + 0.4*2 = 0.8
    p2_util0 = ew.expected_utility_for_action(payoff_p2, p1_probs, 0)
    p2_util1 = ew.expected_utility_for_action(payoff_p2, p1_probs, 1)
    
    print("Player 1's expected utilities (vs opponent [0.3, 0.7]):")
    print(f"Action 0: {p1_util0:.2f} (should be 3*0.3 + 0*0.7 = 0.9)")
    print(f"Action 1: {p1_util1:.2f} (should be 0*0.3 + 2*0.7 = 1.4)")
    
    print("\nPlayer 2's expected utilities (vs opponent [0.6, 0.4]):")
    print(f"Action 0: {p2_util0:.2f} (should be 3*0.6 + 0*0.4 = 1.8)")
    print(f"Action 1: {p2_util1:.2f} (should be 0*0.6 + 2*0.4 = 0.8)")

def test_ew_update():
    """Test ew_update function"""
    ew = ExponentialWeights()
    
    # Test case 1: Simple coordination game
    payoff = np.array([
        [1, 0],
        [0, 1]
    ])
    
    # Initial probabilities
    current_probs = np.array([0.5, 0.5])
    opponent_probs = np.array([0.5, 0.5])
    eta = 1.0
    
    print("\nTesting ew_update with coordination game:")
    print(f"Initial probs: {current_probs}")
    
    # Run a few updates
    for i in range(5):
        current_probs = ew.ew_update(current_probs, opponent_probs, payoff, eta)
        print(f"After update {i+1}: {current_probs}")
    
    # Test case 2: Hawk-Dove game
    payoff = np.array([
        [0, 3],  # (Hawk, Dove) = 3
        [1, 1]   # (Dove, Dove) = 1
    ])
    
    current_probs = np.array([0.5, 0.5])
    opponent_probs = np.array([0.5, 0.5])
    
    print("\nTesting ew_update with Hawk-Dove game:")
    print(f"Initial probs: {current_probs}")
    
    for i in range(5):
        current_probs = ew.ew_update(current_probs, opponent_probs, payoff, eta)
        print(f"After update {i+1}: {current_probs}")

if __name__ == "__main__":
    print("=== Testing expected_utility_for_action ===")
    test_expected_utility()
    
    print("\n=== Testing ew_update ===")
    test_ew_update()
