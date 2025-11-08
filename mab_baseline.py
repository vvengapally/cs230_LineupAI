import numpy as np
import math
import pandas as pd
import os

class LineupBandit:
    def __init__(self, player_ids, num_slots=9):
        self.player_ids = player_ids
        self.num_players = len(player_ids)
        self.num_slots = num_slots
        # Q_values[slot_idx][player_idx] = average reward for this player in this slot
        self.q_values = np.zeros((num_slots, self.num_players))
        # N_counts[slot_idx][player_idx] = number of times this player chosen for this slot
        self.n_counts = np.zeros((num_slots, self.num_players))
        self.total_games = 0

    def select_lineup(self, c_exploration=1.0):
        """
        Selects a lineup using the UCB1 algorithm, ensuring no duplicate players.
        """
        available_players = list(range(self.num_players))
        lineup_indices = []

        for slot in range(self.num_slots):
            best_score = -float('inf')
            best_player_idx = -1

            for player_idx in available_players:
                # UCB1 calculation
                if self.n_counts[slot][player_idx] == 0:
                     # If a player hasn't been tried in this slot, give them infinite priority to ensure exploration
                    score = float('inf')
                else:
                    exploitation = self.q_values[slot][player_idx]
                    exploration = c_exploration * math.sqrt(math.log(self.total_games + 1) / self.n_counts[slot][player_idx])
                    score = exploitation + exploration

                if score > best_score:
                    best_score = score
                    best_player_idx = player_idx

            # Select the best player for this slot and remove from available pool
            lineup_indices.append(best_player_idx)
            available_players.remove(best_player_idx)

        return [self.player_ids[i] for i in lineup_indices], lineup_indices

    def update(self, lineup_indices, reward):
        """
        Updates Q-values and counts based on the game's reward (runs scored).
        """
        self.total_games += 1
        for slot, player_idx in enumerate(lineup_indices):
            self.n_counts[slot][player_idx] += 1
            n = self.n_counts[slot][player_idx]
            # Incremental average update
            self.q_values[slot][player_idx] += (reward - self.q_values[slot][player_idx]) / n

def load_player_ids(filepath='player_probabilities.csv'):
    """Loads player IDs from the CSV file."""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return []
    df = pd.read_csv(filepath, index_col=0)
    return df.index.tolist()

# --- FULL USAGE EXAMPLE (incorporating simulator) ---
from simple_baseball_sim import BaseballSimulator, load_player_data

# 1. Load Data
player_probs = load_player_data()
all_player_ids = load_player_ids()

if player_probs is not None and len(all_player_ids) >= 9:
    # 2. Initialize Simulator and Bandit
    sim = BaseballSimulator(player_probs)
    bandit = LineupBandit(all_player_ids)

    # 3. Training Loop
    num_episodes = 1000
    print(f"Training MAB for {num_episodes} games...")
    for i in range(num_episodes):
        lineup, indices = bandit.select_lineup(c_exploration=1.0)
        runs = sim.simulate_game(lineup)
        bandit.update(indices, runs)
        if (i+1) % 100 == 0:
            print(f"Episode {i+1}/{num_episodes} complete")

    # 4. Get Best Lineup (Exploitation only)
    best_lineup, _ = bandit.select_lineup(c_exploration=0.0)
    print("\n--- Training Complete ---")
    print("Best MAB Lineup found:", best_lineup)

    # 5. Evaluate Best Lineup
    eval_games = 500
    results = [sim.simulate_game(best_lineup) for _ in range(eval_games)]
    print(f"Average Runs for Best Lineup (over {eval_games} games): {np.mean(results):.2f}")
else:
    print("Not enough players found in player_probabilities.csv to run MAB baseline.")
