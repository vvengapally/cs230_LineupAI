import numpy as np
import pandas as pd
import os

class BaseballSimulator:
    def __init__(self, player_probs):
        # player_probs is the DataFrame loaded from CSV
        self.player_probs = player_probs
        self.outcomes = ['single', 'double', 'triple', 'hr', 'walk', 'out']

    def simulate_at_bat(self, batter_id):
        """Simulates a single at-bat for a given batter."""
        try:
            probs = self.player_probs.loc[batter_id]
            weights = [
                probs['single_prob'], probs['double_prob'], probs['triple_prob'],
                probs['hr_prob'], probs['walk_prob'], probs['out_prob']
            ]
            # Ensure weights sum to exactly 1.0 due to potential floating point issues
            weights = np.array(weights)
            weights /= weights.sum()

            return np.random.choice(self.outcomes, p=weights)
        except KeyError:
            # Fallback for unknown player: league average (rough approximation)
            return np.random.choice(self.outcomes, p=[0.15, 0.05, 0.005, 0.03, 0.08, 0.685])

    def advance_runners(self, bases, hit_type):
        """
        Advances runners based on the hit type.
        Simplified logic: runners always advance the same number of bases as the hit.
        Returns updated bases and runs scored on the play.
        """
        runs = 0
        new_bases = [0, 0, 0] # [1st, 2nd, 3rd]

        if hit_type == 'walk':
            # Forced advancement logic
            new_bases = list(bases)
            if bases[0] == 1:
                if bases[1] == 1:
                    if bases[2] == 1:
                        runs += 1 # Bases loaded walk
                    new_bases[2] = 1
                new_bases[1] = 1
            new_bases[0] = 1
            return new_bases, runs

        advance = {'single': 1, 'double': 2, 'triple': 3, 'hr': 4}[hit_type]

        for i in range(2, -1, -1): # Check 3rd, then 2nd, then 1st
            if bases[i] == 1:
                if i + advance >= 3:
                    runs += 1 # Runner scores
                else:
                    new_bases[i + advance] = 1 # Runner advances

        if hit_type != 'hr':
             new_bases[advance - 1] = 1 # Batter reaches base
        else:
            runs += 1 # Batter scores on HR

        return new_bases, runs

    def simulate_inning(self, lineup, lineup_idx):
        """Simulates one half-inning."""
        outs = 0
        runs_in_inning = 0
        bases = [0, 0, 0] # [1st base, 2nd base, 3rd base] (0=empty, 1=occupied)

        while outs < 3:
            batter = lineup[lineup_idx]
            outcome = self.simulate_at_bat(batter)

            if outcome == 'out':
                outs += 1
            else:
                bases, runs = self.advance_runners(bases, outcome)
                runs_in_inning += runs

            lineup_idx = (lineup_idx + 1) % 9 # Move to next batter

        return runs_in_inning, lineup_idx

    def simulate_game(self, lineup, num_innings=9):
        """Simulates a full game for one team's lineup."""
        total_runs = 0
        current_batter_idx = 0
        for _ in range(num_innings):
            runs, current_batter_idx = self.simulate_inning(lineup, current_batter_idx)
            total_runs += runs
        return total_runs

def load_player_data(filepath='player_probabilities.csv'):
    """Loads player probabilities from a CSV file."""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Please run retrosheet_parser.py first.")
        return None

    df = pd.read_csv(filepath, index_col=0) # Assuming first column is player ID
    required_cols = ['single_prob', 'double_prob', 'triple_prob', 'hr_prob', 'walk_prob', 'out_prob']
    if not all(col in df.columns for col in required_cols):
         print(f"Error: CSV is missing required columns: {required_cols}")
         return None
    return df

# 1. Load real probability data
player_probs = load_player_data()

if player_probs is not None:
    # 2. Initialize Simulator
    sim = BaseballSimulator(player_probs)

    # 3. Define a lineup using actual player IDs from your CSV
    available_players = player_probs.index.tolist()
    if len(available_players) < 9:
        print("Not enough players in CSV to make a lineup.")
    else:
        my_lineup = available_players[:9]
        print(f"Simulating with lineup: {my_lineup}")

        # 4. Run a single simulation
        runs_scored = sim.simulate_game(my_lineup)
        print(f"Game Over. Total Runs: {runs_scored}")

        # 5. Run Monte Carlo simulation for baseline
        num_games = 1000
        print(f"Running {num_games} Monte Carlo simulations...")
        results = [sim.simulate_game(my_lineup) for _ in range(num_games)]
        print(f"Average Runs over {num_games} games: {np.mean(results):.2f}")

