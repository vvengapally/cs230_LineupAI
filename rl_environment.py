import numpy as np
import pandas as pd
# Assumes simple_baseball_sim.py is in the same directory
from simple_baseball_sim import BaseballSimulator 

class LineupEnv:
    def __init__(self, player_probs_path='player_probabilities.csv', player_features_path='player_features.csv'):
        # Load Data
        self.probs_df = pd.read_csv(player_probs_path, index_col=0)
        self.features_df = pd.read_csv(player_features_path, index_col=0)
        
        # Ensure alignment (intersection of players in both files)
        common_players = self.probs_df.index.intersection(self.features_df.index)
        self.probs_df = self.probs_df.loc[common_players]
        self.features_df = self.features_df.loc[common_players]
        
        self.player_ids = common_players.tolist()
        self.num_players = len(self.player_ids)
        self.feature_dim = self.features_df.shape[1]
        
        # Initialize internal simulator
        self.sim = BaseballSimulator(self.probs_df)
        
        self.reset()

    def reset(self):
        """Resets the environment for a new lineup construction."""
        self.current_slot = 0
        self.selected_indices = [] # Track chosen players
        self.mask = np.zeros(self.num_players) # 0 = available, 1 = taken
        return self._get_state()

    def _get_state(self):
        """
        Constructs the state vector.
        We return a dictionary so the Policy Network can handle the parts (Features, Mask) separately.
        """
        return {
            'features': self.features_df.values, # Matrix (N, 12)
            'mask': self.mask,                   # Vector (N,)
            'slot': self.current_slot            # Scalar (Int)
        }

    def step(self, action_idx):
        """
        Action: Selecting a player index for the current slot.
        Returns: next_state, reward, done
        """
        if self.mask[action_idx] == 1:
            # In a robust RL loop, we mask actions so this shouldn't happen,
            # but good to have a check.
            raise ValueError(f"Agent selected an unavailable player (Index {action_idx})!")
            
        # Register selection
        self.selected_indices.append(action_idx)
        self.mask[action_idx] = 1
        self.current_slot += 1
        
        done = (self.current_slot == 9)
        reward = 0
        
        if done:
            # Construct the lineup string list based on selected indices
            final_lineup = [self.player_ids[i] for i in self.selected_indices]
            
            # Simulate games to get reward 
            # We simulate 50 games and take the average to reduce variance for the reward signal
            rewards = [self.sim.simulate_game(final_lineup) for _ in range(50)]
            reward = np.mean(rewards)
            
        return self._get_state(), reward, done
