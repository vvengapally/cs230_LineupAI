import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rl_environment import LineupEnv

class PolicyNetwork(nn.Module):
    def __init__(self, num_players, feature_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        
        # Input: Features of ONE player (feature_dim) + Current Slot One-Hot (9)
        # We append the slot encoding to every player's feature vector
        input_dim = feature_dim + 9 
        
        # We process each player independently to get a "score" (logit)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1) # Output a single score per player

    def forward(self, features, slot, mask):
        """
        features: Tensor (N, 12)
        slot: Int (0-8) representing current lineup position
        mask: Tensor (N,) where 1 means already selected
        """
        N = features.shape[0]
        
        # Create one-hot encoding for the current slot
        slot_vec = torch.zeros(9)
        if slot < 9:
            slot_vec[slot] = 1.0
        slot_vec = slot_vec.repeat(N, 1) # Repeat for all players -> (N, 9)
        
        # Concatenate: Player Features + Context (Slot)
        x = torch.cat([features, slot_vec], dim=1) # (N, 12+9)
        
        # Forward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        scores = self.fc3(x).squeeze() # (N,) logits
        

        scores = scores.masked_fill(mask.bool(), -1e9)
        
        return torch.softmax(scores, dim=0)

# --- Training Loop ---

def train_reinforce():

    try:
        env = LineupEnv('player_probabilities.csv', 'player_features.csv')
    except FileNotFoundError:
        print("Error: Data files not found. Run 'advanced_data_parser.py' first.")
        return

    # Initialize Policy Network
    policy = PolicyNetwork(env.num_players, env.feature_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    num_episodes = 5000
    running_reward = 0
    reward_history = []

    print(f"Starting REINFORCE Training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        
        # Generate one lineup (Episode)
        for t in range(9):
            # Prepare Tensors
            feats_t = torch.FloatTensor(state['features'])
            mask_t = torch.FloatTensor(state['mask'])
            slot_t = state['slot']
            
            # Forward Pass
            probs = policy(feats_t, slot_t, mask_t)
            
            # Sample Action (Stochastic Policy)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            # Step Env
            state, reward, done = env.step(action.item())
            
            # Store log probability for gradient calculation
            log_probs.append(dist.log_prob(action))
            
            if done:
                rewards.append(reward) # Reward only comes at the end
        
        # Calculate Loss
        R = rewards[0]
        reward_history.append(R)
        
        # Exponential moving average for baseline
        running_reward = 0.05 * R + (1 - 0.05) * running_reward
        
        # REINFORCE Update: -log_prob * (R - baseline)
        # Baseline is running_average to reduce variance
        baseline = running_reward if episode > 0 else R
        loss = []
        for log_prob in log_probs:
            loss.append(-log_prob * (R - baseline))
            
        optimizer.zero_grad()
        sum(loss).backward()
        optimizer.step()
        
        # Logging
        if episode % 100 == 0:
            print(f"Episode {episode}\tAvg Runs: {running_reward:.3f}")

    # Save Model
    torch.save(policy.state_dict(), 'reinforce_agent.pth')
    print("Training Complete. Model saved as 'reinforce_agent.pth'.")

if __name__ == "__main__":
    train_reinforce()
