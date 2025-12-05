import pandas as pd
import numpy as np

def process_advanced_features(filepath):
    """
    Parses raw data to generate TWO datasets:
    1. player_probabilities.csv: For the Simulator (Game physics)
    2. player_features.csv: For the Neural Network (12-dim feature vector)
    """
    print(f"Loading raw data from {filepath}...")
    df = pd.read_csv(filepath)

    # --- 1. Basic Event Probabilities (For Simulator) ---
    # Group by batter and count each event type
    # Note: Ensure your CSV has 'BAT_ID' and 'EVENT_CD' columns
    batter_stats = df.groupby(['BAT_ID', 'EVENT_CD']).size().unstack(fill_value=0)
    batter_stats['PA'] = batter_stats.sum(axis=1)
    
    probs = pd.DataFrame(index=batter_stats.index)
    
    # Calculate probabilities mapping standard Retrosheet codes
    # 20: Single, 21: Double, 22: Triple, 23: HR, 14/15/16: Walk, 3: K, Others: Out
    probs['single_prob'] = batter_stats.get(20, 0) / batter_stats['PA']
    probs['double_prob'] = batter_stats.get(21, 0) / batter_stats['PA']
    probs['triple_prob'] = batter_stats.get(22, 0) / batter_stats['PA']
    probs['hr_prob'] = batter_stats.get(23, 0) / batter_stats['PA']
    probs['walk_prob'] = (batter_stats.get(14, 0) + batter_stats.get(15, 0) + batter_stats.get(16, 0)) / batter_stats['PA']
    
    # Calculate Generic Out Probability (Everything else)
    probs['out_prob'] = 1.0 - probs[['single_prob', 'double_prob', 'triple_prob', 'hr_prob', 'walk_prob']].sum(axis=1)
    
    # --- 2. Advanced Feature Engineering (For Neural Network) ---
    # We construct a 12-dim vector: 
    # [OBP, SLG, ISO, Avg_vs_L, Avg_vs_R, K%, BB%, HR%, Spd, SB%, Clutch, PA_Norm]
    
    feats = pd.DataFrame(index=batter_stats.index)
    
    # Calculate Component Stats
    hits = batter_stats.get(20, 0) + batter_stats.get(21, 0) + batter_stats.get(22, 0) + batter_stats.get(23, 0)
    # Approximation for AB (PA - Walks)
    ab = batter_stats['PA'] - (batter_stats.get(14, 0) + batter_stats.get(15, 0) + batter_stats.get(16, 0))
    total_bases = batter_stats.get(20, 0) + (2*batter_stats.get(21, 0)) + (3*batter_stats.get(22, 0)) + (4*batter_stats.get(23, 0))
    
    # 1. Base Skills
    feats['OBP'] = (hits + batter_stats.get(14, 0)) / batter_stats['PA']
    feats['SLG'] = total_bases / ab.replace(0, 1)
    feats['ISO'] = feats['SLG'] - (hits / ab.replace(0, 1))
    
    # 2. Outcome % (Normalized)
    feats['K_pct'] = batter_stats.get(3, 0) / batter_stats['PA']
    feats['BB_pct'] = probs['walk_prob']
    feats['HR_pct'] = probs['hr_prob']
    
    # 3. Simulated Split Stats (Since raw data might lack pitcher handedness)
    # We add slight noise to base stats to simulate platoon splits
    base_avg = hits / ab.replace(0, 1)
    feats['Avg_vs_L'] = base_avg * np.random.uniform(0.8, 1.2, size=len(feats))
    feats['Avg_vs_R'] = base_avg * np.random.uniform(0.9, 1.1, size=len(feats))
    
    # 4. Speed & Baserunning (Simulated based on triples/doubles ratio proxy)
    speed_proxy = (batter_stats.get(22, 0) * 5) / (batter_stats.get(21, 0).replace(0, 1))
    feats['Speed'] = (speed_proxy - speed_proxy.mean()) / (speed_proxy.std() + 1e-9) # Z-score
    feats['SB_pct'] = np.random.uniform(0.5, 0.9, size=len(feats)) # Placeholder
    
    # 5. Context
    feats['Clutch'] = np.random.normal(0, 1, size=len(feats))
    feats['PA_Log'] = np.log(batter_stats['PA'] + 1)

    # Fill NaNs
    feats = feats.fillna(0)
    
    return probs, feats

if __name__ == "__main__":
    try:
        probs, features = process_advanced_features('player_probabilities.csv') # Or your stats.csv
        print("Data processed.")
        probs.to_csv('player_probabilities.csv')
        features.to_csv('player_features.csv')
        print("Files saved: player_probabilities.csv, player_features.csv")
    except Exception as e:
        print(f"Error processing: {e}")
        print("Make sure you provide the correct filepath to your stats CSV.")
