import pandas as pd
import numpy as np

def load_and_process_retrosheet(filepath):
    """
    Loads a CSV of Retrosheet play-by-play data and calculates
    event probabilities for each batter.
    """
    # This is a simplified view. Real Retrosheet CSVs have dozens of columns.
    # Key columns we'd need: 'BAT_ID', 'EVENT_TX', 'EVENT_CD'
    # EVENT_CD values (standard Retrosheet):
    # 2: Generic Out, 3: Strikeout, 14: Walk, 15: Intentional Walk, 16: HBP,
    # 20: Single, 21: Double, 22: Triple, 23: Home Run

    print(f"Loading data from {filepath}...")
    # In a real scenario, you'd load your actual large CSV here.
    df = pd.read_csv(filepath)

    # MOCK DATA for demonstration purposes
    #data = {
     #   'BAT_ID': ['player1']*100 + ['player2']*100 + ['player3']*100,
      #  'EVENT_CD': np.random.choice([2, 3, 14, 20, 21, 22, 23], 300, p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.02, 0.03])
    #}
    #df = pd.DataFrame(data)

    # Group by batter and count each event type
    batter_stats = df.groupby(['BAT_ID', 'EVENT_CD']).size().unstack(fill_value=0)

    # Calculate total plate appearances (PA)
    batter_stats['PA'] = batter_stats.sum(axis=1)

    # Calculate probabilities for each outcome
    # We map Retrosheet codes to our simulator's simplified outcomes
    batter_probs = pd.DataFrame(index=batter_stats.index)
    batter_probs['single_prob'] = batter_stats.get(20, 0) / batter_stats['PA']
    batter_probs['double_prob'] = batter_stats.get(21, 0) / batter_stats['PA']
    batter_probs['triple_prob'] = batter_stats.get(22, 0) / batter_stats['PA']
    batter_probs['hr_prob'] = batter_stats.get(23, 0) / batter_stats['PA']
    batter_probs['walk_prob'] = (batter_stats.get(14, 0) + batter_stats.get(15, 0) + batter_stats.get(16, 0)) / batter_stats['PA']
    batter_probs['out_prob'] = 1.0 - batter_probs[['single_prob', 'double_prob', 'triple_prob', 'hr_prob', 'walk_prob']].sum(axis=1)

    return batter_probs

# Example Usage:
player_probs = load_and_process_retrosheet('stats.csv')
# print (batter_probs)
# print(player_probs.head())
player_probs.to_csv('player_probabilities.csv') # Save for the simulator to use
