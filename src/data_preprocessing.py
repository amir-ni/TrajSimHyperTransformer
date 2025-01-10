import pandas as pd
import numpy as np

def load_synthetic_data(num_trajs=1000, max_len=50, seed=42):
    """
    Generates a synthetic dataset of random trajectories (x, y, t).
    Some partial overlaps are introduced for demonstration.
    """
    np.random.seed(seed)
    data = []
    for tid in range(num_trajs):
        length = np.random.randint(10, max_len)
        x = np.cumsum(np.random.randn(length)) + np.random.randint(0,100)
        y = np.cumsum(np.random.randn(length)) + np.random.randint(0,100)
        t = np.arange(length)
        # random partial overlap: let ~20% of them share some sub-route
        if tid % 5 == 0 and tid>0:
            # overlap with tid-1 for half the length
            overlap_len = length//2
            x[:overlap_len] = data[-1]['x'][-overlap_len:]
            y[:overlap_len] = data[-1]['y'][-overlap_len:]
        data.append({
            'tid': tid,
            'x': x,
            'y': y,
            't': t
        })
    return data

def convert_to_dataframe(data):
    """
    For convenience, create a dataframe that has a list of (x,y,t) for each trajectory_id.
    """
    rows = []
    for item in data:
        rows.append({
            'trajectory_id': item['tid'],
            'points': np.stack([item['x'], item['y'], item['t']], axis=1).tolist()
        })
    df = pd.DataFrame(rows)
    return df
