import torch
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    """
    A dataset that returns single trajectories (for offline negative sampling).
    """
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.traj_map = {}
        for i in range(len(df)):
            row = df.iloc[i]
            tid = row['trajectory_id']
            pts = row['points']
            self.traj_map[tid] = pts

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tid = row['trajectory_id']
        pts = self.traj_map[tid]
        return tid, torch.tensor(pts, dtype=torch.float)

def collate_fn(batch):
    """
    Collate function that just returns a list of (tid, points).
    Used for offline negative sampling or embedding.
    """
    tids = [b[0] for b in batch]
    points_list = [b[1] for b in batch]
    return tids, points_list
