import torch
import torch.nn as nn

class DynamicSegmentationGating(nn.Module):
    def __init__(self, d_in, threshold=0.5):
        super().__init__()
        self.fc = nn.Linear(d_in, 1)
        self.threshold = threshold

    def forward(self, z):
        """
        z: (B, L, d_in)
        We'll produce a boundary prob for each i=1..L-1
        Return a list of segments, each is a sub-route.
        NOTE: For simplicity, we do a soft approach here; 
        real gating can use Straight-Through or Gumbel for discrete boundary.
        """
        B, L, d = z.size()
        if L<=1:
            return [z] # trivial single segment
        # deltas
        deltas = z[:,1:,:] - z[:,:-1,:] # (B,L-1,d_in)
        gate_scores = self.fc(deltas).squeeze(-1) # (B,L-1)
        # gating prob
        gating_probs = torch.sigmoid(gate_scores) # (B,L-1)
        # We'll do a simple method: if gating_probs[i] > threshold => new segment
        segments = []
        for b in range(B):
            single = []
            start=0
            for i in range(L-1):
                if gating_probs[b,i]>self.threshold:
                    # boundary at i
                    seg = z[b, start:i+1,:] # (segment)
                    single.append(seg)
                    start=i+1
            # last
            seg = z[b,start:L,:]
            single.append(seg)
            segments.append(single)
        return segments
