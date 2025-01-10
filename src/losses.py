import torch
import torch.nn as nn
import torch.nn.functional as F

class CircleLoss(nn.Module):
    def __init__(self, margin=0.25, gamma=64):
        super().__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, embA, embB, label):
        """
        embA, embB: (B,d)
        label: (B,) => 1 for positive, 0 for negative
        Circle approach: sum_{pos,neg} log(1+exp(gamma[(neg+m)-(pos-m)]))
        We'll do a simple in-batch approach for demonstration
        """
        embA = F.normalize(embA, dim=1)
        embB = F.normalize(embB, dim=1)
        s = torch.sum(embA*embB, dim=1) # (B,)
        pos_mask = (label>0.5).float()
        neg_mask = 1-pos_mask
        # pos s, neg s
        pos_s = s * pos_mask
        neg_s = s * neg_mask
        # We'll do a naive approach: sum of all positives vs. sum of all negatives
        pos_m = pos_s - self.margin
        neg_p = neg_s + self.margin
        # expand to pairwise (B,B) is also possible, but let's keep it simple
        # for demonstration we do single-batch approach
        # if no positive or negative, return 0
        pos_indices = (label>0.5).nonzero(as_tuple=True)[0]
        neg_indices = (label<0.5).nonzero(as_tuple=True)[0]
        if pos_indices.numel()==0 or neg_indices.numel()==0:
            return torch.tensor(0.0, requires_grad=True, device=embA.device)
        # just do a cross combination for demonstration:
        pos_vals = pos_m[pos_indices].unsqueeze(1) # (P,1)
        neg_vals = neg_p[neg_indices].unsqueeze(0) # (1,N)
        diff = neg_vals - pos_vals # (P,N)
        loss_mat = torch.log1p(torch.exp(self.gamma*diff))
        return loss_mat.mean()

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)

class SegmentAlignmentLoss(nn.Module):
    def __init__(self, margin=0.25, gamma=64):
        super().__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, anchor_segs, pos_segs, match_matrix):
        """
        anchor_segs, pos_segs: (K1, d), (K2, d)
        match_matrix: (K1, K2) => 1 if segments overlap, else 0
        For demonstration, we skip an advanced approach
        and just do a naive circle-like approach in-batch.
        """
        if anchor_segs.dim()==1:
            anchor_segs = anchor_segs.unsqueeze(0)
        if pos_segs.dim()==1:
            pos_segs = pos_segs.unsqueeze(0)
        # do pairwise sim => (K1,K2)
        anchor_norm = F.normalize(anchor_segs, dim=1)
        pos_norm = F.normalize(pos_segs, dim=1)
        sims = anchor_norm @ pos_norm.T
        pos_indices = (match_matrix>0.5)
        neg_indices = (match_matrix<0.5)
        if not pos_indices.any() or not neg_indices.any():
            return torch.tensor(0.0,requires_grad=True,device=anchor_segs.device)
        pos_vals = sims[pos_indices] - self.margin
        neg_vals = sims[neg_indices] + self.margin
        # cross
        # for demonstration
        diff = neg_vals.unsqueeze(1) - pos_vals.unsqueeze(0) # (neg_count, pos_count)
        loss_mat = torch.log1p(torch.exp(self.gamma*diff))
        return loss_mat.mean()
