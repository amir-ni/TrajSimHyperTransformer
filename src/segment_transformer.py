import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentMiniTransformer(nn.Module):
    def __init__(self, d_model, nhead=2, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.attn_fc = nn.Linear(d_model, 1)

    def forward(self, seg):
        """
        seg: (L_s, d_model) single segment
        Return (d_model) => segment embedding
        """
        if seg.dim()==2:
            seg = seg.unsqueeze(0) # => (1, L_s, d_model)
        # pad_mask?
        out = self.transformer(seg) # (1, L_s, d_model)
        # attention pooling
        scores = self.attn_fc(out).squeeze(-1) # (1,L_s)
        alpha = F.softmax(scores, dim=1)        # (1,L_s)
        emb = torch.bmm(alpha.unsqueeze(1), out).squeeze(1) # (1,d_model)
        return emb.squeeze(0) # (d_model)
