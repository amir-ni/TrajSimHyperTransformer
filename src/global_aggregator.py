import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAggregator(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.attn_fc = nn.Linear(d_model, 1)

    def forward(self, seg_embs):
        """
        seg_embs: (K, d_model) or list => we shape to (1,K,d_model)
        Return: (d_model) => final global embedding
        """
        if isinstance(seg_embs, list):
            seg_embs = torch.stack(seg_embs, dim=0) # (K, d_model)
        seg_embs = seg_embs.unsqueeze(0) # => (1,K,d_model)
        out = self.transformer(seg_embs) # (1,K,d_model)
        scores = self.attn_fc(out).squeeze(-1) # (1,K)
        alpha = F.softmax(scores, dim=1)
        emb = torch.bmm(alpha.unsqueeze(1), out).squeeze(1) # (1,d_model)
        return emb.squeeze(0)
