import torch
import torch.nn as nn

from fourier_spatio_temporal import FourierSpatioTemporalEncoder
from gating_network import DynamicSegmentationGating
from segment_transformer import SegmentMiniTransformer
from global_aggregator import GlobalAggregator

class MultiLevelHyperTransformer(nn.Module):
    def __init__(self,
                 d_out=128,
                 r_s=16,
                 r_t=8,
                 gating_threshold=0.5,
                 seg_transformer_layers=2,
                 seg_transformer_heads=2,
                 global_transformer_layers=2,
                 global_transformer_heads=4):
        super().__init__()
        # Step 1: Fourier encoder
        self.fourier_enc = FourierSpatioTemporalEncoder(
            d_space=64, d_time=32, d_out=64, r_s=r_s, r_t=r_t
        )
        # Step 2: gating
        self.gating = DynamicSegmentationGating(d_in=64, threshold=gating_threshold)
        # Step 3: segment mini-transformer
        self.seg_trans = SegmentMiniTransformer(
            d_model=64,
            nhead=seg_transformer_heads,
            num_layers=seg_transformer_layers,
            dim_feedforward=128,
            dropout=0.1
        )
        # Step 4: up-project seg => global dim if needed
        self.up_proj = nn.Linear(64, 128)
        # Step 5: global aggregator
        self.global_agg = GlobalAggregator(
            d_model=128,
            nhead=global_transformer_heads,
            num_layers=global_transformer_layers,
            dim_feedforward=256,
            dropout=0.1
        )
        # optional final projection
        self.final_proj = nn.Linear(128, d_out)

    def forward(self, pts):
        """
        pts: (B,L,3) => each row is (x,y,t).
        returns (B, d_out), plus optional segment details.
        """
        B, L, _ = pts.size()
        # 1) Fourier enc
        fourier_out = self.fourier_enc(pts) # (B,L,64)

        # 2) gating => dynamic segments
        all_embeddings = []
        for b in range(B):
            z_b = fourier_out[b,:,:] # (L,64)
            segments_b = self.gating(z_b.unsqueeze(0)) # list of lists => shape: [ [ (seg1) , (seg2), ... ] ]
            # segments_b[0] => list of segments for batch b
            sub_trajs = segments_b[0] # list of (seg_len,64)
            seg_embs=[]
            for seg_data in sub_trajs:
                seg_emb = self.seg_trans(seg_data) # (64) => local seg embedding
                seg_emb = self.up_proj(seg_emb)    # (128)
                seg_embs.append(seg_emb)
            # aggregator
            if len(seg_embs)==1:
                # trivial
                global_emb = seg_embs[0]
            else:
                global_emb = self.global_agg(seg_embs) # (128)
            final = self.final_proj(global_emb)         # (d_out)
            all_embeddings.append(final)
        emb_batch = torch.stack(all_embeddings, dim=0) # (B, d_out)
        return emb_batch
