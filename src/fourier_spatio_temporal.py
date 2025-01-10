import torch
import torch.nn as nn

class FourierSpatioTemporalEncoder(nn.Module):
    def __init__(self, 
                 d_space=64, 
                 d_time=32, 
                 d_out=128, 
                 r_s=16,
                 r_t=8):
        """
        d_space + d_time -> final d_out
        r_s, r_t: number of random Fourier frequencies for space/time.
        """
        super().__init__()
        self.r_s = r_s
        self.r_t = r_t
        self.d_space = d_space
        self.d_time = d_time
        self.d_out = d_out

        # random freq for space
        self.freq_space = nn.Parameter(torch.randn(r_s, 2), requires_grad=False)
        # random freq for time
        self.freq_time = nn.Parameter(torch.randn(r_t, 1), requires_grad=False)

        # MLPs
        self.mlp_space = nn.Sequential(
            nn.Linear(2*r_s*1, d_space//2),
            nn.ReLU(),
            nn.Linear(d_space//2, d_space)
        )
        self.mlp_time = nn.Sequential(
            nn.Linear(2*r_t*1, d_time//2),
            nn.ReLU(),
            nn.Linear(d_time//2, d_time)
        )

        self.proj = nn.Linear(d_space + d_time, d_out)

    def forward(self, x):
        """
        x: (B, L, 3) => each row is (x,y,t).
        Returns (B, L, d_out)
        """
        B, L, _ = x.size()
        xy = x[:,:,:2]  # (B,L,2)
        tt = x[:,:,2:3] # (B,L,1)

        # random fourier for space
        # shape => (B,L,2) -> expand to (B,L,1,2) -> multiply with (r_s,2)? We'll do a broadcast approach
        # simpler approach: flatten batch
        xy_flat = xy.view(-1,2) # (B*L, 2)
        freq_s = self.freq_space  # (r_s,2)
        # compute angles => (B*L, r_s)
        angles_s = xy_flat @ freq_s.T # => (B*L, r_s)
        sin_s = torch.sin(angles_s)
        cos_s = torch.cos(angles_s)
        # => (B*L, 2*r_s)
        space_feats = torch.cat([sin_s, cos_s], dim=1)

        # random fourier for time
        tt_flat = tt.view(-1,1) # (B*L,1)
        freq_t = self.freq_time # (r_t,1)
        angles_t = tt_flat @ freq_t.T # (B*L, r_t)
        sin_t = torch.sin(angles_t)
        cos_t = torch.cos(angles_t)
        time_feats = torch.cat([sin_t, cos_t], dim=1) # (B*L, 2*r_t)

        # MLP
        space_out = self.mlp_space(space_feats) # (B*L, d_space)
        time_out = self.mlp_time(time_feats)    # (B*L, d_time)

        comb = torch.cat([space_out, time_out], dim=1) # (B*L, d_space + d_time)
        final = self.proj(comb) # (B*L, d_out)
        final = final.view(B, L, self.d_out)
        return final
