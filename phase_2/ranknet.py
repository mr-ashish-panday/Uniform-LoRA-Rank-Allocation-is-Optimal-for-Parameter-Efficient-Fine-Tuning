# phase2_ranknet/ranknet.py
import torch
import torch.nn as nn

class RankNet(nn.Module):
    """
    Maps per-layer features + normalized memory budget -> scalar rank proposals.

    Output is positive and smoothly parameterized using Softplus and then clamped
    to [0, max_rank]. Use rounding/clip when applying to actual LoRA injection.
    """
    def __init__(self, feature_dim: int, hidden_size: int = 128, max_rank: int = 64):
        super().__init__()
        self.max_rank = float(max_rank)
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus()   # smooth positive outputs
        )

    def forward(self, features: torch.Tensor, budget_norm: torch.Tensor):
        """
        features: (B, feature_dim)
        budget_norm: scalar tensor or (B,1) tensor (normalized to the same scale as features)
        returns: (B,) predicted continuous ranks in (0, +inf) but clamped to max_rank
        """
        # ensure shapes: budget_norm -> (B,1)
        if budget_norm.dim() == 0:
            b = budget_norm.unsqueeze(0).expand(features.size(0), 1)
        elif budget_norm.dim() == 1 and budget_norm.numel() == features.size(0):
            b = budget_norm.unsqueeze(1)
        elif budget_norm.dim() == 2 and budget_norm.size(0) == features.size(0):
            b = budget_norm
        else:
            # broadcast fallback
            b = budget_norm.unsqueeze(0).expand(features.size(0), 1)
        inp = torch.cat([features, b.to(features.dtype)], dim=1)
        out = self.net(inp).squeeze(1)
        out = torch.clamp(out, min=0.0, max=self.max_rank)
        return out
