"""
Numerically stable AdaIN implementation for ExecuTorch
Replaces InstanceNorm1d with manual mean/std computation with proper epsilon handling
"""

import torch
import torch.nn as nn


class StableAdaIN1d(nn.Module):
    """
    Numerically stable Adaptive Instance Normalization.
    Manually computes mean/std to avoid ExecuTorch InstanceNorm1d issues.
    """
    def __init__(self, style_dim, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.fc = nn.Linear(style_dim, num_features * 2)

        # Learnable parameters for normalization (affine=True equivalent)
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x, s):
        # Manual instance normalization with stable epsilon
        # x shape: (B, C, T)
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, unbiased=False, keepdim=True)

        # Clamp variance to prevent division by zero
        var = torch.clamp(var, min=self.eps)
        std = torch.sqrt(var)

        # Normalize
        x_norm = (x - mean) / std

        # Apply affine transform
        x_norm = x_norm * self.weight + self.bias

        # Apply style modulation
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)

        return (1 + gamma) * x_norm + beta
