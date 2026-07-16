
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """
    AttentionBlock
    """
    def __init__(self, in_channels, g_channels):
        super().__init__()
        self.W_g = nn.Conv2d(g_channels, in_channels, kernel_size=1)
        self.W_x = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.W_h = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.psi = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.relu(g1 + x1)
        psi = self.W_h(psi)
        psi = torch.sigmoid(self.psi(psi))
        return x * psi
