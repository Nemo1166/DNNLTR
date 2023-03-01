import torch
import torch.nn as nn
import torch.nn.functional as F

class DNNLTR(nn.Module):
    def __init__(self, feats_size: int, label_classes: int):
        super().__init__()
        conv_y = feats_size
        conv_x = min(16, feats_size)
        self.conv_local = nn.Conv1d(in_channels=1, out_channels=conv_y, kernel_size=conv_x, stride=1)
        self.conv_global = nn.Conv1d(in_channels=1, out_channels=conv_y, kernel_size=conv_x, stride=1)
        # self.pool = nn.MaxPool1d()

    def forward(self, feats):
        # Global View
        view_global = F.max_pool1d(self.conv_global(feats),self.conv_global(feats).size(dim=2))
        # Local View
        view_local = self.conv_local(feats)
        # Fusion
        dot_product = torch.sum(view_global * view_local, dim=2)
        return torch.sum(dot_product, dim=1)