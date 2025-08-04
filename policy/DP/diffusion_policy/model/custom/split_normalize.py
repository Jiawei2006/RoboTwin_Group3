import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class SplitNormalize(nn.Module):
    def __init__(self, mean1, std1, mean2=None, std2=None, split_index=3):
        super().__init__()
        self.mean1 = mean1
        self.std1 = std1
        self.mean2 = mean2 or mean1
        self.std2 = std2 or std1
        self.split_index = split_index

    def forward(self, x):
        # x shape: [B, C, H, W]
        x1 = TF.normalize(x[:, :self.split_index], mean=self.mean1, std=self.std1)
        x2 = TF.normalize(x[:, self.split_index:], mean=self.mean2, std=self.std2)
        return torch.cat([x1, x2], dim=1)
