import torch
import torch.nn as nn

class CNN_6_to_3(nn.Module):
    def __init__(self, hidden_channels: int = 16):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x.shape = [B, 6, H, W]
        输出: out.shape = [B, 3, H, W]
        """
        print(f"[CNN_6_to_3] Input shape: {x.shape}")
        return self.model(x)
