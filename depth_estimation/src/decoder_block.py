import torch
from torch import nn
from bifpn import BiFPN


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.bifpn = BiFPN([40, 112, 192, 320, 1280])
        self.upsample1 = nn.PixelShuffle(2)
        self.upsample2 = nn.PixelShuffle(4)
        self.bn = nn.BatchNorm2d(20)
        self.act = nn.GELU()

    def forward(self, x):
        x1, x2, _, _, _ = self.bifpn(x)
        x1 = self.upsample1(x1)
        x2 = self.upsample2(x2)
        x = torch.cat([x1, x2], 1)
        x = self.bn(x)
        x = self.act(x)
        return x
