from torch import nn


class SepConvBlock(nn.Module):
    def __init__(self, in_dims, out_dims, kernel_size=3, extras=True):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_dims, in_dims, kernel_size=kernel_size,
            padding=kernel_size//2, groups=in_dims)
        self.point_conv = nn.Conv2d(in_dims, out_dims, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_dims) if extras else None
        self.act = nn.GELU() if extras else None

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        if self.bn:
            x = self.bn(x)
            x = self.act(x)
        return x
