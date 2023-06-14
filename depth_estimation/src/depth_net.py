import torch
from torch import nn
from efficientnet_wrapper import EfficientNetWrapper
from decoder_block import DecoderBlock
from inception_block import InceptionBlock
from sep_conv_block import SepConvBlock


class DepthNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EfficientNetWrapper()
        self.d1 = DecoderBlock()
        self.d2 = DecoderBlock()
        self.d3 = DecoderBlock()
        self.downsample = nn.Sequential(
            nn.Conv2d(3, 60, 1, bias=False),
            nn.MaxPool2d(4)
        )
        self.head = nn.Sequential(
            InceptionBlock(60),
            SepConvBlock(176, 40),
            SepConvBlock(40, 1, extras=False),
        )
        self.act = nn.ReLU6()

    def forward(self, x):
        res1 = self.downsample(x)
        x = self.encoder(x)
        x1 = self.d1(x)
        x2 = self.d2(x)
        x3 = self.d3(x)
        x = torch.cat([x1, x2, x3], 1)
        x += res1
        x = self.head(x)
        x = self.act(x)
        return x
