import torch
from torchvision import models
from lightning import LightningModule
from torch import nn


class EfficientNetWrapper(LightningModule):
    def __init__(self):
        super(EfficientNetWrapper, self).__init__()
        self.efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.base_layers = nn.Sequential(
            self.efficientnet.features[0],
            self.efficientnet.features[1],
            self.efficientnet.features[2],
        )
        self.layer1 = self.efficientnet.features[3]
        self.layer2 = nn.Sequential(
            self.efficientnet.features[4], self.efficientnet.features[5]
        )
        self.layer3 = self.efficientnet.features[6]
        self.layer4 = self.efficientnet.features[7]
        self.layer5 = self.efficientnet.features[8]

    def forward(self, x):
        x = self.base_layers(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        return x1, x2, x3, x4, x5


if __name__ == '__name__':
    wrapper = EfficientNetWrapper()
    fake_data = torch.rand(1, 3, 224, 224)
    features = wrapper(fake_data)
    assert features is not None
