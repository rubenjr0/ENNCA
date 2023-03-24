import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channel, out_channel,
                                kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv_2 = nn.Conv2d(out_channel, out_channel,
                                kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(self.bn1(x))
        x = self.conv_2(x)
        x = F.relu(self.bn2(x))
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        out_channel = in_channel//2
        self.ups = nn.ConvTranspose2d(
            in_channel, out_channel, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(
            out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.ups(x)
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(6)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv_block_1 = ConvBlock(16, 16)
        self.conv_block_2 = ConvBlock(16, 16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(self.pool1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

        x = self.conv_block_1(x)  # Applies 2 convolutions
        x = self.pool3(x)
        x = self.conv_block_2(x)  # Applies 2 convolutions
        x = self.pool4(x)
        return x


class Knowledge(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = ConvBlock(16, 48)
        self.contract = ConvBlock(48, 16)

    def forward(self, x):
        x = self.expand(x)
        x = self.contract(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ups1 = UpsampleBlock(16)
        self.ups2 = UpsampleBlock(8)
        self.ups3 = UpsampleBlock(4)
        self.ups4 = UpsampleBlock(2)

    def forward(self, x):
        x = self.ups1(x)
        x = self.ups2(x)
        x = self.ups3(x)
        x = self.ups4(x)
        return x


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(1, 1)
        self.conv2 = ConvBlock(1, 1)
        self.conv3 = ConvBlock(1, 1)
        self.conv4 = ConvBlock(1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.knowledge = Knowledge()
        self.decoder = Decoder()
        self.head = Head()

    def forward(self, x):
        x = self.encoder(x)
        x = self.knowledge(x)
        x = self.decoder(x)
        return self.head(x)


net = Net()


def main():
    input = Image.open(
        "data/nyu2_train/basement_0001a_out/1.jpg").convert('RGB')
    input = np.array(input).transpose((2, 0, 1)).astype(np.float32) / 255
    input = torch.tensor(input).unsqueeze(0)
    print(input)
    output = net(input)
    print(f'Output shape: {output.shape}')
    print(f'Output: {output}')

    img_arr = output.detach().numpy()[0, 0, :, :]
    img = Image.fromarray((img_arr * 255).astype(np.uint8))
    img.save("output.png")
