from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from src.data import train_dataset_instance


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
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
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
    x, _ = train_dataset_instance[0]
    print(f'Input shape: {x.shape}')
    x = x.unsqueeze(0)
    print(f'Input shape unsqueezed: {x.shape}')

    y = net(x)
    print(f'Output shape: {y.shape}')
    y = y.squeeze(0)
    print(f'Output shape squeezed: {y.shape}')

    img = to_pil_image(y, mode='L')
    img.save("demos/net_output.png")


if __name__ == '__main__':
    main()
