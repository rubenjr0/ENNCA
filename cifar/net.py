import torch
from torch import nn
import torch.nn.functional as F
from cifar_dataset import train_dataset_instance


class ConvBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        return self.pool(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.conv_block_1 = ConvBlock()
        self.conv_block_2 = ConvBlock()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16, 768)
        self.fc2 = nn.Linear(768, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = self.conv_block_1(x)  # Applies 2 convolutions and 1 MaxPool
        x = self.conv_block_2(x)  # Applies 2 convolutions and 1 MaxPool

        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # Softmax so the sum of probabilities is 1
        return F.softmax(self.fc4(x), 1)


net = Net()

if __name__ == '__main__':
    image, label = train_dataset_instance[0]
    print(f'Image shape: {image.shape}')
    x = torch.unsqueeze(image, 0)
    print(f'Input shape: {x.shape} (unsqueezed)')

    output = net(x)
    print(f'Output shape: {output.shape}')
    print(f'Output: {output}')
