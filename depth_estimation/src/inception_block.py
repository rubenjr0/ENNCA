import torch
from torch import nn


class InceptionBlock(nn.Module):
    def __init__(self, in_dims, base=16):
        super().__init__()
        self.branch_1_1 = nn.Conv2d(in_dims, base, 1)
        self.branch_1_3 = nn.Conv2d(base, base*2, 3, padding=1)
        self.branch_1_13 = nn.Conv2d(base*2, base*3, (1, 3), padding=(0, 1))
        self.branch_1_31 = nn.Conv2d(base*2, base*3, (3, 1), padding=(1, 0))

        self.branch_2_1 = nn.Conv2d(in_dims, base, 1)
        self.branch_2_13 = nn.Conv2d(base, base*2, (1, 3), padding=(0, 1))
        self.branch_2_31 = nn.Conv2d(base, base*2, (3, 1), padding=(1, 0))

        self.branch_3_1 = nn.Conv2d(in_dims, base, 1)

        self.downsample = nn.Conv2d(in_dims, base*11, 1, bias=False)

    def forward(self, x):
        branch_1 = self.branch_1_1(x)
        branch_1 = self.branch_1_3(branch_1)
        branch_1_1 = self.branch_1_13(branch_1)
        branch_1_2 = self.branch_1_31(branch_1)

        branch_2 = self.branch_2_1(x)
        branch_2_1 = self.branch_2_13(branch_2)
        branch_2_2 = self.branch_2_31(branch_2)

        branch_3 = self.branch_3_1(x)

        res = self.downsample(x)

        x = torch.cat([branch_1_1, branch_1_2, branch_2_1,
                      branch_2_2, branch_3], dim=1)
        x += res

        return x
