import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, stride=1):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=8,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=5, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(
            
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task: str,
        num_classes: int,
        head_dropout: float = 0.2,
    ):
        super(ResNet, self).__init__()

        self.block1 = ResNetBlock(in_channels, 64, stride=1)
        self.block2 = ResNetBlock(64, 128, stride=1)
        self.block3 = ResNetBlock(128, 128, stride=1)

        self.gap = nn.GlobalAvgPool1d()
        self.dropout = nn.Dropout(head_dropout)

        if task == "classification":
            self.fc = nn.Linear(128, num_classes*out_channels)
        elif task == "regression":
            self.fc = nn.Linear(128, 1)
        else:
            raise ValueError("Task should be either classification or regression")

    def freeze_but_last(self):
        self._freeznt(False)

    def unfreeze(self):
        self._freeznt(True)

    def _freeznt(self, arg0):
        for param in self.block1.parameters():
            param.requires_grad = arg0
        for param in self.block2.parameters():
            param.requires_grad = arg0
        for param in self.block3.parameters():
            param.requires_grad = arg0

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.gap(out)
        out = self.dropout(out)
        out = self.fc(out)

        return out
