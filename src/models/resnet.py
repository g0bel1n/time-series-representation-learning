import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial



class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, stride=1, padding='same', is_final:bool = False):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=8,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=5, stride=stride, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False
        )
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.BatchNorm1d(out_channels) if is_final else nn.Sequential(nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=padding,
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
        in_channels,
        task: str,
         pred_len : int = 96,
        head_dropout: float = 0.2
       
    ):
        out_channels = in_channels
        self.n_channels = in_channels
        super(ResNet, self).__init__()

        self.block1 = ResNetBlock(1, 64)
        self.block2 = ResNetBlock(64, 128)
        self.block3 = ResNetBlock(128, 128, is_final=True)

        self.gap = partial(torch.mean, axis=-1)
        self.dropout = nn.Dropout(head_dropout)

        self.pred_len = pred_len

        if task == "classification":
            self.fc = nn.Linear(128, pred_len)
        elif task == "regression":
            self.fc = nn.Sequential(nn.Linear(128, pred_len), nn.Dropout(head_dropout), nn.Linear(pred_len, pred_len))
        else:
            raise ValueError("Task should be either classification or regression")

    def freeze(self):
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


    def _one_forward(self, x):
        chan_cat_out = torch.zeros(x.shape[0], x.shape[1], self.pred_len, device='cuda:0')
        for chan in range(self.n_channels):
            out = self.block1(x[:, chan, :].unsqueeze(1))
            out = self.block2(out)
            out = self.block3(out)
            out = self.gap(out)
            out = self.dropout(out)
            out = self.fc(out)
            chan_cat_out[:, chan, :] = out.reshape(-1, 1)
        return chan_cat_out

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self._one_forward(x)
        out = out.permute(0, 2, 1)
        if self.pred_len == 1:
            out = out.squeeze(-1)

        return out
        
