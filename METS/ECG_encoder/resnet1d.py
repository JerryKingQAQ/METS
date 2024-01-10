# -*- coding = utf-8 -*-
# @File : resnet1d.py
# @Software : PyCharm
import torch
from torch import nn
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    def __init__(self, in_channels, block, layers, projection_size):
        super(ResNet1D, self).__init__()
        self.out_channels = 64
        self.in_channels = in_channels
        self.conv1 = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, projection_size)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.out_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.out_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.out_channels, out_channels, stride, downsample))
        self.out_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = x.transpose(1, 2)  # 转换维度
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18_1d(in_channels, block=BasicBlock1D, layers=[2, 2, 2, 2], projection_size=768):
    """Constructs a ResNet-18-1D model."""
    # model = ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_features, projection_size)
    model = ResNet1D(in_channels, block=block, layers=layers, projection_size=projection_size)
    return model