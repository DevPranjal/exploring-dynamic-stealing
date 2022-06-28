# This part is borrowed from https://github.com/huawei-noah/Data-Efficient-Model-Compression

from numpy import average
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 1

class GateLayer(nn.Module):
    def __init__(self, num_features, k):
        super(GateLayer, self).__init__()
        self.k = k
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fully_connected = nn.Linear(num_features, 1)
        
    def forward(self, x):
        pooled_features = torch.squeeze(torch.squeeze(self.global_avg_pool(x), -1), -1)
        g = self.fully_connected(pooled_features)
        
        zero = torch.tensor(0.)
        one = torch.tensor(1.)
        condition = torch.max(zero, torch.min(self.k * g + 0.5, one))        
        
        self.conditions = torch.squeeze(condition, -1) > 0.5
        output = x.clone()
        
        for i, sample in enumerate(x):
            if not self.conditions[i]:
                output[i] = torch.zeros_like(sample)
                
        return output


class BasicBlockWithGate(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlockWithGate, self).__init__()
        self.gate = GateLayer(in_planes, K)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut.add_module('conv_shortcut', nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                ))

    def forward(self, x):
        out = self.gate(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DynamicResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, normalize_coefs=None, normalize=False):
        super(DynamicResNet, self).__init__()

        if normalize_coefs is not None:
            self.mean, self.std = normalize_coefs

        self.normalize = normalize

        self.in_planes = 64
 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x, out_feature=False):

        if self.normalize:
            # Normalize according to the training data normalization statistics
            x -= self.mean
            x /= self.std

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        if out_feature == False:
            return out
        else:
            return out,feature


def DynamicResNet18_8x(num_classes=10):
    return DynamicResNet(BasicBlockWithGate, [2,2,2,2], num_classes)
