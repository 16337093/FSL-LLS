import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


ceil = True
inp =True

class ResNetBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes, eps=2e-5)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-5)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-5)

        self.convr = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, padding=1)
        self.bnr = nn.BatchNorm2d(planes, eps=2e-5)

        self.relu = nn.ReLU(inplace=inp)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil)

    def forward(self, x):
        identity = self.convr(x)
        identity = self.bnr(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out).contiguous()
        out = self.maxpool(out)
        return out


class Resnet12(nn.Module):
    def __init__(self, num_class=64):
        super(Resnet12, self).__init__()

        self.inplanes = 3
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1, inplace=inp)
        self.layer1 = self._make_layer(ResNetBlock, 64)
        self.layer2 = self._make_layer(ResNetBlock, 128)
        self.layer3 = self._make_layer(ResNetBlock, 256)
        self.layer4 = self._make_layer(ResNetBlock, 512)

        # Lateral layers
        self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        # global weight
        self.weight = nn.Conv2d(512, num_class, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=inp)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='conv2d')
                if m.bias is None:
                    continue
                else:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    def _make_layer(self, block, planes):
        layers = []
        layers.append(block(self.inplanes, planes))
        self.inplanes = planes

        return nn.Sequential(*layers)
    
    def transfer(self, x, beta=0.8):
        x1 = self.layer1(x)

        x2 = self.layer2(x1)

        x3 = self.layer3(x2)

        x4 = self.layer4(x3)
        x3 = self._upsample_add(self.toplayer(x4), x3)

        pred = self.weight(x4)

        pred = F.softmax(pred, 1)
        proto = (self.weight.weight*pred.unsqueeze(2)).sum(1)

        b, c, w, h = x4.size()
        x4 = F.normalize(x4.reshape(b, -1), dim=1).reshape(b, c, w, h)
        proto = F.normalize(proto.reshape(b, -1), dim=1).reshape(b, c, w, h)
        x4 = beta*x4+(1-beta)*proto
        return [x4, x3]


    def forward(self, x, labels=None):
        # Bottom-up
        x1 = self.layer1(x)
        x1 = self.dropout(x1)

        x2 = self.layer2(x1)
        x2 = self.dropout(x2)

        x3 = self.layer3(x2)
        x3 = self.dropout(x3)

        x4 = self.layer4(x3)
        x4 = self.dropout(x4)

        x3 = self._upsample_add(self.toplayer(x4), x3)

        return [x4, x3]

