import torch
import torch.nn as nn
from collections import OrderedDict
import math
if __name__ == '__main__':
    from utils import init_param, Scaler, replace_model
else:
    from .utils import init_param, Scaler, replace_model

class layer(nn.Module):
    def __init__(self, in_channels, out_channels, rank=-1, rate=1, track=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.up_conv =  nn.Conv2d(in_channels, rank, kernel_size=3, stride=1, padding=1)
        self.down_conv =  nn.Conv2d(rank, out_channels, kernel_size=1, stride=1, padding=0)
        self.scale = Scaler(rate)
        self.norm = nn.BatchNorm2d(out_channels, momentum=None, track_running_stats=track)
        self.relu = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(2)

    def forward(self, x, using_adapter=False):
        if using_adapter:
            y = self.conv(x)
            x = y + self.down_conv(self.up_conv(x))
            return  self.maxPool(self.relu(self.norm(self.scale(x))))
        else:
            return  self.maxPool(self.relu(self.norm(self.scale(self.conv(x)))))
        


class Conv(nn.Module):
    def __init__(self, data_shape, hidden_size, classes_size, rank =4, rate=1, track=True):
        super().__init__()
        self.rate = rate
        self.scaler = Scaler(rate)
        self.data_shape = data_shape
        self.hidden_size = hidden_size
        self.classes_size = classes_size
        self.track = track
        self.rank = rank

        # self.layer0 = self._make_layer(data_shape[0], hidden_size[0])
        # self.layer1 = self._make_layer(hidden_size[0], hidden_size[1])
        # self.layer2 = self._make_layer(hidden_size[1], hidden_size[2])
        # self.layer3 = self._make_layer(hidden_size[2], hidden_size[3])
        
        self.layer0 = layer(data_shape[0], hidden_size[0], rank)
        self.layer1 = layer(hidden_size[0], hidden_size[1], rank)
        self.layer2 = layer(hidden_size[1], hidden_size[2], rank)
        self.layer3 = layer(hidden_size[2], hidden_size[3], rank)

        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(hidden_size[-1], classes_size)

    
    def forward(self, x, using_adapter=False):
        x = self.layer3(self.layer2(self.layer1(self.layer0(x, using_adapter), using_adapter), using_adapter), using_adapter)
        return self.linear(self.flatten(self.avgPool(x)))


if __name__ == '__main__':
    x = torch.rand((2, 3, 32, 32))
    # l = layer(3, 64, 4)
    # y = l(x, True)
    model = Conv([3], [64, 128, 256, 512], 10, )
    y = model(x)
    print(y.size())