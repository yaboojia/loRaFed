import torch
import torch.nn as nn
from collections import OrderedDict
import math
if __name__ == '__main':
    from utils import init_param, Scaler, replace_model
else:
    from .utils import init_param, Scaler, replace_model

class Conv(nn.Module):
    def __init__(self, data_shape, hidden_size, classes_size, rate=1, track=True):
        super().__init__()
        self.rate = rate
        self.scaler = Scaler(rate)
        self.data_shape = data_shape
        self.hidden_size = hidden_size
        self.classes_size = classes_size
        self.track = track

        self.layer0 = self._make_layer(data_shape[0], hidden_size[0])
        self.layer1 = self._make_layer(hidden_size[0], hidden_size[1])
        self.layer2 = self._make_layer(hidden_size[1], hidden_size[2])
        self.layer3 = self._make_layer(hidden_size[2], hidden_size[3])

        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(hidden_size[-1], classes_size)


    def _make_layer(self, in_channels, out_channels, track=True):
        layer = nn.Sequential(
            OrderedDict(
                [
                    ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
                    ('scale', Scaler(self.rate)),
                    ('norm', nn.BatchNorm2d(out_channels, momentum=None, track_running_stats=track)),
                    ('relu', nn.ReLU(inplace=True)),
                    ('maxPool', nn.MaxPool2d(2)),
                ]
            )
        )
        return layer
    
    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(self.layer0(x))))
        return self.linear(self.flatten(self.avgPool(x)))
    

    

def conv(data_shape, hidden_size, classes_size, model_rate=1, track=True):
    # hidden_size = [int(model_rate * x) for x in hidden_size]
    model = Conv(data_shape, hidden_size, classes_size, model_rate, track)
    model.apply(init_param)
    return model

if __name__ == '__main__':
    x = torch.rand(2, 3, 32, 32)
    model = conv([3],  [64, 128, 256, 512], 10)
    print(model(x).size())
    # print(model)
    replace_model(model, 0.125)
    print(model(x).size())
    print(model.layer0.conv.up_conv)
    

    







