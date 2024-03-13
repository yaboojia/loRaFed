import torch.nn as nn


class LoraConv(nn.Module):
    def __init__(self, in_channels, out_channels, rank):
        super().__init__()
        self.up_conv = nn.Conv2d(in_channels, rank, kernel_size=3, stride=1, padding=1)
        self.down_conv = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1, padding=1)

    def forward(self, x):
        return self.down_conv(self.up_conv(x))
    

# 递归替换conv2d层为loraconv2d层
def replace_model(module, rate):
    if rate == 1:
        return
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            # 创建一个新的loraconv2d层，使用与原conv2d层相同的参数
            cout, cin, m, n = child.weight.size()
            rank = int((cin*cout*m*n*rate) / (cout + cin*m*n))
            # print(f"{name} RANK is: {rank}")
            loraconv = LoraConv(cin, cout, rank)
            # 将新的loraconv2d层替换原conv2d层
            setattr(module, name, loraconv)
        else:
            # 递归替换子模块的conv2d层
            replace_model(child, rate)


def init_param(m):
    if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m


# 缩小或者扩大input中tensor的规模
class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, input):
        output = input / self.rate if self.training else input
        return output
