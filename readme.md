## This is a respository for addressing heterorious client federate learning using LoRa.
## Capacity distribution

| capacity |  |  |  |  |  | 
| :----- | :------: | -----: | -----: | -----: | -----: |
| rate |  1 | 1/2 | 1/4 | 1/8 | 1/16 | 
|distribution| 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 

## conv2d - LoRa
### before Conv2d
```python
torch.nn.Module.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

```
### after

```python
class LoraConv(nn.Module):
    def __init__(self, in_channels, out_channels, rank):
        super().__init__()
        self.up_conv = nn.Conv2d(in_channels, rank, kernel_size=3, stride=1, padding=1)
        self.down_conv = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1, padding=1)

    def forward(self, x):
        return self.down_conv(self.up_conv(x))
```
### parameters and the calculation of rank
$
(inchannels + outchannels) * (kernel_m * kernel_n) * rate = (inchannels+rank) * (kernel_m * kernel_n) + (rank + outchannels) * (1*1)
$

## conv2dLoRa -> conv2d
up_conv.weight
> rank, in_channels, kernel_m, kernel_n

down_conv.weight
> out_channels, rank, 1, 1

conv2d.weight
> out_channels, in_channels, kernel_m, kernel_n

lora -> conv2d
> (out_channels, rank\*1\*1) @ (rank, in_channels\*kernel_m\*kernel_n)

## conv2d -> conv2dLoRa
```python
U, S, V = torch.svd(conv2d.weight)

B = U[:, :rank] * torch.sqrt(S[:rank])
A = torch.sqrt(S[:rank]).unsqueeze(1) * V[:, :rank].t()
# print(f"{rank}   {B.size()}     {A.size()}")

up_weight = A.view(A.size(0), -1, m, n)
up_bias = torch.zeros((up_weight.size(0)))

down_weight = B.view(B.size(0), B.size(1), 1, 1)
down_bias = torch.zeros((down_weight.size(0)))
```

## experimnet
| method | conv + cifar10 |  
| :----- | :------: | 
| loRaFL | 0.52(Non-convergence)  |  
| roll | 0.92  | 




