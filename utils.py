from collections import OrderedDict
import torch


def lora2full(params: OrderedDict) -> OrderedDict:
    new_params = OrderedDict()
    for k, v in params.items():
        if 'conv' in k:
            if 'conv.up_conv.weight' in k:
                # print(k.split('.')[0])
                x = params[k.split('.')[0] + '.conv.up_conv.weight']
                y = params[k.split('.')[0] + '.conv.down_conv.weight']
                x_s, y_s = x.size(), y.size()
                z = y.view(y_s[0], -1) @ x.view(x_s[0], -1)
                z = z.view(y_s[0], x_s[1], x_s[2], x_s[3])
                z_bias = torch.zeros((y_s[0]))

                new_params[k.split('.')[0] + '.conv.weight'] = z
                new_params[k.split('.')[0] + '.conv.bias'] = z_bias
            else:
                continue
        else:
            new_params[k] = v
    return new_params



# SVD
def full2lora(params: OrderedDict, rate: float) -> OrderedDict:
    new_params = OrderedDict()
    for key, value in params.items():
        if 'conv' in key:
            print(key)
            if 'weight' in key:
                cout, cin, m, n = value.size()
                w_2dim = value.view(cout, -1)
                rank = int((cin*cout*m*n*rate) / (cout + cin*m*n))
                U, S, V = torch.svd(w_2dim)

                B = U[:, :rank] * torch.sqrt(S[:rank])
                A = torch.sqrt(S[:rank]).unsqueeze(1) * V[:, :rank].t()
                print(f"{rank}   {B.size()}     {A.size()}")

                up_weight = A.view(A.size(0), -1, m, n)
                up_bias = torch.zeros((up_weight.size(0)))

                down_weight = B.view(B.size(0), B.size(1), 1, 1)
                down_bias = torch.zeros((down_weight.size(0)))


                new_params[key.replace('weight', 'up_conv.weight')] = up_weight
                new_params[key.replace('weight', 'up_conv.bias')] = up_bias

                new_params[key.replace('weight', 'down_conv.weight')] = down_weight
                new_params[key.replace('weight', 'down_conv.bias')] = down_bias            
            else:
                continue
    else:
        new_params[key] = value
    return new_params

