import torch
import torch.nn.functional as F
import torch.optim as optim

import yaml
import random
from collections import OrderedDict

from expdataset.dataset import load_datasets
from utils import get_model, lora2full, full2lora, merge_list_params

def main():
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    print(cfg)
    trainloaders, _, testloads = load_datasets(cfg, cfg['num_clients'], batch_size=cfg['batch_size'], dataset=cfg['dataset'])
    global_model = get_model(cfg['model_name'], cfg['classes'], 1)

    model_list = {}
    for rate in cfg['submodel_rate']:
        # model_list[str(rate)] = get_model(cfg['model_name'], cfg['classes'], rate).state_dict()
        model_list[str(rate)] = OrderedDict()
    
    for i in range(cfg['rounds']):
        print(f"The {i+1} rounds:")
        num_activate_users = int(cfg['num_clients'] * cfg['frc'])
        user_idxs = torch.randperm(cfg['num_clients'])[:num_activate_users]
        dic_local_params = {'1':[], '0.5':[], '0.25':[], '0.125':[], '0.0625':[]}
        for user in user_idxs:
            user = int(user)
            local_trainloaders = trainloaders[user] 

            dis = torch.distributions.Categorical(probs=torch.tensor(cfg['probs']))
            rate = cfg['submodel_rate'][dis.sample()]

            client_model = get_model(cfg['model_name'], cfg['classes'], rate)
            load_globalAlist2client_model(global_model, model_list, client_model, rate)
            train(client_model, local_trainloaders, dic_local_params, cfg, rate)
        merge_lora2list_model(dic_local_params, model_list)
        merge_list2global_model(model_list, global_model)
        test(global_model, testloads, cfg)


def load_globalAlist2client_model(global_model, model_list, client_model, rate):
    if rate == 1:
        client_model.load_state_dict(global_model.state_dict())
        return
    global_params = global_model.state_dict()
    lora_params = model_list[str(rate)]
    # for k, v in lora_params.items():
    #     print(k)
    loraparams_from_global = full2lora(global_params, lora_params, rate)
    model_params = merge_list_params([lora_params, loraparams_from_global])
    client_model.load_state_dict(model_params)
    

def merge_lora2list_model(dic_local_params, model_list):
    for k, v in dic_local_params.items():
        if len(v) == 0:
            continue
        temp_params = merge_list_params(v)
        model_list[str(k)] = temp_params

def merge_list2global_model(model_list, global_model):
    global_params = global_model.state_dict()
    merge_full_params = [lora2full(v, global_params) for k, v in model_list.items() if k != '1']
    new_global_params = merge_list_params(merge_full_params + [model_list['1']])
    global_model.load_state_dict(new_global_params)


def train(client_model, loacl_trainloaders, dic_local_params, cfg, rate):
    device, lr = cfg['device'], cfg['lr']
    client_model.to(device)
    optimizer = optim.SGD(client_model.parameters(), lr=lr, momentum=cfg['momentum'],
                              weight_decay=cfg['weight_decay'])
    client_model.train()
    local_loss = 0
    for local_epoch in range(cfg['epoch']):
        for batch in loacl_trainloaders:
            img, target = batch[0], batch[1]
            img, target = img.to(device), target.to(device)
            client_model.zero_grad()
            output = client_model(img)
            loss = F.cross_entropy(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), 1)
            optimizer.step()
            # 将loss相加
            local_loss += loss
    # print('     user_idx {}: Local Loss {}'.format(user_idx, local_loss))
    client_model.cpu()
    dic_local_params[str(rate)].append(client_model.state_dict())
    

def test(global_model, testset, cfg):
    # 测试global model
    device, lr = cfg['device'], cfg['lr']
    optimizer = optim.SGD(global_model.parameters(), lr=lr, momentum=cfg['momentum'],
                              weight_decay=cfg['weight_decay'])
    correct, total = 0, 0
    global_model.to(device)
    global_model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in testset:
            img, target = batch[0], batch[1]
            img, target = img.to(device), target.to(device)
            output = global_model(img)
            loss = F.cross_entropy(output, target)
            _, predicted = torch.max(output, dim=1)
            test_loss += loss.data
            total += img.shape[0]
            correct += (predicted == target).sum().item()
    print(' Global Model Acc: {}, Test Loss: {}'.format(correct / total, test_loss))
    # if cfg['use_wandb']:
        # wandb.log({'acc': correct / total, 'loss': test_loss})
    global_model.cpu()

    
if __name__ == '__main__':
    main()
