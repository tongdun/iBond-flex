#
#  Copyright 2020 The FLEX Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# 验证自造数据的结果正确性(协议自身)
# 在一个特定数据集下训练，loss低于多少，ks达到多少

import os
import time

import torch
import numpy as np
import pandas as pd
import pylab as pl
from torch import nn
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader

from flex.api import make_protocol
from flex.constants import HE_FM_FT, HE_FM_FP
from flex_test.fed_config_example import fed_conf_guest

# 模型训练参数，可根据实际情况动态调整
train_param = {
    'lr': 0.01,
    'num_epochs': 5,
    'batch_size': 64
}


def test_train():
    def setup_seed(seed):
        torch.manual_seed(seed)  # 为CPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        torch.backends.cudnn.deterministic = True  # CuDNN卷积使用确定性算法

    setup_seed(0)  # 固定随机数种子，保证每次运行网络的时候相同输入的输出是固定的

    guest_train_data = pd.read_csv('/mnt/nfs/datasets/shap_finance/train_guest.csv')

    # guest提供目标label数据
    guest_train_label = guest_train_data['Label']
    guest_train_data = guest_train_data.iloc[:, 2:]

    # 对数据先做归一化处理，归一化方法为max-min方法
    guest_train_min = np.min(guest_train_data, axis=0)
    guest_train_max = np.max(guest_train_data, axis=0)
    guest_train_data = (guest_train_data - guest_train_min) / (guest_train_max - guest_train_min)


    # PyTorch加载训练数据
    guest_dataset = TensorDataset(torch.Tensor(np.array(guest_train_data)), torch.Tensor(np.array(guest_train_label)))
    guest_dataloader = DataLoader(guest_dataset, train_param['batch_size'], drop_last=False)

    # 模型使用随机梯度下降的方法，并使用设定的学习率来最小化训练模型中的误差
    class FactorizationMachines(nn.Module):
        def __init__(self, in_dim, m):
            super().__init__()
            self.theta = nn.Parameter(torch.Tensor(in_dim,).uniform_(-1, 1))
            self.v = nn.Parameter(torch.Tensor(in_dim, m).uniform_(-1, 1))

    model = FactorizationMachines(2, 5)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), train_param['lr'])
    # 联邦通信初始化，从外部调入，与上面调用示例中的federal_info相同，根据实际部署可以调整server, role, local_id, job_id等
    federal_info = fed_conf_guest

    sec_param = []

    protocol = make_protocol(HE_FM_FT, federal_info, sec_param, None)
    protocol2 = make_protocol(HE_FM_FP, federal_info, sec_param, None)
    loss_list = []

    # 训练过程，目标使loss值不断降低，将每次训练的loss值记录
    for epoch in range(train_param['num_epochs']):
        for i, data in enumerate(guest_dataloader):
            feature, label = data
            gradient_theta, gradient_v = protocol.exchange(model.theta.detach().numpy(),
                                                             model.v.detach().numpy(),
                                                             feature.numpy(), label.numpy())
            predict = protocol2.exchange(model.theta.detach().numpy(),
                                         model.v.detach().numpy(),
                                         feature.numpy())
            loss = criterion(torch.Tensor(predict), label)

            predict[predict >= 0.5] = 1
            predict[predict < 0.5] = 0

            acc = metrics.accuracy_score(label, predict)

            loss_list.append(loss.item())
            optimizer.zero_grad()
            model.theta.grad = torch.Tensor(gradient_theta)
            model.v.grad = torch.Tensor(gradient_v)
            optimizer.step()

            print(f'loss is {loss}')
            print(f'predict is {predict}')
            print(f'acc = {acc}')

    # 为了观察方便，打印loss值随着训练次数迭代的过程，命名guest_train_results_loss.png并保存在当前目录下
    pl.plot(list(range(1, len(loss_list) + 1)), loss_list, 'r-', label='loss value')
    pl.legend()
    pl.xlabel('iters')
    pl.ylabel('loss')
    pl.title('factorization machines loss in training')
    pl.show()
    pl.savefig(os.path.join(os.path.dirname(__file__), "guest_train_results_loss.png"))


if __name__ == '__main__':
    start = time.time()
    test_train()
    end = time.time()
    print(f'time is {end - start}')
