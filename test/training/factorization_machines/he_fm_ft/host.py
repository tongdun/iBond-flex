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
import time

import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from flex.api import make_protocol
from flex.constants import HE_FM_FT, HE_FM_FP
from test.fed_config_example import fed_conf_host

# 模型训练参数，可根据实际情况动态调整
train_param = {
    'lr': 0.01,
    'num_epochs': 5,
    'batch_size': 64
}


def test_train():
    # 固定随机数种子，保证每次运行网络的时候相同输入的输出是固定的
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(0)

    host_train_data = pd.read_csv('/mnt/nfs/datasets/shap_finance/train_host.csv')
    host_train_data = host_train_data.iloc[:, 1:]

    # 对数据做归一化处理，归一化方法guest和host需要统一
    host_train_min = np.min(host_train_data, axis=0)
    host_train_max = np.max(host_train_data, axis=0)
    host_train_data = (host_train_data - host_train_min) / (host_train_max - host_train_min)

    # PyTorch加载训练数据
    host_dataset = TensorDataset(torch.Tensor(np.array(host_train_data)))
    host_dataloader = DataLoader(host_dataset, train_param['batch_size'], drop_last=False)

    # 模型使用随机梯度下降的方法，并使用设定的学习率来最小化训练模型中的误差
    class FactorizationMachines(nn.Module):
        def __init__(self, in_dim, m):
            super().__init__()
            self.theta = nn.Parameter(torch.Tensor(in_dim, ).uniform_(-1, 1))
            self.v = nn.Parameter(torch.Tensor(in_dim, m).uniform_(-1, 1))

    model = FactorizationMachines(10, 5)
    optimizer = torch.optim.SGD(model.parameters(), train_param['lr'])

    # 联邦通信初始化，从外部调入，与上面调用示例中的federal_info相同，根据实际部署可以调整server, role, local_id, job_id等
    federal_info = fed_conf_host

    sec_param = []

    protocol = make_protocol(HE_FM_FT, federal_info, sec_param, None)
    protocol2 = make_protocol(HE_FM_FP, federal_info, sec_param, None)

    # 参与联邦训练过程，目标使guest的loss值不断降低或达到一定阈值
    for epoch in range(train_param['num_epochs']):
        for i, data in enumerate(host_dataloader):
            feature = data[0]
            gradient_theta, gradient_v = protocol.exchange(model.theta.detach().numpy(),
                                                           model.v.detach().numpy(),
                                                           feature.numpy())
            predict = protocol2.exchange(model.theta.detach().numpy(),
                                         model.v.detach().numpy(),
                                         feature.numpy())

            optimizer.zero_grad()
            model.theta.grad = torch.Tensor(gradient_theta)
            model.v.grad = torch.Tensor(gradient_v)
            optimizer.step()


if __name__ == '__main__':
    start = time.time()
    test_train()
    end = time.time()
    print(f'time is {end - start}')
