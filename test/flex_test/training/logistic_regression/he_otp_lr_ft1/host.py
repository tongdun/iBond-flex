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

import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from flex.api import make_protocol
from flex.constants import HE_OTP_LR_FT1
from flex_test.fed_config_example import fed_conf_host

# 模型训练参数，可根据实际情况动态调整
train_param = {
    'lr': 1,
    'num_epochs': 10,
    'batch_size': 64
}


def test_train():
    # 固定随机数种子，保证每次运行网络的时候相同输入的输出是固定的
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(0)

    data = load_breast_cancer()
    host_train_data = data.data[:, 20:]  # host取20列之后的特征作为联邦训练的特征向量(训练数据共30维特征)

    # 对数据做归一化处理，归一化方法guest和host需要统一
    host_train_min = np.min(host_train_data, axis=0)
    host_train_max = np.max(host_train_data, axis=0)
    host_train_data = (host_train_data - host_train_min) / (host_train_max - host_train_min)

    # PyTorch加载训练数据
    host_dataset = TensorDataset(torch.Tensor(host_train_data))
    host_dataloader = DataLoader(host_dataset, train_param['batch_size'], drop_last=False)

    # 模型使用随机梯度下降的方法，并使用设定的学习率来最小化训练模型中的误差
    class LogisticRegression(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.theta = nn.Parameter(torch.randn((in_dim)))

    model = LogisticRegression(10)
    optimizer = torch.optim.SGD(model.parameters(), train_param['lr'])

    # 联邦通信初始化，从外部调入，与上面调用示例中的federal_info相同，根据实际部署可以调整server, role, local_id, job_id等
    federal_info = fed_conf_host

    # 安全参数，使用的加密方法为paillier加密，密钥长度为1024位，与guest保持一致
    sec_param = [['paillier', {"key_length": 1024}], ]

    # HE_OTP_LR_FT1协议初始化
    protocol = make_protocol(HE_OTP_LR_FT1, federal_info, sec_param, None)

    # 参与联邦训练过程，目标使guest的loss值不断降低或达到一定阈值
    for epoch in range(train_param['num_epochs']):
        for i, data in enumerate(host_dataloader):
            feature = data[0]
            gradient = protocol.exchange(model.theta.detach().numpy(), feature.numpy())
            optimizer.zero_grad()
            model.theta.grad = torch.Tensor(gradient)
            optimizer.step()


if __name__ == '__main__':
    test_train()
