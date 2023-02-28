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

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_boston, load_breast_cancer
from flex.api import make_protocol
from flex.constants import HE_LINEAR_FT
from flex_test.fed_config_example import fed_conf_host


def test_train():
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(0)

    data = load_boston()
    x_train = data.data[:, 6:]
    # x_train = x_train / np.max(np.abs(x_train), axis=0)

    x_train_min = np.min(x_train, axis=0)
    x_train_max = np.max(x_train, axis=0)
    x_train = (x_train - x_train_min) / (x_train_max - x_train_min)

    train_param = {
        'lr': 0.1,
        'num_epochs': 10,
        'iter_per_epoch': 8,
        'batch_size': 64
    }

    my_dataset = TensorDataset(torch.as_tensor(x_train))
    my_dataloader = DataLoader(my_dataset, train_param['batch_size'], drop_last=False)

    class LinearRegression(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.theta = nn.Parameter(torch.randn((in_dim)))

    model = LinearRegression(7)
    print(model.state_dict())

    # criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), train_param['lr'])

    federal_info = fed_conf_host

    sec_param = [['paillier', {"key_length": 1024}], ]

    protocol = make_protocol(HE_LINEAR_FT, federal_info, sec_param, None)

    for epoch in range(train_param['num_epochs']):
        print(f"epoch: {epoch}")
        for i, data in enumerate(my_dataloader):
            feature = data[0]
            u = (feature * model.theta).sum(dim=1)
            result = protocol.exchange(u.detach().numpy())
            result = torch.as_tensor(result)
            gradient = torch.mean(feature * result.unsqueeze(-1), dim=0)
            optimizer.zero_grad()
            model.theta.grad = torch.as_tensor(gradient).float()
            optimizer.step()
            print('theta:', model.theta)


if __name__ == '__main__':
    test_train()
