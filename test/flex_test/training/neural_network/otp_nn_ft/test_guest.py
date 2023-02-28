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


import os
import torch
import torchvision
import torchvision.transforms as transforms

from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from flex.api import make_protocol
from flex.constants import HE_OTP_LR_FT1
from flex_test.fed_config_example import fed_conf_guest



# # transforms
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))])
#
# # datasets
# trainset = torchvision.datasets.FashionMNIST('./data',
#     download=True,
#     train=True,
#     transform=transform)
# testset = torchvision.datasets.FashionMNIST('./data',
#     download=True,
#     train=False,
#     transform=transform)


# 验证自造数据的结果正确性(协议自身)


# 在一个特定数据集下训练，loss低于多少，ks达到多少