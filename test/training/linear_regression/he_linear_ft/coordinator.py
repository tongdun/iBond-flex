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
from flex.api import make_protocol
from flex.constants import HE_LINEAR_FT
from test.fed_config_example import fed_conf_coordinator


def test_train():
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(0)

    federal_info = fed_conf_coordinator

    sec_param = [['paillier', {"key_length": 1024}], ]

    train_param = {
        'lr': 0.1,
        'num_epochs': 10,
        'iter_per_epoch': 8,
        'batch_size': 64
    }

    protocol = make_protocol(HE_LINEAR_FT, federal_info, sec_param, None)

    for i in range(train_param['iter_per_epoch'] * train_param['num_epochs']):
        protocol.exchange()


if __name__ == '__main__':
    test_train()
