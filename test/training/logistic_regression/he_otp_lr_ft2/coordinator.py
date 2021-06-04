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

from flex.constants import HE_OTP_LR_FT2
from flex.api import make_protocol

from test.fed_config_example import fed_conf_coordinator

train_param = {
    'num_epochs': 10,
    'train_rounds': 9  # train_rounds根据数据大小/batch_size得到
}


def test_train():
    # 联邦通信初始化，从外部调入，与上面调用示例中的federal_info相同，根据实际部署可以调整server, role, local_id, job_id等
    federal_info = fed_conf_coordinator

    # 安全参数，使用的加密方法为paillier加密，密钥长度为1024位
    sec_param = [['paillier', {"key_length": 1024}], ]

    # HE_OTP_LR_FT2协议初始化
    trainer = make_protocol(HE_OTP_LR_FT2, federal_info, sec_param)

    # 训练过程
    for epoch in range(train_param['num_epochs']):
        for i in range(train_param['train_rounds']):
            trainer.exchange()


if __name__ == '__main__':
    test_train()