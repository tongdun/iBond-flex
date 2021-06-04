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

"""FixedPrecisionTensor: represent float type using LongTensor
"""
import numpy as np
import torch


class FixedPrecisionTensor(torch.Tensor):
    """Fiexed precision tensor
    """
    @staticmethod
    def __new__(cls, data, precision, *args, **kwargs):
        data = torch.as_tensor(data) * (10 ** precision)
        if data.dtype is torch.float or data.dtype is torch.double:
            data = torch.round(data)
        data = torch.as_tensor(data, dtype=torch.int64)
        tensor = torch.Tensor._make_subclass(cls, data, *args, **kwargs)
        tensor.precision = precision
        return tensor

    def __init__(self, data, precision):  # pylint: disable-all
        self.precision = precision

    def __repr__(self):
        return f'FP({self.precision}):data:{super().__repr__()}'

    def __str__(self):
        return self.__repr__()


def make_fixed_precision_tensor(tensor, precision):
    """Convert tensor to FixedPrecisionTensor
    """
    return FixedPrecisionTensor(tensor, precision)


def decode_tensor(tensor,
                  precision,
                  target_type=torch.Tensor,
                  target_dtype=torch.float):
    """Convert back to FloatTensor
    """
    if target_type == torch.Tensor:
        ret = torch.as_tensor(tensor, dtype=torch.double)
        ret /= 10 ** precision
        ret = torch.as_tensor(ret, dtype=target_dtype)
        return ret
    if target_type == np.ndarray:
        ret = torch.as_tensor(tensor, dtype=torch.double)
        ret /= 10 ** precision
        ret = ret.numpy()
        if target_dtype in [np.int, np.int0, np.int16, np.int64, np.int8]:
            ret = ret.round()
        ret = ret.astype(target_dtype)
        return ret
    raise NotImplementedError(f'target_type: {target_type} not supported')


if __name__ == '__main__':
    raw = torch.as_tensor([0.68, 0.7661, 0.9999])
    print('raw:', raw)
    enc = FixedPrecisionTensor(raw, 3)
    print('enc:', enc)
    dec = decode_tensor(enc, 3)
    print('dec:', dec)
    dec_np = decode_tensor(enc, 3, np.ndarray, np.int)
    print('dec_np:', dec_np)
