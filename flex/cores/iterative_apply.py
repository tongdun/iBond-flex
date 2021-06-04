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

from typing import Union

import numpy as np
import torch

from flex.crypto.paillier.encryptor import PaillierEncryptor
from flex.crypto.paillier.decryptor import PaillierDecryptor


def iterative_encryption(encryptor: PaillierEncryptor, plaintext: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
    if isinstance(plaintext, list):
        return [iterative_encryption(encryptor, item) for item in plaintext]
    elif isinstance(plaintext, np.ndarray):
        return encryptor.encrypt(plaintext)
    else:
        raise TypeError(f"Type {type(plaintext)} is not valid.")


def iterative_decryption(decryptor: PaillierDecryptor, ciphertext: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
    if isinstance(ciphertext, list):
        return [iterative_decryption(decryptor, item) for item in ciphertext]
    elif isinstance(ciphertext, np.ndarray):
        return decryptor.decrypt(ciphertext)
    else:
        raise TypeError(f"Type {type(ciphertext)} is not valid.")


def iterative_divide(input: Union[list, np.ndarray, torch.Tensor, float, int], scalar) -> Union[
    list, np.ndarray, torch.Tensor, float, int]:
    if isinstance(input, list):
        return [iterative_divide(item, scalar) for item in input]
    elif isinstance(input, (np.ndarray, torch.Tensor, float, int)):
        return input / scalar
    else:
        raise TypeError(f"Type {type(input)} is not valid.")
