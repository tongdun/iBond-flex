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


from flex.crypto.fpe.api import generate_fpe_encryptor
import numpy as np

def test_encryptor_decryptor():
    key = [np.random.bytes(16), np.random.bytes(32)]
    n = [7, 20]
    t = [b'', b'123']

    for i in range(2):
        encryptor = generate_fpe_encryptor(key[i], n[i], t[i], method='ff1', encrypt_algo='aes')
        x = np.linspace(0, 2 ** n[i] - 1, 2 ** n[i]).astype(int)

        y = encryptor.encrypt(x)
        z = encryptor.decrypt(y)

        assert np.all(x == np.unique(y))
        assert np.all(x == z)
