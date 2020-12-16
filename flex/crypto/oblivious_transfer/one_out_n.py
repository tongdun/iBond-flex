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

import secrets
import hashlib
import base64
from typing import List

from flex.tools.ionic import VariableChannel
from py_ecc.secp256k1.secp256k1 import multiply, add, G, P

from Crypto.Cipher import AES
from Crypto import Random


class OneOutN_OT(object):
    def __init__(self, n: int, remote_id: str):
        """

        Args:
            n: int, message number that server need to provide.

        """
        self.n = n
        self.secrets_generator = secrets.SystemRandom()
        self.bs = AES.block_size
        self.var_chan = VariableChannel('oblivious_transfer', remote_id)


    def _pad(self, s):
        return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s) - 1:])]

    def client(self, index: int) -> str:
        """

        Args:
            index: int, which index the client want to get from server's input list.

        Returns: str, message

        """
        param_s = self.var_chan.recv(tag='ot_param_s')

        x = self.secrets_generator.randint(1, P-1)
        param_r = add(multiply(param_s, int(index)), multiply(G, x))
        self.var_chan.send(param_r, tag='ot_param_r')

        key_pub = multiply(param_s, x)
        key = hashlib.sha256(str(int("04" + "%064x" % key_pub[0] + "%064x" % key_pub[1], 16)).encode('utf-8')).digest()

        recv_msg_list = self.var_chan.recv(tag='ot_encrypted_messages')

        if not (0 <= index <= self.n - 1):
            raise ValueError(f"Index {index} is supposed to be a number between 1 and n.")

        recv_msg = base64.b64decode(recv_msg_list[index])
        iv = recv_msg[:AES.block_size]
        cipher = recv_msg[AES.block_size:]
        aes = AES.new(key, AES.MODE_CBC, iv)
        decrypted_message = self._unpad(aes.decrypt(cipher)).decode('utf-8')
        return decrypted_message

    def server(self, msg_list: List[str]) -> None:
        """

        Args:
            msg_list: list, list consist of string.

        Returns: None

        """
        y = self.secrets_generator.randint(1, P-1)
        param_s = multiply(G, y)

        self.var_chan.send(param_s, tag='ot_param_s')
        param_r = self.var_chan.recv(tag='ot_param_r')

        encrypted_msg_list = []
        iv = Random.new().read(AES.block_size)

        for i, msg in enumerate(msg_list):
            key_pub = multiply(add(param_r, multiply(param_s, (-1) * i)), y)
            key = hashlib.sha256(
                str(int("04" + "%064x" % key_pub[0] + "%064x" % key_pub[1], 16)).encode('utf-8')).digest()
            aes = AES.new(key, AES.MODE_CBC, iv)
            cipher = aes.encrypt(self._pad(msg).encode())
            encrypted_msg_list.append(base64.b64encode(iv + cipher))

        self.var_chan.send(encrypted_msg_list, tag='ot_encrypted_messages')
        return

