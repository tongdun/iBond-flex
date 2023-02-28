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

"""MPC: basic ops for additive share tensors
"""
from concurrent.futures import ThreadPoolExecutor, wait
import torch
import numpy as np
import tensorflow as tf

from flex.crypto.smpc.smpc_protocol.io.ibond_io \
    import IBondIOTH as IOModule


class ASTTH(IOModule):
    """Torch backend of additive sharing tensor
    """
    def __init__(self, jobid, conf_file, ibond_conf):
        super().__init__(jobid, conf_file, ibond_conf)

        self.tag = 0

    def _generate_shares(self, secret, n_workers, field_size=None):
        """Generate additive sharings of secret

        Arguments:
            secret: the secret to be shared
            n_workers: the number of pieces of secret to be generated
            L: the field to work with
        Returns:
            list of shares of secrets
        """
        min_value = -(self.field // 2)
        max_value = (self.field - 1) // 2
        if field_size:
            min_value = -(field_size // 2)
            max_value = (field_size - 1) // 2
        random_type = torch.LongTensor
        if not isinstance(secret, random_type):
            secret = secret.type(random_type)

        random_shares = [random_type(secret.shape)
                         for _ in range(n_workers - 1)]
        for share in random_shares:
            share.random_(min_value, max_value)

        if n_workers == 1:
            return [secret]
        shares = []
        for i in range(n_workers):
            if i == 0:
                share = random_shares[i]
            elif i < n_workers - 1:
                share = random_shares[i] - random_shares[i - 1]
            else:
                share = secret - random_shares[i - 1]
            if field_size:
                share = self._cutoff(share, field_size)
            shares.append(share)
        return shares

    def cutoff_precision(self, tensor):
        """Restore precision
        """
        base = 10 ** self.precision
        ret = self._cutoff(tensor * base, self.field)
        ret = ret // base
        return ret

    def _generate_zero_shares(self, n_workers, shape, field_size=None):
        """Generate additive sharings of zero

        Arguments:
            n_workers: the number of pieces of secret to be generated
            shape: the shape of zero tensor
            L: the field work with
        Returns:
            list of shares of zero
        """
        if shape is None:
            shape = [1]
        zero = torch.zeros(*shape, dtype=torch.int64)
        shares = self._generate_shares(zero, n_workers, field_size)
        return shares

    def _generate_triple_cmd(self, a_size, b_size, cmd):
        target = self.field
        triple_a = torch.randint(-(target // 2),
                                 (target - 1) // 2,
                                 a_size)
        triple_b = torch.randint(-(target // 2),
                                 (target - 1) // 2,
                                 b_size)
        if cmd == torch.matmul:
            triple_c = self.machine_matmul(triple_a, triple_b)
        else:
            triple_c = cmd(triple_a, triple_b)
        return triple_a, triple_b, triple_c

    def _generate_triple_matmul(self, a_size, b_size):
        return self._generate_triple_cmd(a_size, b_size, torch.matmul)

    def _generate_triple_mul(self, a_size, b_size):
        return self._generate_triple_cmd(a_size, b_size, torch.mul)

    def _cutoff(self, tensor, field_size):
        min_val = -(field_size // 2)
        max_val = (field_size - 1) // 2
        mask_over = tensor > max_val
        mask_undr = tensor < min_val
        if mask_over.any():
            return self._cutoff(tensor - (mask_over.long() * field_size),
                                field_size)
        if mask_undr.any():
            return self._cutoff(tensor + (mask_undr.long() * field_size),
                                field_size)
        return tensor

    def _trunct(self, tensor, precision=None):
        if precision is None:
            precision = self.precision
        if precision == 0:
            return tensor
        tensor = tensor // (10 ** precision)
        tensor = tensor.long()
        return tensor

    def _untrunct(self, tensor, precision=None):
        if precision is None:
            precision = self.precision
        tensor *= 10 ** precision
        tensor = tensor.long()

        return tensor

    def share_raw(self, raw):
        """Share raw to other parties

        Arguments:
            raw: the data to be shared
                for Arbiter: could be None
        Returns:
            [r0, r1, ... , rN]
                for Arbiter: None
        """
        self.tag += 1
        if self.role == 'ARBITER':
            return None
        num_party = self.num_party

        with ThreadPoolExecutor(max_workers=8) as executor:
            _ = [executor.submit(self.send_msg,
                                 party,
                                 raw,
                                 f'raw_{self.jobid}{self.tag}_'
                                 f'{self.world_id}')
                 for party in range(num_party) if party != self.world_id]
            recv_tasks = [executor.submit(self.receive_msg,
                                          f'raw_{self.jobid}{self.tag}_'
                                          f'{party}',
                                          party)
                          for party in range(num_party) if
                          party != self.world_id]
            wait(recv_tasks)

        ret = [task.result() for task in recv_tasks]
        ret.insert(self.world_id, raw)

        return ret

    def share_secrets(self, secret, field_size=None):
        """Share secret to other parties

        Arguments:
            secret: the secret to be shared
                for Arbiter: could be None
        Returns:
            [sh0, sh1, ... , shN]
                for Arbiter: None
        """
        self.tag += 1
        if self.role == 'ARBITER':
            return None
        num_party = self.num_party

        shares = self._generate_shares(secret, num_party, field_size)
        with ThreadPoolExecutor(max_workers=8) as executor:
            _ = [executor.submit(self.send_msg,
                                 party,
                                 shares[party],
                                 f'share_{self.jobid}{self.tag}_'
                                 f'{self.world_id}')
                 for party in range(num_party) if party != self.world_id]
            recv_tasks = [executor.submit(self.receive_msg,
                                          f'share_{self.jobid}{self.tag}_'
                                          f'{party}',
                                          party)
                          for party in range(num_party) if
                          party != self.world_id]
            wait(recv_tasks)

        ret = [task.result() for task in recv_tasks]
        ret.insert(self.world_id, shares[self.world_id])

        return ret

    def share_zero_from(self, shape, src, field_size=None):
        """Party/Arbiter src generate share of zero and sent to other parties.

        Args:
            shape: the shape of zero
                for Arbiter if not src: could be None
            src: the party's world_id who will share
            L: field size
        Returns:
            sh: the share of zero
                for Arbiter: None
        """
        self.tag += 1
        if self.world_id == src:
            num_party = self.num_party
            shares = self._generate_zero_shares(num_party, shape, field_size)
            with ThreadPoolExecutor(max_workers=8) as executor:
                _ = [executor.submit(self.send_msg,
                                     party,
                                     shares[party],
                                     f'sharezerofrom_{src}_{self.jobid}'
                                     f'{self.tag}')
                     for party in range(num_party) if party != self.world_id]
            if self.role == 'PARTY':
                return shares[self.world_id]
            return None
        if self.role == 'ARBITER':
            return None
        return self.receive_msg(f'sharezerofrom_{src}_{self.jobid}'
                                f'{self.tag}', src)

    def share_secret_from(self, secret, src, field_size=None):
        """Party/Arbiter src share secret to other parties.

        Args:
            secret: to be shared
                for Arbiter if not src: could be None
            src: the party's world_id who will share
            L: field size
        Returns:
            sh: the share of secret
                for Arbiter: None
        """
        self.tag += 1
        if self.world_id == src:
            num_party = self.num_party
            shares = self._generate_shares(secret, num_party, field_size)
            with ThreadPoolExecutor(max_workers=8) as executor:
                _ = [executor.submit(self.send_msg,
                                     party,
                                     shares[party],
                                     f'sharefrom_{src}_{self.jobid}{self.tag}')
                     for party in range(num_party) if party != self.world_id]
            if self.role == 'PARTY':
                return shares[self.world_id]
            return None
        if self.role == 'ARBITER':
            return None
        return self.receive_msg(f'sharefrom_{src}_{self.jobid}{self.tag}', src)

    def broadcast_to_parties_from(self, tensor, src):
        """Broadcast tensor to all parties, from src

        Args:
            tensor: to be broadcasted
                for Arbiter if not src: could be None
            src: the sender
        Returns:
            tensor: which is broadcasted
                for Arbiter: None if not src, tensor if src
        """
        self.tag += 1
        num_party = self.num_party
        if self.world_id == src:
            with ThreadPoolExecutor(max_workers=8) as executor:
                _ = [executor.submit(self.send_msg,
                                     party,
                                     tensor,
                                     f'broadcastfrom_{src}_'
                                     f'{self.jobid}{self.tag}')
                     for party in range(num_party) if party != self.world_id]
            return tensor
        if self.role == 'PARTY':
            return self.receive_msg(f'broadcastfrom_{src}_'
                                    f'{self.jobid}{self.tag}', src)
        return None

    def broadcast_slice_to_parties_from(self, tensors, src):
        """Broadcast tensors[i] to each parties[i], from src

        Args:
            tensors: to be broadcasted
                for Arbiter if not src: could be None
            src: the sender
        Returns:
            slice: which is broadcasted
                for Arbiter: None
        """
        self.tag += 1
        num_party = self.num_party
        if self.world_id == src:
            with ThreadPoolExecutor(max_workers=8) as executor:
                _ = [executor.submit(self.send_msg,
                                     party,
                                     tensors[party],
                                     f'broadcastslicefrom_{src}_'
                                     f'{self.jobid}{self.tag}')
                     for party in range(num_party) if party != self.world_id]
            if self.role == 'PARTY':
                return tensors[self.world_id]
            return None
        if self.role == 'PARTY':
            return self.receive_msg(f'broadcastslicefrom_{src}_'
                                    f'{self.jobid}{self.tag}', src)
        return None

    def broadcast_to_arbiter(self, tensor):
        """Broadcast tensor to all parties, from src

        Args:
            tensor: to be broadcasted
                for Arbiter: could be None
        Returns:
            [tensor]: which is broadcasted
                for Party: None
                for Arbiter: [tensor]
        """
        self.tag += 1
        num_party = self.num_party
        dst = self.num_party  # arbiter

        if self.role == 'PARTY':
            self.send_msg(dst,
                          tensor,
                          f'broadcast_to_arbiter_{self.world_id}'
                          f'{self.jobid}{self.tag}')
            return None
        # Arbiter
        with ThreadPoolExecutor(max_workers=8) as executor:
            recv_tasks = [executor.submit(self.receive_msg,
                                          f'broadcast_to_arbiter_{party}'
                                          f'{self.jobid}{self.tag}',
                                          party)
                          for party in range(num_party)]
            wait(recv_tasks)
        ret = [task.result() for task in recv_tasks]
        return ret

    def reconstruct(self, x_sh):
        """Reconstruct from share of secret. It will not cutoff to field,
        so you need to do it yourself when needed.

        Args:
            x_sh: to be reconstructed
                for Arbiter: could be None
        Returns:
            x: the secret
                for Arbiter: None
        """
        # x_sh canbe of tensor or tuple of tensors
        self.tag += 1
        num_party = self.num_party

        if self.role == 'ARBITER':
            return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            _ = [executor.submit(self.send_msg,
                                 party,
                                 x_sh,
                                 f'reconstruct_{self.jobid}{self.tag}_'
                                 f'{self.world_id}')
                 for party in range(num_party) if party != self.world_id]
            recv_tasks = [executor.submit(self.receive_msg,
                                          f'reconstruct_{self.jobid}'
                                          f'{self.tag}_{party}',
                                          party)
                          for party in range(num_party)
                          if party != self.world_id]
            wait(recv_tasks)
        ret = [task.result() for task in recv_tasks]
        ret.insert(self.world_id, x_sh)

        if isinstance(x_sh, tuple):
            return tuple(map(sum, zip(*ret)))
        return sum(ret)

    def reconstruct_for(self, x_sh, dst):
        """Reconstruct from share of secret for dst. It will not cutoff to field,
        so you need to do it yourself when needed.

        Args:
            x_sh: to be reconstructed
                for Arbiter: could be None
        Returns:
            x: the secret for dst
            None: for other parties
                for Arbiter: None
        """
        # x_sh canbe of tensor or tuple of tensors
        self.tag += 1
        num_party = self.num_party

        if self.role == 'ARBITER':
            return None

        if self.world_id != dst:
            self.send_msg(dst, x_sh, f'reconstruct_{self.jobid}{self.tag}_{self.world_id}')
            return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            recv_tasks = [executor.submit(self.receive_msg,
                                          f'reconstruct_{self.jobid}'
                                          f'{self.tag}_{party}',
                                          party)
                          for party in range(num_party)
                          if party != self.world_id]
            wait(recv_tasks)
        ret = [task.result() for task in recv_tasks]
        ret.insert(self.world_id, x_sh)

        if isinstance(x_sh, tuple):
            return tuple(map(sum, zip(*ret)))
        return sum(ret)

    def machine_matmul(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """
        determine which matmul is best available
        """
        if self.cuda:
            return ASTTH.npmatmul(lhs, rhs)
        return ASTTH.npmatmul(lhs, rhs)

    @staticmethod
    def tfmatmul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """matmul using tensorflow for performance
        """
        lhs_tf = tf.convert_to_tensor(lhs.numpy())
        rhs_tf = tf.convert_to_tensor(rhs.numpy())
        ret_tf = tf.matmul(lhs_tf, rhs_tf)
        ret = torch.from_numpy(ret_tf.numpy())
        return ret

    @staticmethod
    def npmatmul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """matmul using numpy for performance
        """
        ret = np.matmul(lhs.numpy(), rhs.numpy())
        ret = torch.from_numpy(ret)
        return ret
