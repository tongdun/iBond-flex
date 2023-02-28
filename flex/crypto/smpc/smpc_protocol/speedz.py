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

"""SPDZ for MPC
"""
import logging
import torch

from flex.crypto.smpc.smpc_protocol.\
    additive_sharing_tensor import ASTTH


class SPDZTH(ASTTH):
    """Mul and Matmul
    """
    def mul(self, x_sh, y_sh):
        """Elementwise multiply of x and y

        Arguments:
            x_sh: LHS of multiply
                for Arbiter: x_sh be zeros of the same size
            y_sh: RHS of multiply
                for Arbiter: y_sh be zeros of the same size

        Returns:
            share of x*y
                for Arbiter: None
        """
        return self.mul_cmd(x_sh, y_sh, torch.mul)

    def matmul(self, x_sh, y_sh):
        """Matrix multiply of x and y

        Arguments:
            x_sh: LHS of multiply
                for Arbiter: x_sh be zeros of the same size
            y_sh: RHS of multiply
                for Arbiter: y_sh be zeros of the same size

        Returns:
            share of x*y
                for Arbiter: None
        """
        return self.mul_cmd(x_sh, y_sh, torch.matmul)

    def mul_cmd(self, x_sh, y_sh, cmd):
        """cmd(x, y)

        Arguments:
            x_sh: LHS of cmd
                for Arbiter: x_sh be zeros of the same size
            y_sh: RHS of cmd
                for Arbiter: y_sh be zeros of the same size
            cmd: the operator

        Returns:
            share of x*y
                for Arbiter: None
        """
        self.tag += 1
        num_party = self.num_party
        # gen (a,b,c)
        if self.role == 'ARBITER':
            logging.info('Here is arbiter generating triple')
            triple = self._generate_triple_cmd(x_sh.shape, y_sh.shape, cmd)
            a_shs = self._generate_shares(triple[0], num_party)
            b_shs = self._generate_shares(triple[1], num_party)
            c_shs = self._generate_shares(triple[2], num_party)

            logging.info('Triple generated. Sending...')
            # tensors = list(zip(a_shs, b_shs, c_shs))
            self.broadcast_slice_to_parties_from(list(zip(a_shs,
                                                          b_shs,
                                                          c_shs)),
                                                 num_party)

            self.tag += 1  # for reconstruct
            return None

        # Party
        logging.info('Waiting for triple_%s%i', self.jobid, self.tag)
        triple_sh = self.broadcast_slice_to_parties_from(None, num_party)

        # reconstruct delta, epsilon
        logging.info('Reconstruct delta_epsilon...')
        delta, epsilon = self.reconstruct((x_sh - triple_sh[0],
                                           y_sh - triple_sh[1]))

        # put together
        logging.info('Calculating final result')
        is_first_party = 1 if self.world_id == 0 else 0
        if cmd == torch.matmul:
            res = self.machine_matmul(delta, triple_sh[1] + (epsilon if is_first_party else 0)) \
                + self.machine_matmul(triple_sh[0], epsilon) \
                + triple_sh[2]
        else:
            res = cmd(delta, epsilon * is_first_party + triple_sh[1]) \
                + cmd(triple_sh[0], epsilon) + triple_sh[2]

        if self.precision > 0:
            res = self._trunct(res)
            return res

        return res
