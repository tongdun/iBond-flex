"""secure nn implementation
"""
import math
import random
import torch

from .speedz import SPDZTH


class secureNNTH(SPDZTH):
    # secureNN
    @staticmethod
    def _get_n_bits(field):
        return round(math.log(field, 2))

    @staticmethod
    def _decompose64(tensor, field=2**64):
        """decompose a tensor into its binary representation."""
        shp = list(tensor.shape) + [64]
        tensor = tensor.view(-1)
        ret = []
        for v_item in tensor:
            val = v_item.item()
            if val >= 0:
                tmp = list(map(int, list('{:0>64b}'.format(val))))
                tmp.reverse()
                ret.append(tmp)
            else:
                tmp = list(map(int, list('{:0>64b}'.format(field + val))))
                tmp.reverse()
                ret.append(tmp)
        # res = torch.tensor(ret, dtype=torch.int64)
        res = torch.LongTensor(ret)
        res = res.reshape(shp)
        return res

    @staticmethod
    def _decompose(tensor, field):
        """decompose a tensor into its binary representation."""
        n_bits = secureNNTH._get_n_bits(field)
        powers = torch.arange(n_bits, dtype=torch.int64)
        for _ in range(len(tensor.shape)):
            powers = powers.unsqueeze(0)
        tensor = tensor.unsqueeze(-1)
        moduli = 2 ** powers
        tensor = torch.fmod((tensor / moduli.type_as(tensor)), 2)
        return tensor

    def select_share(self, alpha_sh, x_sh, y_sh):
        """ Performs select share protocol
        If the bit alpha_sh is 0, x_sh is returned
        If the bit alpha_sh is 1, y_sh is returned

        Args:
            x_sh: the first share to select
            y_sh: the second share to select
            alpha_sh: the bit to choose between x_sh and y_sh
                for Arbiter: x_sh, y_sh, alpha_sh should be same shape

        Return:
            z_sh = (1 - alpha_sh) * x_sh + alpha_sh * y_sh
                for Arbiter: None
        """
        self.tag += 1

        u_sh = self.share_zero_from(alpha_sh.shape, 0)
        c_sh = self.mul(alpha_sh, y_sh - x_sh)

        if self.role == 'PARTY':
            return x_sh + c_sh + u_sh
        return None

    def select_share00(self, alpha_sh, x_sh, y_sh):
        """ Performs select share protocol
        If the bit alpha_sh is -1, x_sh is returned
        If the bit alpha_sh is 1, y_sh is returned

        Args:
            x_sh: the first share to select
            y_sh: the second share to select
            alpha_sh: the bit to choose between x_sh and y_sh
                for Arbiter: x_sh, y_sh, alpha_sh should be same shape

        Return:
            z_sh = (1/2 - 1/2*alpha_sh) * x_sh + (1/2 + 1/2*alpha_sh) * y_sh
                for Arbiter: None
        """
        self.tag += 1

        u_sh = self.share_zero_from(alpha_sh.shape, 0)
        c_sh = self.mul(alpha_sh, y_sh - x_sh)

        if self.role == 'PARTY':
            return (x_sh + y_sh + c_sh) // 2 + u_sh
        return None

    def private_compare(self, x_bit_sh, r, beta, L):
        """
        Perform privately x > r

        args:
            x_bit_sh: the private tensor
                for Arbiter: could be None
            r: the threshold commonly held by the workers
                for Arbiter: could be None
            beta: a boolean commonly held by the workers to
                hide the result of computation for the crypto provider
                for Arbiter: could be None
            L(int): field size for r
                for Arbiter: could be None

        return:
            β′ = β ⊕ (x > r).
                for Arbiter: beta_prime
        """
        if L is None:
            L = self.field
        self.tag += 1
        p = 67

        s, perm = None, None
        if self.world_id == 0:
            s = torch.randint(1, p, x_bit_sh.shape)
            perm = torch.randperm(x_bit_sh.shape[-1])

        ret = self.broadcast_to_parties_from((s, perm), 0)
        if self.role == 'PARTY':
            s, perm = ret

        mask = None
        if self.role == 'PARTY':
            # 1)
            t = r + 1
            t_bit = secureNNTH._decompose64(t)
            r_bit = secureNNTH._decompose64(r)

            # if beta == 0
            # 5)
            j = 1 if self.world_id == 0 else 0
            w = x_bit_sh + j * r_bit - 2 * r_bit * x_bit_sh

            # 6)
            wc = w.flip(-1).cumsum(-1).flip(-1) - w
            c_beta0 = j * r_bit - x_bit_sh + j + wc
            c_beta0[..., -1] = 2 * j - c_beta0[..., -1]

            # elif beta == 1 AND r != 2^l - 1
            # 8)
            w = x_bit_sh + j * t_bit - 2 * t_bit * x_bit_sh
            # 9)
            wc = w.flip(-1).cumsum(-1).flip(-1) - w
            c_beta1 = -j * t_bit + x_bit_sh + j + wc
            c_beta1[..., -1] = 2 * j - c_beta1[..., -1]

            # Mask combination to execute the if / else statements of 4), 7)
            # no use beta.unsqueeze_(-1) # side effect
            c = (1 - beta.unsqueeze(-1)) * c_beta0 \
                + beta.unsqueeze(-1) * c_beta1

            # 14)
            # Hide c values
            mask = s * c

            # Permute the mask
            idx = [slice(None)] * (len(x_bit_sh.shape) - 1) + [perm]
            mask = mask[idx]

        # send to party C
        ret = self.broadcast_to_arbiter(mask)

        beta_prime = None
        if self.role == 'ARBITER':
            result = sum(ret)
            result = self._cutoff(result, p)
            # print(result)
            # beta_prime = torch.any(result == 0).item()
            beta_prime = (result == 0).sum(-1)
            # print(beta_prime)

        beta_prime = self.broadcast_to_parties_from(beta_prime, self.num_party)

        return beta_prime

    def private_compare_unsigned(self, x_bit_sh, r, beta, L):
        """
        Perform privately x > r, treated as unsigned number

        args:
            x_bit_sh: the private tensor
                for Arbiter: could be None
            r: the threshold commonly held by the workers
                for Arbiter: could be None
            beta: a boolean commonly held by the workers to
                hide the result of computation for the crypto provider
                for Arbiter: could be None
            L(int): field size for r
                for Arbiter: could be None

        return:
            β′ = β ⊕ (x > r).
                for Arbiter: beta_prime
        """
        if L is None:
            L = self.field
        self.tag += 1
        p = 67

        s, u, perm = None, None, None
        if self.world_id == 0:
            s = torch.randint(1, p, x_bit_sh.shape)
            u = torch.randint(1, p, x_bit_sh.shape)
            perm = torch.randperm(x_bit_sh.shape[-1])

        ret = self.broadcast_to_parties_from((s, u, perm), 0)
        if self.role == 'PARTY':
            s, u, perm = ret

        mask = None
        if self.role == 'PARTY':
            # 1)
            t = r + 1
            t_bit = secureNNTH._decompose64(t)
            r_bit = secureNNTH._decompose64(r)

            # if beta == 0
            # 5)
            j = 1 if self.world_id == 0 else 0
            w = x_bit_sh + j * r_bit - 2 * r_bit * x_bit_sh

            # 6)
            wc = w.flip(-1).cumsum(-1).flip(-1) - w
            c_beta0 = j * r_bit - x_bit_sh + j + wc

            # elif beta == 1 AND r != 2^l - 1
            # 8)
            w = x_bit_sh + j * t_bit - 2 * t_bit * x_bit_sh
            # 9)
            wc = w.flip(-1).cumsum(-1).flip(-1) - w
            c_beta1 = -j * t_bit + x_bit_sh + j + wc

            # else
            # 11)
            c_igt1 = (1 - j) * (u + 1) - j * u
            c_ie1 = (1 - 2 * j) * u

            l1_mask = j
            c_else = l1_mask * c_ie1 + (1 - l1_mask) * c_igt1

            # Mask for the case r == -1 #2^l - 1
            r_mask = (r == -1).long()

            # Mask combination to execute the
            # if / else statements of 4), 7), 10)
            c = (1 - beta.unsqueeze(-1)) * c_beta0 + \
                beta.unsqueeze(-1) * (1 - r_mask.unsqueeze(-1)) * c_beta1 + \
                beta.unsqueeze(-1) * r_mask.unsqueeze(-1) * c_else

            # 14)
            # Hide c values
            mask = s * c

            # Permute the mask
            idx = [slice(None)] * (len(x_bit_sh.shape) - 1) + [perm]
            mask = mask[idx]

        # send to party C
        ret = self.broadcast_to_arbiter(mask)

        beta_prime = None
        if self.role == 'ARBITER':
            result = sum(ret)
            result = self._cutoff(result, p)
            # print(result)
            # beta_prime = torch.any(result == 0).item()
            beta_prime = (result == 0).sum(-1)
            # print(beta_prime)

        beta_prime = self.broadcast_to_parties_from(beta_prime, self.num_party)

        return beta_prime

    def msb(self, a_sh):
        """
        Compute the most significant bit in a_sh, this is an implementation
        SecureNN paper https://eprint.iacr.org/2018/442.pdf

        Args:
            a_sh (AdditiveSharingTensor): the tensor of study
                for Arbiter: zero with same shape
        Return:
            the most significant bit
                for Arbiter: None
        """

        # field of a_sh is 2**64 - 1
        field_size = self.field
        self.tag += 1
        input_shape = a_sh.shape
        num_party = self.num_party
        a_sh = a_sh.view(-1)

        # the commented out numbers below correspond to the
        # line numbers in Table 5 of the SecureNN paper
        # https://eprint.iacr.org/2018/442.pdf

        # Common Randomness
        # Party A and send to B
        to_send = None
        beta = None  # fake
        if self.world_id == 0:
            beta = torch.zeros_like(a_sh).random_(0, 2)
            u_shs = self._generate_zero_shares(num_party,
                                               a_sh.shape,
                                               field_size - 1)
            to_send = [(beta, u_shs[i]) for i in range(num_party)]

        ret = self.broadcast_slice_to_parties_from(to_send, 0)

        if self.role == 'PARTY':
            beta, u = ret
            # logging.info(f'beta: {beta}')
            # logging.info(f'u: {u}')

        # 1)
        # Arbiter
        to_send = None
        if self.role == 'ARBITER':
            x = torch.zeros_like(a_sh).random_(-(field_size - 1) // 2,
                                               (field_size - 1 - 1) // 2)
            x_sh = self._generate_shares(x, num_party, field_size - 1)
            x_bit = secureNNTH._decompose64(x, field_size - 1)
            _x_bit_0 = x_bit[..., 0]
            x_bit0_sh = self._generate_shares(_x_bit_0, num_party)
            x_bit_sh = self._generate_shares(x_bit, num_party)
            # logging.info(f'arbiter: generated x: {x}')
            # logging.info(f'arbiter: x_bit = {x_bit}')
            to_send = list(zip(x_sh, x_bit_sh, x_bit0_sh))

        ret = self.broadcast_slice_to_parties_from(to_send, num_party)

        r_rec = torch.zeros_like(a_sh)  # fake
        if self.role == 'PARTY':
            x_sh, x_bit_sh, x_bit0_sh = ret

            # logging.info(f'x_sh: {x_sh}')
            # logging.info(f'x_bit_sh: {x_bit_sh}')
            # logging.info(f'x_bit0_sh: {x_bit0_sh}')

            # 2)
            y_sh = a_sh * 2
            r_sh = y_sh + x_sh

            # 3)
            r_rec = self.reconstruct(r_sh)
            r_rec = self._cutoff(r_rec, field_size - 1)
            r_0 = secureNNTH._decompose64(r_rec, field_size - 1)[..., 0]

        else:  # arbiter
            self.tag += 1  # for reconstruct

        # 4)
        beta_prime = self.private_compare_unsigned(x_bit_sh,
                                                   r_rec,
                                                   beta,
                                                   field_size)
        gamma = torch.zeros_like(a_sh)
        delta = torch.zeros_like(a_sh)
        if self.role == 'PARTY':
            # logging.info(f'beta_prime= {beta_prime}')

            # 5)
            beta_prime_sh = beta_prime if self.world_id == 0 else \
                            torch.zeros_like(beta_prime)
            # logging.info(f'beta_prime_sh= {beta_prime_sh}')

            # 7)
            j = 1 if self.world_id == 0 else 0
            gamma = beta_prime_sh + (j * beta) - (2 * beta * beta_prime_sh)
            # logging.info(f'gamma(beta_prime^beta)={gamma}')

            # 8)
            delta = x_bit0_sh + (j * r_0) - (2 * r_0 * x_bit0_sh)
            # logging.info(f'delta(r0^x0)={delta}')

        # 9)
        theta = self.mul(gamma, delta)

        if self.role == 'PARTY':
            # logging.info(f'theta={theta}')
            # 10)
            a = gamma + delta - (theta * 2) + u
            # logging.info(f'a(r0^x0^(x>r))={a}')

            if len(input_shape):
                a = a.view(*list(input_shape))

            return a
        return None

    def share_convert(self, a_sh):
        """Convert share a_sh from field 2**64 to 2**64-1

        Args:
            a_sh (AdditiveSharingTensor): the tensor of study
                for Arbiter: could be None
        Return:
            the converted share
                for Arbiter: None
        """
        field_size = self.field
        self.tag += 1
        # u_sh = self.share_zero_from(a_sh.shape, 0, L - 1)

        if self.role == 'PARTY':
            return self._cutoff(a_sh, field_size - 1)
        return a_sh

    def relu_deriv_p(self, a_sh):
        """simple ge0 algorithm
        """
        # logging.info('enter relu_deriv_p: tag:', self.tag)
        self.tag += 1
        original_shape = a_sh.shape
        print('relu_deriv_p: original_shape:', original_shape)
        aa_sh = a_sh.reshape([-1])  # flatten
        size = aa_sh.shape[0]
        scale = None
        bias = None
        sgn = None
        if self.world_id == 0:
            # scale = torch.randint(-64, 64, [size]).float()
            # scale = 2 ** scale
            scale = [2 ** random.randint(0, 256) for idx in range(size)]
            # scale = [1 for _ in range(size)]
            bias = [2 ** random.randint(0, 256) for idx in range(size)]
            # bias = [random.random() * scale[idx] for idx in range(size)]
            # bias = [8 for _ in range(size)]
            sgn = [1 if random.randint(0, 1) else -1 for idx in range(size)]
            # sgn = [-1 for _ in range(size)]

        ret = self.broadcast_to_parties_from((scale, bias, sgn), 0)
        if self.role == 'PARTY':
            scale, bias, sgn = ret

        if self.role == 'PARTY':
            bias_i = 1 if self.world_id % 2 else -1
            if bias_i == -1 and self.world_id == self.num_party - 1:
                bias_i = 0
            aa_sh = aa_sh.tolist()
            aa_sh = [val * _scale * _sgn + _bias * bias_i
                     for (val, _scale, _bias, _sgn) in zip(aa_sh, scale,
                                                           bias, sgn)]

        ret = self.broadcast_to_arbiter(aa_sh)

        u_shs = torch.zeros_like(a_sh)
        beta_prime = None
        if self.role == 'ARBITER':
            # result = sum(ret)
            # result = [sum(v) for v in zip(*ret)]
            # beta_prime = result >= 0
            beta_prime = [int(sum(v) >= 0) for v in zip(*ret)]
            # logging.info('beta_prime:', beta_prime)
            # beta_prime[beta_prime == 0] = -1
            u_shs = self._generate_zero_shares(self.num_party, aa_sh.shape)
            beta_prime = torch.LongTensor(beta_prime).reshape_as(u_shs[0])
            u_shs[0] += beta_prime
            # print(beta_prime)

        beta_prime = self.broadcast_slice_to_parties_from(u_shs,
                                                          self.num_party)
        if self.role == 'PARTY':
            # ret = beta_prime * scale.sign()  # TODO: possible overflow
            j = 1 if self.world_id == 0 else 0
            for idx, _sgn in enumerate(sgn):
                if _sgn < 0:
                    beta_prime[idx] = j - beta_prime[idx]
            beta_prime = beta_prime.reshape(original_shape)
            if len(beta_prime.shape) == 1:
                beta_prime.unsqueeze_(-1)
            return beta_prime
        return None

    def relu00(self, a_sh):
        """Compute Reul(a)

        Args:
            a_sh (AdditiveSharingTensor): the tensor of study
                for Arbiter: zero with the same shape
        Return:
            the converted share
                for Arbiter: None
        """
        self.tag += 1
        # u_sh = self.share_zero_from(a_sh.shape, 0)

        ge0 = self.relu_deriv_p(a_sh)
        if self.role == 'ARBITER':
            ge0 = torch.zeros_like(a_sh)
        ret = self.mul(a_sh, ge0)
        if self.role == 'PARTY':
            return (ret + a_sh) // 2  # + u_sh
        return None

    def relu_deriv(self, a_sh):
        """Compute a>=0

        Args:
            a_sh (AdditiveSharingTensor): the tensor of study
                for Arbiter: zero with the same shape
        Return:
            the converted share
                for Arbiter: None
        """
        # logging.info('enter relu_deriv: tag:', self.tag)
        self.tag += 1
        # u_sh = self.share_zero_from(a_sh.shape, 0)

        # 1)
        y_sh = a_sh * 2

        # 2)
        y_sh = self.share_convert(y_sh)
        # logging.info('enter relu_deriv.msb: tag:', self.tag)

        # 3)
        alpha_sh = self.msb(y_sh)

        # 4)
        j = 1 if self.world_id == 0 else 0

        if self.role == 'PARTY':
            gamma_sh = j - alpha_sh  # + u_sh
            return gamma_sh
        return None

    def relu(self, a_sh):
        """Compute Reul(a)

        Args:
            a_sh (AdditiveSharingTensor): the tensor of study
                for Arbiter: zero with the same shape
        Return:
            the converted share
                for Arbiter: None
        """
        self.tag += 1
        # u_sh = self.share_zero_from(a_sh.shape, 0)

        ge0 = self.relu_deriv(a_sh)
        if self.role == 'ARBITER':
            ge0 = torch.zeros_like(a_sh)
        print('ge0.shape:', ge0.shape)
        print('a_sh.shape:', a_sh.shape)
        ret = self.mul(a_sh, ge0)
        if self.role == 'PARTY':
            if self.precision > 0:
                ret = ret * (10 ** self.precision)
            return ret  # + u_sh
        return None

    def relu_p(self, a_sh):
        """Compute Reul(a)

        Args:
            a_sh (AdditiveSharingTensor): the tensor of study
                for Arbiter: zero with the same shape
        Return:
            the converted share
                for Arbiter: None
        """
        self.tag += 1
        # u_sh = self.share_zero_from(a_sh.shape, 0)

        ge0 = self.relu_deriv_p(a_sh)
        if self.role == 'ARBITER':
            ge0 = torch.zeros_like(a_sh)
        print('ge0.shape:', ge0.shape)
        print('a_sh.shape:', a_sh.shape)
        ret = self.mul(a_sh, ge0)
        if self.role == 'PARTY':
            if self.precision > 0:
                ret = ret * (10 ** self.precision)
            return ret  # + u_sh
        return None

    def minpool(self, x_sh):
        """Compute min of list of values

        Args:
            x_sh (AdditiveShareingTensor): the tensor of study
                for Arbiter: zero with the same shape

        Return:
            share of min value, index of this value in x_sh
                for Arbiter: None
        """
        # field_size = self.field
        self.tag += 1

        x_sh = x_sh.contiguous().view(-1)

        # u_sh = self.share_zero_from(x_sh.shape, 0)
        # v_sh = self.share_zero_from(x_sh.shape, 0)
        u_sh = self.share_zero_from([1], 0)
        v_sh = self.share_zero_from([1], 0)

        min_sh = x_sh[0]
        ind_sh = self.share_zero_from([1], 0)
        if self.role == 'ARBITER':
            ind_sh = torch.LongTensor(data=[0])

        for idx in range(1, len(x_sh)):
            w_sh = min_sh - x_sh[idx]
            beta_sh = self.relu_deriv(w_sh)
            if self.role == 'ARBITER':
                beta_sh = torch.LongTensor(data=[0])
                min_sh = x_sh[0]
            min_sh = self.select_share(beta_sh, min_sh, x_sh[idx])
            if self.role == 'ARBITER':
                min_sh = torch.LongTensor(data=[0])
            idx_sh = self.share_zero_from([1], 0)
            if self.world_id == 0:
                idx_sh += idx
            if self.role == 'ARBITER':
                idx_sh = torch.LongTensor(data=[0])
                ind_sh = torch.LongTensor(data=[0])

            print(f'beta_sh:{beta_sh}')
            print(f'ind_sh:{ind_sh}')
            print(f'inx_sh:{idx_sh}')
            ind_sh = self.select_share(beta_sh, ind_sh, idx_sh)

        if self.role == 'ARBITER':
            return None

        return min_sh + u_sh, ind_sh + v_sh

    def maxpool(self, x_sh):
        """Compute max of list of values

        Args:
            x_sh (AdditiveShareingTensor): the tensor of study
                for Arbiter: zero with the same shape

        Return:
            share of max value, index of this value in x_sh
                for Arbiter: None
        """
        # field_size = self.field
        self.tag += 1

        x_sh = x_sh.contiguous().view(-1)

        # u_sh = self.share_zero_from(x_sh.shape, 0)
        # v_sh = self.share_zero_from(x_sh.shape, 0)
        u_sh = self.share_zero_from([1], 0)
        v_sh = self.share_zero_from([1], 0)

        max_sh = x_sh[0]
        ind_sh = self.share_zero_from([1], 0)
        if self.role == 'ARBITER':
            ind_sh = torch.LongTensor(data=[0])

        for idx in range(1, len(x_sh)):
            # w_sh = min_sh - x_sh[idx]
            w_sh = x_sh[idx] - max_sh
            beta_sh = self.relu_deriv(w_sh)
            if self.role == 'ARBITER':
                beta_sh = torch.LongTensor(data=[0])
                max_sh = x_sh[0]
            max_sh = self.select_share(beta_sh, max_sh, x_sh[idx])
            if self.role == 'ARBITER':
                max_sh = torch.LongTensor(data=[0])
            idx_sh = self.share_zero_from([1], 0)
            if self.world_id == 0:
                idx_sh += idx
            if self.role == 'ARBITER':
                idx_sh = torch.LongTensor(data=[0])
                ind_sh = torch.LongTensor(data=[0])

            print(f'beta_sh:{beta_sh}')
            print(f'ind_sh:{ind_sh}')
            print(f'inx_sh:{idx_sh}')
            ind_sh = self.select_share(beta_sh, ind_sh, idx_sh)

        if self.role == 'ARBITER':
            return None

        return max_sh + u_sh, ind_sh + v_sh

    def minpool2(self, x_sh, dim=0):
        """Compute min of list of values

        Args:
            x_sh (AdditiveShareingTensor): the tensor of study ####
                for Arbiter: zero with the same shape

        Return:
            share of min [value], [index] of this value in x_sh ####
                for Arbiter: None
        """
        # field_size = self.field
        self.tag += 1

        ret_val = []
        ret_idx = []

        if dim == 0:
            for col in range(x_sh.shape[1]):
                ret = self.minpool(x_sh[:, col])
                if self.role == 'PARTY':
                    ret_val.append(ret[0])
                    ret_idx.append(ret[1])
        if dim == 1:
            for row in range(x_sh.shape[0]):
                ret = self.minpool(x_sh[row, :])
                if self.role == 'PARTY':
                    ret_val.append(ret[0])
                    ret_idx.append(ret[1])

        if self.role == 'ARBITER':
            return None

        return torch.LongTensor(ret_val), torch.LongTensor(ret_idx)
