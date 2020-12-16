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

from .. import gmpy_math


class PaillierPublicKey(object):
    """Contains a public key and associated encryption methods.
    """
    __slots__ = ['g', 'n', 'nsquare', 'max_int']

    def __init__(self, n):
        self.g = n + 1
        self.n = n
        self.nsquare = gmpy_math.mul(n, n)
        self.max_int = n // 3 - 1

    def __repr__(self):
        hashcode = hex(hash(self))[2:]
        return "<PaillierPublicKey {}>".format(hashcode[:10])

    def __eq__(self, other):
        return self.n == other.n

    def __hash__(self):
        return hash(self.n)


class PaillierPrivateKey(object):
    """Contains a private key and associated decryption method.
    """
    __slots__ = ['public_key', 'p', 'q', 'psquare',
                 'qsquare', 'q_inverse', 'hp', 'hq']

    def __init__(self, public_key, p, q):
        # if not p * q == public_key.n:
        if not gmpy_math.mul(p, q) == public_key.n:
            raise ValueError(
                "given public key does not match the given p and q")
        if p == q:
            raise ValueError("p and q have to be different")

        self.public_key = public_key
        if q < p:
            self.p = q
            self.q = p
        else:
            self.p = p
            self.q = q

        self.psquare = gmpy_math.mul(self.p, self.p)
        self.qsquare = gmpy_math.mul(self.q, self.q)
        self.q_inverse = gmpy_math.invert(self.q, self.p)
        self.hp = self.__h_func(self.p, self.psquare)
        self.hq = self.__h_func(self.q, self.qsquare)

    def __eq__(self, other):
        return self.p == other.p and self.q == other.q

    def __hash__(self):
        return hash((self.p, self.q))

    def __repr__(self):
        hashcode = hex(hash(self))[2:]

        return "<PaillierPrivateKey {}>".format(hashcode[:10])

    def __h_func(self, x, xsquare):
        """Computes the h-function as defined in Paillier's paper page.
        """
        return gmpy_math.invert(self.__l_func(gmpy_math.powmod(self.public_key.g,
                                                               x - 1, xsquare), x), x)

    def __l_func(self, x, p):
        """computes the L function as defined in Paillier's paper.
        """
        return (x - 1) // p


def generate_paillier_keypair(n_length: int = 1024, seed: int = None) -> (PaillierPublicKey, PaillierPrivateKey):
    """Return a new :class:`PaillierPublicKey` and :class:`PaillierPrivateKey`.

    Args:
      n_length: int, key size in bits.
      seed: int, None for default.

    Returns:
      tuple: The generated :class:`PaillierPublicKey` and
      :class:`PaillierPrivateKey`
    """
    p = q = n = None
    n_len = 0
    i = 1
    while n_len != n_length:
        if seed:
            p = gmpy_math.getprimeover(n_length // 2, seed)
        else:
            p = gmpy_math.getprimeover(n_length // 2)

        q = p
        while q == p:
            if seed:
                q = gmpy_math.getprimeover(n_length // 2, seed + i)
                i += 1
            else:
                q = gmpy_math.getprimeover(n_length // 2)

        n = gmpy_math.mul(p, q)
        n_len = n.bit_length()

    public_key = PaillierPublicKey(n)
    private_key = PaillierPrivateKey(public_key, p, q)

    return public_key, private_key
