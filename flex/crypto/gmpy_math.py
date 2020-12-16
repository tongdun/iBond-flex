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

"""
Modified based on https://github.com/FederatedAI/FATE.
"""

import random
import gmpy2

POWMOD_GMP_SIZE = pow(2, 64)


def mul(a, b):
    return int(gmpy2.mul(a, b))


def crt(mp, mq, p, q, q_inverse, n):
    """the Chinese Remainder Theorem as needed for decryption.
       return the solution modulo n=pq.
   """
    # u = (mp - mq) * self.q_inverse % self.p
    # x = (mq + (u * self.q)) % self.public_key.n

    u = gmpy2.mul(mp-mq, q_inverse) % p
    x = (mq + gmpy2.mul(u, q)) % n
    return int(x)


def mulmod(a, b, c):
    '''

    reutn int: (a * b) % c
    '''
    return int(gmpy2.mul(a, b) % c)


def powmod(a: int, b: int, c: int) -> int:
    """
    return int: (a ** b) % c
    """

    if a == 1:
        return 1

    if max(a, b, c) < POWMOD_GMP_SIZE:
        return pow(a, b, c)

    else:
        return int(gmpy2.powmod(a, b, c))


def invert(a, b):
    """return int: x, where a * x == 1 mod b
    """
    x = int(gmpy2.invert(a, b))

    if x == 0:
        raise ZeroDivisionError('invert(a, b) no inverse exists')

    return x


def getprimeover(n, seed=None):
    """return a random n-bit prime number
    """
    if not seed:
        r = gmpy2.mpz(random.SystemRandom().getrandbits(n))
    else:
        random.seed(seed)
        r = random.getrandbits(n)
    r = gmpy2.bit_set(r, n - 1)

    return int(gmpy2.next_prime(r))


def isqrt(n):
    """ return the integer square root of N """

    return int(gmpy2.isqrt(n))
