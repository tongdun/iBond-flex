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

from .. import gmpy_math
from .fixedpoint_number import FixedPointNumber
from .obfuscator import apply_obfuscation
from .raw_encrypt import raw_encrypt
from .keypair import PaillierPublicKey


class PaillierEncryptedNumber(object):
    """
    Represents the Paillier encryption of a float or int.

    Supports encrypted add and mul. 
    """

    __slots__ = ('public_key', 'exponent', '__ciphertext', '__is_obfuscator')

    def __init__(self, public_key: PaillierPublicKey, ciphertext: int, exponent: int = 0):
        """
        Create a new instance of Paillier encrypt Number.
        Arg:
            public_key: PaillierPublicKey
            ciphertext: int
            exponent: int
        Return:
            instance of PaillierEncryptedNumber
        """
        self.public_key = public_key
        self.__ciphertext = ciphertext
        self.exponent = exponent
        self.__is_obfuscator = False

    def ciphertext(self, be_secure: bool = True) -> int:
        """return the ciphertext of the PaillierEncryptedNumber.
        """
        if be_secure and not self.__is_obfuscator:
            self.apply_obfuscation()

        return self.__ciphertext

    def apply_obfuscation(self) -> None:
        """ciphertext by multiplying by r ** n with random r
        """
        self.__ciphertext = apply_obfuscation(
            self.__ciphertext, self.public_key)
        self.__is_obfuscator = True

    def __add__(self, other: '__class__') -> '__class__':
        if isinstance(other, __class__):
            return self.__add_encryptednumber(other)
        else:
            return self.__add_scalar(other)

    def __radd__(self, other: '__class__') -> '__class__':
        return self.__add__(other)

    def __sub__(self, other: '__class__') -> '__class__':
        return self + (other * -1)

    def __rsub__(self, other: '__class__') -> '__class__':
        return other + (self * -1)

    def __rmul__(self, scalar: Union[int, float]) -> '__class__':
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Union[int, float]) -> '__class__':
        return self.__mul__(1 / scalar)

    def __mul__(self, scalar: Union[int, float]) -> '__class__':
        """return Multiply by an scalar(such as int, float)
        """
        if isinstance(scalar, PaillierEncryptedNumber):
            raise ValueError(
                f"PaillierEncryptedNumber * PaillierEncryptedNumber is not allowed.")

        encode = FixedPointNumber.encode(
            scalar, self.public_key.n, self.public_key.max_int)
        plaintext = encode.encoding

        if plaintext < 0 or plaintext >= self.public_key.n:
            raise ValueError("Scalar out of bounds: %i" % plaintext)

        if plaintext >= self.public_key.n - self.public_key.max_int:
            # Very large plaintext, play a sneaky trick using inverses
            neg_c = gmpy_math.invert(self.ciphertext(
                False), self.public_key.nsquare)
            neg_scalar = self.public_key.n - plaintext
            ciphertext = gmpy_math.powmod(
                neg_c, neg_scalar, self.public_key.nsquare)
        else:
            ciphertext = gmpy_math.powmod(self.ciphertext(
                False), plaintext, self.public_key.nsquare)

        exponent = self.exponent + encode.exponent

        return PaillierEncryptedNumber(self.public_key, ciphertext, exponent)

    def __increase_exponent_to(self, new_exponent: int) -> '__class__':
        """return PaillierEncryptedNumber: 
           new PaillierEncryptedNumber with same value but having great exponent.
        """
        if new_exponent < self.exponent:
            raise ValueError("New exponent %i should be great than old exponent %i" % (
                new_exponent, self.exponent))

        factor = pow(FixedPointNumber.BASE, new_exponent - self.exponent)
        new_encryptednumber = self.__mul__(factor)
        new_encryptednumber.exponent = new_exponent

        return new_encryptednumber

    def __align_exponent(self, x: '__class__', y: '__class__') -> ('__class__', '__class__'):
        """return x,y with same exponet
        """
        if x.exponent < y.exponent:
            x = x.__increase_exponent_to(y.exponent)
        elif x.exponent > y.exponent:
            y = y.__increase_exponent_to(x.exponent)

        return x, y

    def __add_scalar(self, scalar: Union[int, float]) -> '__class__':
        """return PaillierEncryptedNumber: z = E(x) + y
        """
        encoded = FixedPointNumber.encode(scalar,
                                          self.public_key.n,
                                          self.public_key.max_int,
                                          max_exponent=self.exponent)

        return self.__add_fixpointnumber(encoded)

    def __add_fixpointnumber(self, encoded: FixedPointNumber) -> '__class__':
        """return PaillierEncryptedNumber: z = E(x) + FixedPointNumber(y)
        """
        if self.public_key.n != encoded.n:
            raise ValueError(
                "Attempted to add numbers encoded against different public keys!")

        # their exponents must match, and align.
        x, y = self.__align_exponent(self, encoded)

        encrypted_scalar = raw_encrypt(
            y.encoding, x.public_key, 1)  # x.public_key.raw_encrypt(, 1)
        encryptednumber = self.__raw_add(
            x.ciphertext(False), encrypted_scalar, x.exponent)

        return encryptednumber

    def __add_encryptednumber(self, other: '__class__') -> '__class__':
        """return PaillierEncryptedNumber: z = E(x) + E(y)
        """
        if self.public_key != other.public_key:
            raise ValueError("add two numbers have different public key!")

        # their exponents must match, and align.
        x, y = self.__align_exponent(self, other)

        encryptednumber = self.__raw_add(x.ciphertext(
            False), y.ciphertext(False), x.exponent)

        return encryptednumber

    def __raw_add(self, e_x: '__class__', e_y: '__class__', exponent: int) -> '__class__':
        """return the integer E(x + y) given ints E(x) and E(y).
        """
        ciphertext = gmpy_math.mulmod(e_x, e_y, self.public_key.nsquare)

        return PaillierEncryptedNumber(self.public_key, ciphertext, exponent)
