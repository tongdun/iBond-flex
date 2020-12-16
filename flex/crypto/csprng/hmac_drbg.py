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
import hashlib
import hmac

HMAC_DRBG_MAX_ITERATIONS = 2 ** 48
HMAC_DRBG_MAX_BITS_PER_REQUEST = 2 ** 19


class HMacDRBG(object):
    def __init__(self, entropy: bytes, personalization_string: bytes = b""):
        """
        Implement a cryptographically secure pseudo-random number generator algorithm: HMAC_DRBG (NIST SP 800-90A).
        Parameters are partly referenced from recommendations provided by Appendix D of NIST SP 800-90A.
        Support security strength 256 bits.

        The code references the implementation of python-hmac-drbg library in this URL:
        "https://github.com/fpgaminer/python-hmac-drbg/blob/master/hmac_drbg/hmac_drbg.py".
        Args:
            entropy: bytes， secrect for initializing DRBG.
            personalization_string: bytes, additional input used together with entropy
               for initializing DRBG，default b''.
        """
        self.security_strength = 256
        self.reseed_counter = 1
        self.K = b"\x00" * 32
        self.V = b"\x01" * 32
        self.max_iterations = HMAC_DRBG_MAX_ITERATIONS
        self.__instantiate(entropy, personalization_string)

    def __hmac(self, key: bytes, data: bytes) -> bytes:
        """
        Hmac using sha256 hash.
        """
        return hmac.new(key, data, hashlib.sha256).digest()

    def __update(self, provided_data: bytes = b""):
        """
        Update internal states.
        """
        self.K = self.__hmac(self.K, self.V + b"\x00" + provided_data)
        self.V = self.__hmac(self.K, self.V)

        if len(provided_data) != 0:
            self.K = self.__hmac(self.K, self.V + b"\x01" + provided_data)
            self.V = self.__hmac(self.K, self.V)

    def __instantiate(self, entropy: bytes, personalization_string: bytes):
        """
        Set hmac_drbg seed for onetime key generation.
        """
        if len(personalization_string) * 8 > 256:
            raise RuntimeError("personalization_string cannot exceed 256 bits.")

        if (len(entropy) * 8 * 2) < (3 * self.security_strength):
            raise RuntimeError("entropy must be at least %f bits." % (1.5 * self.security_strength))

        if len(entropy) * 8 > 1000:
            raise RuntimeError("entropy cannot exceed 1000 bits.")

        seed_material = entropy + personalization_string

        self.K = b"\x00" * 32
        self.V = b"\x01" * 32

        self.__update(seed_material)
        self.reseed_counter = 1

    def reseed(self, entropy: bytes):
        """
        If max iteration(2**48) is reached, reseed function must be called to re-initialize the onetime key seed.
        One rarely need to do this in practice.
        """
        if (len(entropy) * 8 * 2) < (3 * self.security_strength):
            raise RuntimeError("entropy must be at least %f bits." % (1.5 * self.security_strength))

        if len(entropy) * 8 > 1000:
            raise RuntimeError("entropy cannot exceed 1000 bits.")

        self.__update(entropy)
        self.reseed_counter = 1

    def generate(self, num_bytes: int) -> bytes:
        """
        Generate presudo random numbers counted by num_bytes.
        """
        if (num_bytes * 8) > HMAC_DRBG_MAX_BITS_PER_REQUEST:
            raise RuntimeError("generate cannot generate more than 2**19 bits in a single call.")

        if self.reseed_counter > self.max_iterations:
            return None

        res = b""

        while len(res) < num_bytes:
            self.V = self.__hmac(self.K, self.V)
            res += self.V

        self.__update()
        self.reseed_counter += 1

        return res[:num_bytes]
