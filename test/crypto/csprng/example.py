from flex.crypto.csprng.api import generate_csprng_generator
from flex.constants import *

if __name__ == '__main__':
    drbg = generate_csprng_generator(b'542435464554342576476747656736767657676545234546', b'', method=CRYPTO_HMAC_DRBG)
    onetime_key = drbg.generate(2**16)
    print(onetime_key)

    drbg.reseed(b'e4243546455434c576476747656736767657676d4523454a')
    print(drbg.generate(2**7))
