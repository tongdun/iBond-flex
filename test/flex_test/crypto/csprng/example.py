from flex.crypto.csprng.api import generate_csprng_generator
from flex.constants import *

if __name__ == '__main__':
    # 生成伪随机数生成器实例
    drbg = generate_csprng_generator(entropy=b'542435464554342576476747656736767657676545234546',
                                     personalization_string=b'', method=CRYPTO_HMAC_DRBG)

    # 调用generate方法生成伪随机数串，返回长度为num_bytes的字节串
    onetime_key = drbg.generate(num_bytes=2 ** 16)
    print(onetime_key)

    # 通过entropy重置伪随机数生成器，当生成伪随机数次数超过2**48次后推荐重置生成器
    drbg.reseed(b'e4243546455434c576476747656736767657676d4523454a')
    # 调用generate方法生成新的伪随机数串
    print(drbg.generate(num_bytes=2 ** 7))
