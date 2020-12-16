import numpy as np

from flex.crypto.fpe.api import generate_fpe_encryptor
from flex.constants import *

key = b'1234567890123456'
n = 15
t = b''
encryptor = generate_fpe_encryptor(key, n, t, method=CRYPTO_FF1, encrypt_algo=CRYPTO_AES)

x = 32767
y = encryptor.encrypt(x)
z = encryptor.decrypt(y)
print(x, z)

x = np.array([i for i in range(2**15)], dtype=np.uint32)
y = encryptor.encrypt(x)
z = encryptor.decrypt(y)
print(z)

