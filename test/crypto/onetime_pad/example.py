import numpy as np
import torch
from flex.crypto.onetime_pad.api import generate_onetime_pad_encryptor

if __name__ == '__main__':
    secret_key = b'1' * 48
    encryptor = generate_onetime_pad_encryptor(secret_key)

    x = [np.array([0.54356], dtype=np.float32),
         [torch.tensor([0.54356, 0.45345], dtype=torch.float32), np.array([[455432], [54352]], dtype=np.int32)]]
    en_x = encryptor.encrypt(x, alpha=2)
    z = encryptor.decrypt(en_x, alpha=2)
    print(x)
    print(z)

    encryptor2 = generate_onetime_pad_encryptor(secret_key)
    y = [np.array([0.4325], dtype=np.float32),
         [torch.tensor([0.5434, 0.5631], dtype=torch.float32), np.array([[65643452], [542794]], dtype=np.int32)]]
    en_y = encryptor2.encrypt(y, alpha=-2)
    z = (en_x + en_y).decode()
    print(z)

    encryptor3 = generate_onetime_pad_encryptor(secret_key)
    en_y = encryptor3.encrypt(y, alpha=-1)
    z = encryptor.decrypt(en_x + en_y, 1)
    print(z)

