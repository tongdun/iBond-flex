import numpy as np
from flex.crypto.paillier.api import generate_paillier_encryptor_decryptor

if __name__ == '__main__':

    pe, pd = generate_paillier_encryptor_decryptor()

    plain_param1 = np.random.random(100).astype(np.float32)
    encrypted_param1 = pe.encrypt(plain_param1)
    plain_param2 = pd.decrypt(encrypted_param1)
    print(np.all(plain_param1 - plain_param2 < 1e-3))

    # 将加密器进行序列化并传输
    import pickle
    s = pickle.dumps(pe)
    pe = pickle.loads(s)

    # 支持numpy.array的直接加密和解密（并自动多进程并发处理）
    plain_param1 = np.random.random((128, 4)).astype(np.float32)
    encrypted_param1 = pe.encrypt(plain_param1)
    plain_param2 = pd.decrypt(encrypted_param1)

    ## 密文计算的并行加速
    from flex.crypto.paillier import parallel_ops
    pe, pd = generate_paillier_encryptor_decryptor()

    # x = np.random.randint(100, size=1000)
    x = np.random.random(size=(1000, 2))
    y = np.random.randint(1000, size=(1000, 2))
    en_x = pe.encrypt(x)
    en_y = pe.encrypt(y)

    en_result = parallel_ops.add(en_x, y)
    result = pd.decrypt(en_result)
    print(np.all(x + y - result < 1e-3))

    en_result = parallel_ops.add(en_x, en_y)
    result = pd.decrypt(en_result)
    print(np.all(x + y - result < 1e-3))

    en_result = parallel_ops.mul(en_x, y)
    result = pd.decrypt(en_result)
    print(np.all(x * y - result < 1e-3))
