import numpy as np

from flex.crypto.id_filter.api import generate_id_filter

if __name__ == '__main__':
    id_filter1 = generate_id_filter(31)
    id_filter1.update([3, 7, 11, 17])

    id_filter2 = generate_id_filter(31, np.zeros((1 << 31), dtype=np.bool))
    id_filter2.update([1, 3, 17, 19])

    id_filter3 = id_filter1 == id_filter2
    id_filter4 = id_filter1 & id_filter2

    secret_key = b'2b7e151628aed2a6abf7158809cf4f3c'
    id_filter2 = id_filter1.permute(secret_key)
    id_filter3 = id_filter2.inv_permute(secret_key)
