from typing import List, Tuple, Union
from logging import getLogger

import numpy as np
import pandas as pd
from scipy import sparse


logger = getLogger("Common_tools_")

def onehot_encoding(data: np.ndarray, 
    uniq_vals: List, col_name: str, to_sparse=False) -> Tuple:

    new_names = list(map(lambda x: col_name+"_"+str(x), uniq_vals))
    d = dict(zip(uniq_vals, list(range(len(uniq_vals)))))
    data = list(map(lambda x: d[x], data))
    data = np.eye(len(uniq_vals))[data]

    if to_sparse:
        data = sparse.coo_matrix(data)

    return data, new_names


def array_hstack(arrays: List[Union[np.ndarray, sparse.coo_matrix]]):
    dtype = list(map(lambda x: isinstance(x, sparse.coo_matrix), arrays))
    if np.sum(dtype) == len(arrays):
        return sparse.hstack(arrays)
    
    dtype = [isinstance(arr, np.ndarray) for arr in arrays]
    if np.sum(dtype) == len(arrays):
        return np.concatenate(arrays, axis=1)
