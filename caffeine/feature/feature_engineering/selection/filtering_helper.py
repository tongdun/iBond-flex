from typing import List, Union
from logging import getLogger

import numpy as np

from caffeine.utils import ClassMethodAutoLog

logger = getLogger("Filtering-Helper")


@ClassMethodAutoLog()
def update_matrix(data_values: np.ndarray, idx: Union[np.ndarray, List]) -> np.ndarray:
    """
    Get new matrix by idx.

    Args:
        data_values: np.ndarray, original data.
        idx: np.ndarray or list, indices of columns.

    Return:
        np.ndarray.
    """
    data_values = data_values[:, idx]
    return data_values

@ClassMethodAutoLog()
def federation_guest_idx(guest_ivs, rm_id, cum_dims, dims, down_feature_num) -> np.ndarray:
    """
    Guest select features.

    Args:
        guest_ivs: np.ndarray, iv values for filtering.
        rm_id: np.ndarray, index to be removed.
        cum_dims: np.ndarray, cumsum dimensions.
        dims: List[int], feature dimensions of each each party.

    Return:
        list, guest remaining indices.
    """
    if dims[-1] != 0:
        guest_rm_id  = rm_id[np.where(rm_id >= cum_dims[-2])[0]] - cum_dims[-2]
        guest_idx = np.delete(np.arange(len(guest_ivs)), guest_rm_id)
        guest_idx = down_num_feature(guest_idx, guest_ivs, guest_rm_id, down_feature_num)
    else:
        guest_idx = np.array([0])
    logger.info(f'***** after federation guest_idx {guest_idx}')
    return guest_idx

@ClassMethodAutoLog()
def federation_host_idx(host_ivs, rm_id, cum_dims, down_feature_num):
    """
    Host select features.

    Args:
        host_ivs: list of np.ndarray, iv values for filtering.
        rm_id: np.ndarray, index to be removed.
        cum_dims: np.ndarray, cumsum dimensions.

    Return:
        list, guest remaining indices.
    """
    host_rm_id = get_host_rm_id(rm_id, cum_dims)
    host_idx = []
    for i in range(len(host_rm_id)):
        host_idx.append(np.delete(np.arange(len(host_ivs[i])), host_rm_id[i]))

    # down filter features
    for i in range(len(host_idx)):
        host_idx[i] = down_num_feature(host_idx[i], host_ivs[i], host_rm_id[i], down_feature_num)
    logger.info(f'***** after federation host_idx {host_idx}')
    return host_idx

@ClassMethodAutoLog()
def get_host_rm_id(rm_id: np.ndarray, cum_dims: np.ndarray) -> List[np.ndarray]:
    """
    Calculate removed index for host.

    Args:
        rm_id: np.ndarray, index to be removed.
        cum_dims: np.ndarray, cumsum dimensions.

    Return:
        list of np.ndarray, index to be selected for all hosts.
    """
    host_rm_id = []
    for i in range(len(cum_dims)-1):
        if i == 0:
            host_rm_id.append(rm_id[np.where(rm_id < cum_dims[0])[0]])
        else:
            host_rm_id.append(rm_id[np.where((cum_dims[i-1] <= rm_id) & (rm_id < cum_dims[i]))[0]] - cum_dims[i-1])
    # logger.info(f'***** after remove host_idx {rm_id} {host_rm_id}')
    return host_rm_id

@ClassMethodAutoLog()
def get_rm_index(matrix: np.ndarray, thres: float, key_vals: np.ndarray) -> np.ndarray:
    """
    Get removed indices if any value of matrix is bigger than thres given key_vals.

    Args:
        matrix: np.ndarray, coefficient matrix.
        thres: float, thres to compare.
        key_vals: np.ndarray, key values to filter.

    Return:
        np.ndarray, indices to be removed.
    """
    pairs = np.argwhere(abs(matrix) > thres)
    rm_id = np.zeros(0)
    for i in range(pairs.shape[0]):
        s0, s1 = pairs[i, :]
        if s0 > s1:
            if key_vals[s0] > key_vals[s1]:
                if s0 not in rm_id:
                    rm_id = np.append(rm_id, s1)
            else:
                if s1 not in rm_id:
                    rm_id = np.append(rm_id, s0)
    rm_id = np.unique(sorted(rm_id)).astype(int)
    logger.info(f'***** after remove idx {rm_id} {pairs}')
    return rm_id

@ClassMethodAutoLog()
def down_num_feature(index: np.ndarray, 
                    ivs: np.ndarray, 
                    rm_id: np.ndarray,
                    down_feature_num: int) -> List:
    """
    To cover the down_feature_num.

    Args:
        index: np.ndarray, remaining indices.
        ivs: np.ndarray, input iv values for all features.
        rm_id: np.ndarray, to be removed indices.

    Return:
        list, remaining indices after covering the down_feature_num.
    """
    index = index.tolist()
    logger.info(f'>>>> before index {index} {rm_id}')
    if len(index) < down_feature_num:
        tmp = ivs[rm_id].argsort()[::-1]
        index = index + rm_id[tmp][: (down_feature_num - len(index))].tolist()
    logger.info(f'>>>> after index {index} {rm_id}')
    return np.array(index).astype(int)