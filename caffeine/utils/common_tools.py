#!/usr/bin/python3
#
#  _____                     _               _______                 _   _____        __  __     _
# |_   _|                   | |             (_) ___ \               | | /  __ \      / _|/ _|   (_)
#   | | ___  _ __   __ _  __| |_   _ _ __    _| |_/ / ___  _ __   __| | | /  \/ __ _| |_| |_ ___ _ _ __   ___
#   | |/ _ \| '_ \ / _` |/ _` | | | | '_ \  | | ___ \/ _ \| '_ \ / _` | | |    / _` |  _|  _/ _ \ | '_ \ / _ \
#   | | (_) | | | | (_| | (_| | |_| | | | | | | |_/ / (_) | | | | (_| | | \__/\ (_| | | | ||  __/ | | | |  __/
#   \_/\___/|_| |_|\__, |\__,_|\__,_|_| |_| |_\____/ \___/|_| |_|\__,_|  \____/\__,_|_| |_| \___|_|_| |_|\___|
#                   __/ |
#                  |___/
#
#  Copyright 2020 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#
#  Project name: iBond
#
#  File name: common_tools.py
#
#  Create date: 2020/12/21
#
import numpy as np
import uuid
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, List, Union, Tuple

from .exceptions import ShapeMismatchError


def is_binary(labels: np.ndarray):
    '''
    Check if labels are binary.
    '''
    classes = np.unique(labels)
    return len(classes) == 2


def get_class_weight(labels: np.ndarray, class_weight: Union[Dict, str, None] = 'balanced') -> dict:
    '''
    Get class weights for labels in training dataset
    Args:
        labels: np.ndarray, shape (n_samples,). Array of original class labels per sample [y].
        class_weight: dict, 'balanced' or None
                    If 'balanced', class weights will be given by
                    ``n_samples / (n_classes * np.bincount(y))``.
                    If a dictionary is given, keys are classes and values
                    are corresponding class weights.
                    If None is given, the class weights will be uniform.

    Returns:
        dict: A dictionary of weights for corresponding labels. -> {class_name: class_weight}

    -----

    **Examples:**

    >>> from caffeine.utils.common_tools import get_class_weight
    >>> labels = np.array([1,1,2,2,3,3,3,3])
    >>> class_weights = get_class_weight(labels)
    >>> print(class_weights)
    {
        1: 1.33333333,  # 2*1.33333333 = 2.666666
        2: 1.33333333,  # 2*1.33333333 = 2.666666
        3: 0.66666667   # 4*0.66666667 = 2.666668
    }
    '''
    labels = mcl_repr(labels, squeeze=True)
    # single column
    classes = np.unique(labels)
    # classes is a sorted unique elements of labels
    weights = compute_class_weight(class_weight, classes, labels)
    class_weights = {
        class_name: weights[idx] for idx, class_name in enumerate(classes)
    }
    return class_weights

# pylint: disable-unsubscriptable-object
def bcl_repr(binary: Union[List, np.ndarray]) -> np.ndarray:
    """
    Format binary classification prediction or label vectors.

    Args:
        binary: list or np.ndarray, is labels or predictions in binary classification.
                Supported input shapes:
                    1. (n_sample), The input will be convert to np.ndarray.
                    2, (1, n_sample), (n_sample, 1), These kind of input will be flatten to (n_sample).
                    3. (n_sample, 2), The first elements of the second dimension will be dropped. [:, 1]
                    4. (2, n_sample), The second elements of the first dimension will be dropped. [0, :]

    Return:
        np.ndarray: flattened predictions/labels of the postive.

    -----

    **Examples:**

    >>> from caffeine.utils.common_tools import bcl_repr
    >>> labels = np.array([[1,0,1,1]])
    >>> labels = bcl_repr(labels)
    >>> print(labels)
    array([1, 0, 1, 1])

    """
    binary = np.array(binary)
    shape: Tuple = binary.shape

    if len(shape) == 1:
        return binary
    elif len(shape) == 2:
        if shape[0] == 1 or shape[1] == 1:
            return binary.flatten()
        elif shape[1] == 2:
            return binary[:, 1].flatten()
        elif shape[0] == 2:
            return binary[0, :].flatten()

    raise ShapeMismatchError(
            f'Unknown shape {shape} for binary predictions or labels.')


def mcl_repr(multiple: Union[List, np.ndarray], squeeze: bool, classes: List = []) -> np.ndarray:
    """
    Format multiclass classification labels.

    Args:
        multiple: Union[List, np.ndarray], representation of the mult-class labels.
            Supported input shapes:
                1. (n_sample), The input will be convert to np.ndarray.
                2, (1, n_sample), (n_sample, 1), These kind of input will be flatten to (n_sample).
                3. (n_sample, m), one hot.
        squeeze: bool, if squeeze, return 1-D.
        classes: int, [] to find.

    Returns:
        np.ndarray: if squeeze, flattened 1-D ints, else m-D.
    """
    multiple = np.array(multiple)
    shape: Tuple = multiple.shape

    if len(shape) == 1 or (len(shape) == 2 and (shape[0] == 1 or shape[1] == 1)):
        # 1, 2
        multiple = multiple.flatten()
        if squeeze:
            return multiple
        else:
            if len(classes) <= 0:
                classes, inv = np.unique(multiple, return_inverse=True)
                return np.eye(len(classes))[inv]
            else:
                num_classes = len(classes)
                num_samples = len(multiple)
                mapping = {}
                for i,c in enumerate(classes):
                    mapping[c] = np.zeros(num_classes)
                    mapping[c][i] = 1
                return np.array(
                    [mapping[c] for c in multiple.flatten()]
                )
    elif len(shape) == 2:
        if squeeze:
            # NOTE different to binary case
            return np.argmax(multiple, axis = 1)
        else:
            return multiple

    raise ShapeMismatchError(
            f'Unknown shape {shape} for binary predictions or labels.')


def gen_module_id(prefix=''):
    return f'{prefix}-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}-{uuid.uuid1().hex}'
