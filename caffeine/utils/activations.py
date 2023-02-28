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
#  File name:                                                                           
#                                                                                              
#  Create date:  2020/12/23                                                                    

import numpy as np
from typing import Union


def sigmoid_single(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


def sigmoid(input: np.ndarray) -> np.ndarray:
    '''
    This is a sigmoid function.

    Args:
        input: shape (n,m) numpy array. Input data to calc sigmoid result.

    Returns:
        sigmoid: shape (n,m) numpy array, the returned values are always floats. The sigmoid result.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import sigmoid
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> sigmoid(input)

    >>> #Output:
    >>> array([0.73105858, 0.88079708, 0.95257413, 0.98201379, 0.99330715, \
        0.99752738, 0.99908895, 0.99966465, 0.99966465, 0.99908895, \
        0.99752738, 0.99330715, 0.98201379, 0.95257413, 0.88079708, \
        0.73105858])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    out = 0.5*np.ones(x.shape)

    # non negative x
    non_negative_indices = (x >= 0)
    z = np.exp(-x[non_negative_indices])
    out[non_negative_indices] = 1.0 / (1.0 + z)

    # negative x
    negative_indices = (x < 0)
    z = np.exp(x[negative_indices])
    out[negative_indices] = z / (1.0 + z)

    return out


def sigmoid_gradient(input: np.ndarray) -> np.ndarray:
    '''
    This is a function for calc gradient of sigmoid.

    Args:
        input: shape (n,m) numpy array. Input data to calc gradient of sigmoid.

    Returns:
        gradient: shape (n,m) numpy array. The gradient of sigmoid.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import sigmoid_gradient
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> sigmoid_gradient(input)

    >>> #Output:
    >>> array([0.19661193, 0.10499359, 0.04517666, 0.01766271, 0.00664806, \
        0.00246651, 0.00091022, 0.00033524, 0.00033524, 0.00091022, \
        0.00246651, 0.00664806, 0.01766271, 0.04517666, 0.10499359, \
        0.19661193])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    return sigmoid(input)*(1. - sigmoid(input))


def hard_sigmoid(input: np.ndarray, threshold: float = 2.5, inplace: bool = False) -> np.ndarray:
    '''
    This is a hard sigmoid function.

    Args:
        input: shape (n,m) numpy array. Input data to calc hard sigmoid result.
        threshold: float, default 2.5. Threshold for build cutting boundery.
        inplace: bool, default False. Inplace the original data or not.

    Returns:
        hard sigmoid: shape (n,m) numpy array. The hard sigmoid result.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import hard_sigmoid
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> hard_sigmoid(input)

    >>> #Output:
    >>> array([0.7, 0.9, 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , \
        1. , 0.9, 0.7])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    x[x < (-1.0 * threshold)] = 0.0
    x[((-1.0 * threshold) <= x) & (x <= threshold)] = (1.0 / (2.0 * threshold)) \
                                                      * x[((-1.0 * threshold) <= x) & (x <= threshold)] + 0.5
    x[x > threshold] = 1.0
    if inplace:
        input = x
    return x


def hard_sigmoid_gradient(input: np.ndarray, threshold: float = 2.5, inplace: bool = False) -> np.ndarray:
    '''
    This is a function for calc gradient of hard sigmoid.

    Args:
        input: shape (n,m) numpy array. Input data to calc gradient of hard sigmoid.
        threshold: float, default 2.5. Threshold for build cutting boundery.
        inplace: bool, default False. Inplace the original data or not.

    Returns:
        gradient: shape (n,m) numpy array. The gradient of hard sigmoid.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import hard_sigmoid_gradient
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> hard_sigmoid_gradient(input)

    >>> #Output:
    >>> array([0.2, 0.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , \
        0. , 0.2, 0.2])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    x[x < (-1.0 * threshold)] = 0.0
    x[((-1.0 * threshold) <= x) & (x <= threshold)] = (1.0 / (2.0 * threshold))
    x[x > threshold] = 0.0
    if inplace:
        input = x
    return x


def logsigmoid(input: np.ndarray) -> np.ndarray:
    '''
    This is a logsigmoid function.

    Args:
        input: shape (n,m) numpy array. Input data to calc logsigmoid result.

    Returns:
        logsigmoid: shape (n,m) numpy array, the returned values are always floats. The logsigmoid result.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import logsigmoid
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> logsigmoid(input)

    >>> #Output:
    >>> array([0.31326169, 0.12692801, 0.04858735, 0.01814993, 0.00671535, \
        0.00247569, 0.00091147, 0.00033541, 0.00033541, 0.00091147, \
        0.00247569, 0.00671535, 0.01814993, 0.04858735, 0.12692801, \
        0.31326169])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    return np.log(sigmoid(input))


def logsigmoid_gradient(input: np.ndarray) -> np.ndarray:
    '''
    This is a function for calc gradient of logsigmoid.

    Args:
        input: shape (n,m) numpy array. Input data to calc gradient of logsigmoid.

    Returns:
        gradient: shape (n,m) numpy array. The gradient of logigmoid.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import logsigmoid_gradient
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> logsigmoid_gradient(input)

    >>> #Output:
    >>> array([0.26894142, 0.11920292, 0.04742587, 0.01798621, 0.00669285, \
        0.00247262, 0.00091105, 0.00033535, 0.00033535, 0.00091105, \
        0.00247262, 0.00669285, 0.01798621, 0.04742587, 0.11920292, \
        0.26894142])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    return 1. - sigmoid(input)


def linear(input: np.ndarray) -> np.ndarray:
    '''
    This is a linear function.

    Args:
        input: shape (n,m) numpy array. Input data to calc linear result.

    Returns:
        linear: shape (n,m) numpy array, the returned values are always floats. The linear result.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import linear
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> linear(input)

    >>> #Output:
    >>> array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    return x


def linear_gradient(input: np.ndarray) -> np.ndarray:
    '''
    This is a function for calc gradient of linear.

    Args:
        input: shape (n,m) numpy array. Input data to calc gradient of linear.

    Returns:
        gradient: shape (n,m) numpy array. The gradient of linear.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import linear_gradient
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> linear_gradient(input)

    >>> #Output:
    >>> array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    return np.full_like(x, 1.0, dtype=x.dtype)


def tanh(input: np.ndarray) -> np.ndarray:
    '''
    This is a tanh function.

    Args:
        input: shape (n,m) numpy array. Input data to calc tanh result.

    Returns:
        tanh: shape (n,m) numpy array, the returned values are always floats. The tanh result.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import tanh
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> tanh(input)

    >>> #Output:
    >>> array([0.76159416, 0.96402758, 0.99505475, 0.9993293 , 0.9999092 , \
        0.99998771, 0.99999834, 0.99999977, 0.99999977, 0.99999834, \
        0.99998771, 0.9999092 , 0.9993293 , 0.99505475, 0.96402758, \
        0.76159416])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid x value type')
    x = input.astype(float)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_gradient(input: np.ndarray) -> np.ndarray:
    '''
    This is a function for calc gradient of tanh.

    Args:
        input: shape (n,m) numpy array. Input data to calc gradient of tanh.

    Returns:
        gradient: shape (n,m) numpy array. The gradient of tanh.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import tanh_gradient
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> tanh_gradient(input)

    >>> #Output:
    >>> array([4.19974342e-01, 7.06508249e-02, 9.86603717e-03, 1.34095068e-03, \
        1.81583231e-04, 2.45765474e-05, 3.32610934e-06, 4.50140597e-07, \
        4.50140597e-07, 3.32610934e-06, 2.45765474e-05, 1.81583231e-04, \
        1.34095068e-03, 9.86603717e-03, 7.06508249e-02, 4.19974342e-01])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    return 1.0 - ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))) ** 2.0


def hard_tanh(input: np.ndarray, min_value: float = -1.0, max_value: float = 1.0, inplace: bool = False):
    '''
    This is a hard tanh function.

    Args:
        input: shape (n,m) numpy array. Input data to calc hard tanh result.
        min_value: float, default 1.0. Left bondery for input of hard tanh. 
        max_value: float, default 1.0. Right bondery for input of hard tanh.
        inplace: bool, default False. Inplace the original data or not.

    Returns:
        hard tanh: shape (n,m) numpy array. The hard tanh result.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import hard_tanh
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> hard_tanh(input)

    >>> #Output:
    >>> array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    x[x < min_value] = -1.0
    x[x > max_value] = 1.0
    if inplace:
        input = x
    return x


def hard_tanh_gradient(input, min_value=-1.0, max_value=1.0, inplace=False):
    '''
    This is a function for calc gradient of hard tanh.

    Args:
        input: shape (n,m) numpy array. Input data to calc gradient of hard tanh.
        min_value: float, default 1.0. Left bondery for input of hard tanh. 
        max_value: float, default 1.0. Right bondery for input of hard tanh.
        inplace: bool, default False. Inplace the original data or not.

    Returns:
        gradient: shape (n,m) numpy array. The gradient of hard tanh.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import hard_tanh_gradient
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> hard_tanh_gradient(input)

    >>> #Output:
    >>> array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    x[x < min_value] = 0.0
    x[(min_value <= x) & (x <= max_value)] = 1.0
    x[x > max_value] = 0.0
    if inplace:
        input = x
    return x


def softplus(input: np.ndarray) -> np.ndarray:
    '''
    This is a softplus function.

    Args:
        input: shape (n,m) numpy array. Input data to calc softplus result.

    Returns:
        softplus: shape (n,m) numpy array, the returned values are always floats. The softplus result.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import softplus
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> softplus(input)

    >>> #Output:
    >>> array([1.31326169, 2.12692801, 3.04858735, 4.01814993, 5.00671535, \
        6.00247569, 7.00091147, 8.00033541, 8.00033541, 7.00091147, \
        6.00247569, 5.00671535, 4.01814993, 3.04858735, 2.12692801, \
        1.31326169])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    return np.log(1.0 + np.exp(x))


def softplus_gradient(input: np.ndarray) -> np.ndarray:
    '''
    This is a function for calc gradient of softplus.

    Args:
        input: shape (n,m) numpy array. Input data to calc gradient of softplus.

    Returns:
        gradient: shape (n,m) numpy array. The gradient of softplus.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import softplus_gradient
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> softplus_gradient(input)

    >>> #Output:
    >>> array([0.50673119, 0.61052201, 0.66027407, 0.68068009, 0.68850805, \
        0.69143329, 0.69251569, 0.69291473, 0.69291473, 0.69251569, \
        0.69143329, 0.68850805, 0.68068009, 0.66027407, 0.61052201, \
        0.50673119])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    return np.exp(x) / (1.0 + np.exp(x)) * np.log(2.0)


def softsign(input: np.ndarray) -> np.ndarray:
    '''
    This is a softsign function.

    Args:
        input: shape (n,m) numpy array. Input data to calc softsign result.

    Returns:
        softsign: shape (n,m) numpy array, the returned values are always floats. The softsign result.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import softsign
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> softsign(input)

    >>> #Output:
    >>> array([0.5       , 0.66666667, 0.75      , 0.8       , 0.83333333, \
        0.85714286, 0.875     , 0.88888889, 0.88888889, 0.875     , \
        0.85714286, 0.83333333, 0.8       , 0.75      , 0.66666667, \
        0.5       ])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    return x / (1.0 + np.abs(x))


def softsign_gradient(input: np.ndarray) -> np.ndarray:
    '''
    This is a function for calc gradient of softsign.

    Args:
        input: shape (n,m) numpy array. Input data to calc gradient of softsign.

    Returns:
        gradient: shape (n,m) numpy array. The gradient of softsign.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import softsign_gradient
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> softsign_gradient(input)

    >>> #Output:
    >>> array([0.25      , 0.11111111, 0.0625    , 0.04      , 0.02777778, \
        0.02040816, 0.015625  , 0.01234568, 0.01234568, 0.015625  , \
        0.02040816, 0.02777778, 0.04      , 0.0625    , 0.11111111, \
        0.25      ])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    return 1.0 / ((1.0 + np.abs(x)) ** 2.0)


def exponential(input: np.ndarray) -> np.ndarray:
    '''
    This is a exponential function.

    Args:
        input: shape (n,m) numpy array. Input data to calc exponential result.

    Returns:
        exponential: shape (n,m) numpy array, the returned values are always floats. The exponential result.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import exponential
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> exponential(input)

    >>> #Output:
    >>> array([2.71828183e+00, 7.38905610e+00, 2.00855369e+01, 5.45981500e+01, \
        1.48413159e+02, 4.03428793e+02, 1.09663316e+03, 2.98095799e+03, \
        2.98095799e+03, 1.09663316e+03, 4.03428793e+02, 1.48413159e+02, \
        5.45981500e+01, 2.00855369e+01, 7.38905610e+00, 2.71828183e+00])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    return np.exp(x)


def exponential_gradient(input: np.ndarray) -> np.ndarray:
    '''
    This is a function for calc gradient of exponential.

    Args:
        input: shape (n,m) numpy array. Input data to calc gradient of exponential.

    Returns:
        gradient: shape (n,m) numpy array. The gradient of exponential.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import exponential_gradient
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> exponential_gradient(input)

    >>> #Output:
    >>> array([2.71828183e+00, 7.38905610e+00, 2.00855369e+01, 5.45981500e+01, \
        1.48413159e+02, 4.03428793e+02, 1.09663316e+03, 2.98095799e+03, \
        2.98095799e+03, 1.09663316e+03, 4.03428793e+02, 1.48413159e+02, \
        5.45981500e+01, 2.00855369e+01, 7.38905610e+00, 2.71828183e+00])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    return np.exp(x)


def softmax(input: np.ndarray, axis=-1) -> np.ndarray:
    '''
    This is a softmax function.

    Args:
        input: shape (n,m) numpy array. Input data to calc softmax result.

    Returns:
        softmax: shape (n,m) numpy array, the returned values are always floats. The softmax result.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import softmax
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> softmax(input)

    >>> #Output:
    >>> array([2.88306385e-04, 7.83698007e-04, 2.13031205e-03, 5.79078854e-03, \
        1.57409953e-02, 4.27884614e-02, 1.16311097e-01, 3.16166341e-01, \
        3.16166341e-01, 1.16311097e-01, 4.27884614e-02, 1.57409953e-02, \
        5.79078854e-03, 2.13031205e-03, 7.83698007e-04, 2.88306385e-04])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    y = np.exp(x - np.max(x, axis, keepdims=True))
    return y / np.sum(y, axis, keepdims=True)


# todo need gradient


def relu(input: np.ndarray, inplace: bool = False) -> np.ndarray:
    '''
    This is a relu function.

    Args:
        input: shape (n,m) numpy array. Input data to calc relu result.
        inplace: bool, default False. Inplace the original data or not.

    Returns:
        relu: shape (n,m) numpy array. The relu result.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import relu
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> relu(input)

    >>> #Output:
    >>> array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    x[x <= 0.0] = 0.0
    if inplace:
        input = x
    return x


def relu_gradient(input: np.ndarray, inplace: bool = False) -> np.ndarray:
    '''
    This is a function for calc gradient of relu.

    Args:
        input: shape (n,m) numpy array. Input data to calc gradient of relu.
        inplace: bool, default False. Inplace the original data or not.

    Returns:
        gradient: shape (n,m) numpy array. The gradient of relu.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import relu_gradient
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> relu_gradient(input)

    >>> #Output:
    >>> array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    x[x > 0.0] = 1.0
    x[x <= 0.0] = 0.0
    if inplace:
        input = x
    return x


def relu6(input: np.ndarray, alpha: float = 0.0, max_value: float = 6.0, threshold: float = 0.0, inplace=False):
    '''
    This is a relu6 function.

    Args:
        input: shape (n,m) numpy array. Input data to calc relu6 result.
        alpha: float, default 0.0 . Slope of negative part.
        max_value: float, default 6.0 . Max output value.
        threshold: float, default 0.0 . Threshold of activation value.
        inplace: bool, default False. Inplace the original data or not.

    Returns:
        relu6: shape (n,m) numpy array. The relu6 result.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import relu6
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> relu6(input)

    >>> #Output:
    >>> array([1., 2., 3., 4., 5., 6., 6., 6., 6., 6., 6., 5., 4., 3., 2., 1.])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    if max_value:
        x[x >= max_value] = max_value
    x[x < threshold] = alpha * (x[x < threshold] - threshold)
    if inplace:
        input = x
    return x


def relu6_gradient(input: np.ndarray, alpha: float = 0.0, max_value: float = 6.0, threshold: float = 0.0,
                   inplace=False):
    '''
    This is a function for calc gradient of relu6.

    Args:
        input: shape (n,m) numpy array. Input data to calc gradient of relu6.
        alpha: float, default 0.0 . Slope of negative part.
        max_value: float, default 6.0 . Max output value.
        threshold: float, default 0.0 . Threshold of activation value.
        inplace: bool, default False. Inplace the original data or not.

    Returns:
        gradient: shape (n,m) numpy array. The gradient of relu6.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import relu6_gradient
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> relu6_gradient(input)

    >>> #Output:
    >>> array([1., 2., 3., 4., 5., 0., 0., 0., 0., 0., 0., 5., 4., 3., 2., 1.])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    if max_value:
        x[x >= max_value] = 0.0
    x[x < threshold] = alpha
    if inplace:
        input = x
    return x


def leaky_relu(input: np.ndarray, negative_slop: float = 0.01, inplace: bool = False):
    '''
    This is a leaky relu function.

    Args:
        input: shape (n,m) numpy array. Input data to calc leaky relu result.
        negative_slop: float, default 0.01 . Slope of negative part.
        inplace: bool, default False. Inplace the original data or not.

    Returns:
        leaky relu: shape (n,m) numpy array. The leaky relu result.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import leaky_relu
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> leaky_relu(input)

    >>> #Output:
    >>> array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    x[x <= 0.0] = negative_slop * x[x <= 0.0]
    if inplace:
        input = x
    return x


def leaky_relu_gradient(input: np.ndarray, negative_slop: float = 0.01, inplace: bool = False):
    '''
    This is a function for calc gradient of leaky relu.

    Args:
        input: shape (n,m) numpy array. Input data to calc gradient of leaky relu.
        negative_slop: float, default 0.01 . Slope of negative part.
        inplace: bool, default False. Inplace the original data or not.

    Returns:
        gradient: shape (n,m) numpy array. The gradient of leaky relu.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import leaky_relu_gradient
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> leaky_relu_gradient(input)

    >>> #Output:
    >>> array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    x[x > 0.0] = 1.0
    x[x <= 0.0] = negative_slop
    if inplace:
        input = x
    return x


def elu(input: np.ndarray, alpha: float = 1.0, inplace: bool = False):
    '''
    This is a elu function.

    Args:
        input: shape (n,m) numpy array. Input data to calc elu result.
        alpha: float, default 1.0 . Slope of negative part.
        inplace: bool, default False. Inplace the original data or not.

    Returns:
        elu: shape (n,m) numpy array. The elu result.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import elu
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> elu(input)

    >>> #Output:
    >>> array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    x[x <= 0.0] = alpha * (np.exp(x[x <= 0.0]) - 1.0)
    if inplace:
        input = x
    return x


def elu_gradient(input: np.ndarray, alpha: float = 1.0, inplace: bool = False):
    '''
    This is a function for calc gradient of elu.

    Args:
        input: shape (n,m) numpy array. Input data to calc gradient of elu.
        alpha: float, default 1.0 . Slope of negative part.
        inplace: bool, default False. Inplace the original data or not.

    Returns:
        gradient: shape (n,m) numpy array. The gradient of elu.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import elu_gradient
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> elu_gradient(input)

    >>> #Output:
    >>> array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    x[x > 0.0] = 1.0
    x[x <= 0.0] = x[x <= 0.0] + alpha
    if inplace:
        input = x
    return x


def selu(input: np.ndarray, inplace: bool = False):
    '''
    This is a selu function.

    Args:
        input: shape (n,m) numpy array. Input data to calc selu result.
        inplace: bool, default False. Inplace the original data or not.

    Returns:
        selu: shape (n,m) numpy array. The selu result.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import selu
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> selu(input)

    >>> #Output:
    >>> array([1.05070099, 2.10140197, 3.15210296, 4.20280395, 5.25350494, \
        6.30420592, 7.35490691, 8.4056079 , 8.4056079 , 7.35490691, \
        6.30420592, 5.25350494, 4.20280395, 3.15210296, 2.10140197, \
        1.05070099])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    x[x > 0.0] = scale * x[x > 0.0]
    x[x <= 0.0] = scale * alpha * (np.exp(x[x <= 0.0]) - 1.0)
    if inplace:
        input = x
    return x


def selu_gradient(input: np.ndarray, inplace: bool = False):
    '''
    This is a function for calc gradient of selu.

    Args:
        input: shape (n,m) numpy array. Input data to calc gradient of selu.
        inplace: bool, default False. Inplace the original data or not.

    Returns:
        gradient: shape (n,m) numpy array. The gradient of selu.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import selu_gradient
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., 8., 7., 6., 5., 4., 3., 2., 1.])
    >>> selu_gradient(input)

    >>> #Output:
    >>> array([1.05070099, 1.05070099, 1.05070099, 1.05070099, 1.05070099, \
        1.05070099, 1.05070099, 1.05070099, 1.05070099, 1.05070099, \
        1.05070099, 1.05070099, 1.05070099, 1.05070099, 1.05070099, \
        1.05070099])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    x[x > 0.0] = scale
    x[x <= 0.0] = scale * alpha * np.exp(x[x <= 0.0])
    if inplace:
        input = x
    return x

# def prelu(input, num_parameters=1, a=0.25):
#     if not np.issubdtype(input.dtype, np.number):
#         raise TypeError('Invalid input value type')
#     input[input<0.0] = a*input[input<0.0]
#     return


# TODO, miss sgn_gradient
def sgn(input: np.ndarray, inplace: bool = False) -> np.ndarray:
    '''
    This is a sgn function.

    Args:
        input: shape (n,m) numpy array. Input data to calc sgn result.
        inplace: bool, default False. Inplace the original data or not.

    Returns:
        sgn: shape (n,m) numpy array. The sgn result.

    -----

    **Examples:**

    >>> from caffeine.utils.activations import sgn
    >>> input = np.array([1., 2., 3., 4., 5., 6., 7., 8., -8., -7., -6., -5., -4., -3., -2., -1.])
    >>> sgn(input)

    >>> #Output:
    >>> array([1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''
    if not np.issubdtype(input.dtype, np.number):
        raise TypeError('Invalid input value type')
    x = input.astype(float)
    x[x >= 0.0] = 1.0
    x[x < 0.0] = 0.0
    if inplace:
        input = x
    return x

