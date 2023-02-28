#!/usr/bin/python3
#
#  Copyright 2020 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#
#  Project name: iBond
#
#  File name:
#
#  Create date:  2020/12/25

import numpy as np
import time

from caffeine.utils import activations

activation_function = ['sigmoid', 'sigmoid_gradient',
                       'hard_sigmoid', 'hard_sigmoid_gradient',
                       'logsigmoid', 'logsigmoid_gradient',
                       'linear', 'linear_gradient',
                       'tanh', 'tanh_gradient',
                       'hard_tanh', 'hard_tanh_gradient',
                       'softplus', 'softplus_gradient',
                       'softsign', 'softsign_gradient',
                       'exponential', 'exponential_gradient',
                       'softmax',      #'softmax_gradient',
                       'relu',         'relu_gradient',
                       'relu6',        'relu6_gradient',
                       'leaky_relu',   'leaky_relu_gradient',
                       'elu',          'elu_gradient',
                       'selu',         'selu_gradient']

def call(func_name, input):
    return activations.__dict__[func_name](input)


def test_1d_activations():
    # You can add any activation_function in here for test.
    raw = np.arange(-100, 100, 0.001, dtype=np.float32)

    test_cases = [raw,
                raw.astype(int),
                raw.reshape(1, -1),
                np.array([raw, raw])]

    for func_name in activation_function:
        for test_case in test_cases:
            start = time.time()
            result = call(func_name, test_case)
            print(f'Test {func_name} takes {time.time()-start} seconds.')
	    # TODO remove ascending check
            #assert np.all(result[1:] - result[:-1] >= 0)


def test_2d_activations():
    # You can add any activation_function in here for test.
    raw = np.arange(-100, 100, 0.001, dtype=np.float32)

    test_cases = [raw.reshape(1, -1),
                  raw.reshape(100 , -1)]

    for func_name in activation_function:
        for test_case in test_cases:
            start = time.time()
            result = call(func_name, test_case)
            result = result.flatten()
            print(f'Test {func_name} takes {time.time()-start} seconds.')
	    # TODO remove ascending check
            #assert np.all(result[1:] - result[:-1] >= 0)

if __name__ == '__main__':
    test_1d_activations()
    test_2d_activations()
