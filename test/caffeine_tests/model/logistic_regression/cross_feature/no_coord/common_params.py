#!/usr/bin/python3
#
#  Copyright 2021 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#
#  Project name: iBond
#
# -*- coding: utf-8 -*-
# @Time    : 2022/11/30 19:22
# @Author  : iBond Authors
# @File    : commoe_params.py
# -*- File description -*-
# -*- File description -*-
from functools import partial

from caffeine.model.logistic_regression.cross_feature.no_coord.guest import HeteroLogisticRegressionNoCoordGuest
from caffeine.model.logistic_regression.cross_feature.no_coord.host import HeteroLogisticRegressionNoCoordHost
from caffeine_tests.model.hetero_mainflow import case_template_no_coord, run_processes

##############################################################################
case_prefix = 'hetero_lr_nocoord_'

learners = [
    HeteroLogisticRegressionNoCoordGuest,
    HeteroLogisticRegressionNoCoordHost,
]

test_cases = {
    'uci_credit_default': case_template_no_coord(
        train_files = [
            '/mnt/nfs/datasets/UCI_credit/hetero/train_2_0.csv',
            '/mnt/nfs/datasets/UCI_credit/hetero/train_2_1.csv'
        ],
        test_files = [
            '/mnt/nfs/datasets/UCI_credit/hetero/test_2_0.csv',
            '/mnt/nfs/datasets/UCI_credit/hetero/test_2_1.csv'
        ],
        descs = [
            {'id_desc': ['id'], 'y_desc': ['default.payment.next.month']},
            {'id_desc': ['id']}
        ],
        meta_params = {
            'train_param': {
                'encryptor': {
                    'type': 'Paillier',
                    'parameters': {
                        'key_length': 1024,
                    }
                },
            },
        },
        learners = learners,
        metrics = {
            'ks': lambda x: x > 0.35,
            'auc': lambda x: x > 0.7
        },
        timeout = 600
    )
}

test_case = test_cases['uci_credit_default']