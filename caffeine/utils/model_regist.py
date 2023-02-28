#!/usr/bin/python3
#
#  Copyright 2021 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#                                                                                              
#  Project name: iBond                                                                         
#                                                                                              
#  File name:                                                                           
#                                                                                              
#  Create date:  2021/3/22                                                                    

from typing import Callable
from enum import Enum, unique

from caffeine.model.trees.cross_feature.guest import HeteroDecisionTreeGuest
from caffeine.model.trees.cross_feature.host import HeteroDecisionTreeHost


@unique
class RegistedModel(Enum):
    '''
    This is Regist table for models which is able to work. All the following models is good to go.
    '''
    SampleAligner = 'SampleAligner'
    HeteroDecisionTreeGuest = 'HeteroDecisionTreeGuest'
    HeteroDecisionTreeHost = 'HeteroDecisionTreeHost'
    HeteroLogisticRegressionGuest = 'HeteroLogisticRegressionGuest'
    HeteroLogisticRegressionHost = 'HeteroLogisticRegressionHost'
    HeteroLogisticRegressionCoord = 'HeteroLogisticRegressionCoord'


def get_model_class(registed_class_name: str) -> Callable:
    return eval(registed_class_name)


def init_estimator(estimator, context, federal_info=None, mode='train'):
    if mode == 'train':
        estimator_param = {
            'train_param': estimator['estimator_param'],
            'federal_info': federal_info,
            'security': estimator['estimator_security'],
        }
    elif mode == 'predict':
        estimator_param = {
            'predict_param': estimator['estimator_param'],
            'security': estimator['estimator_security'],
        }
    else:
        raise Exception(f'Unsupported mode {mode}')
    return eval(estimator['estimator_name'])(estimator_param, context, estimator['estimator_module'])
