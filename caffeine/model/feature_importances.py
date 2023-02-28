#!/usr/bin/python3
#
#  Copyright 2021 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#                                                                                              
#  Project name: iBond                                                                         
#                                                                                              
#  File name:                                                                           
#                                                                                              
#  Create date:  2021/3/14
from typing import Dict, Optional

#todo 这里主要是基于保存的模型去提取feature importances，便于模型的联合feature_importances的计算

def get_feature_importances(model, previous_feature_importance: Optional[Dict]):
    if previous_feature_importance:
        for column in previous_feature_importance:
            previous_feature_importance[column] = model._feature_importance[column]
        return previous_feature_importance
    else:
        return model._feature_importance


# ensemble = {
#     'n_estimators': 0, #这个n_estiamtors 改动量童谣兼容预测的过程
#     'learning_rate': 0.001,
#     'checkpoint_interval': 5,
# }
# meta_param = {'ensemble_param': List, 'estimator_params':List, 'estimator_names':List, 'federal_info':Dict}

'''
每个estimator的param包括：该有的都有，
每个estimator的信息，都需要靠自己生成，而不是让模型去整理。
ensemble 的方式直接在测试的过程中进行定义即可
给到了ensemble里面以后通过key来控制
还要给到cpu核数

'''
