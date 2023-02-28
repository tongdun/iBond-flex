#!/usr/bin/python3
#
#  Copyright 2021 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#                                                                                              
#  Project name: iBond                                                                         
#                                                                                              
#  File name:                                                                           
#                                                                                              
#  Create date:  2021/4/6                                                                    

import random


def goss(data, top_percent, random_percent, gradient_schema='grad'):
    # todo 之后基于数据的类型进行修正即可使用
    length = data.shape[0]
    top_data_num = length * top_percent
    random_data_num = length * random_percent
    fact = (1 - top_percent) / random_percent
    sorted_data = sorted(data.items(), key=lambda x: x[gradient_schema], reverse=True)
    idx = list(range(length))
    top_data_idx = idx[:top_data_num]
    random_data_idx = random.sample(idx[top_data_num:, ], random_data_num)
    data[top_data_idx, gradient_schema] *= fact
    output_data = sorted_data[top_data_idx + random_data_idx, :]
    return output_data
