from typing import Dict, List
import multiprocessing
from multiprocessing import Queue, Process, Manager
from logging import getLogger
import math

import numpy as np
from copy import deepcopy
from flex.constants import IV_FFS
from flex.api import make_protocol
from flex.commu.ionic_so import commu
from collections import OrderedDict

from caffeine.feature.parallel_module import ParallelWorkerBase

logger = getLogger('BinParallelWorker')


class WOEParallelWorker(ParallelWorkerBase):
    def __init__(self, parallel_num):
        super().__init__(parallel_num)

    def run_host(self, length: int, data: np.ndarray, fea_cols: List[str], is_category: List[bool], 
                    is_fillna: List[bool], split_points: list, federal_info: dict, sec_param: dict, algo_param: dict, 
                    en_label: list):
        p_ls = []
        results = Manager().dict()

        idx_list, block_num = self._cal_block(length)
        logger.info(f'*******block_num {block_num} {length} {data.shape}')
        
        for i in range(self.process_num):
            index_list = idx_list[i*block_num: min(length,(i+1)*block_num)]
            if len(index_list) == 0:
                break
            input_params = dict()
            input_params['data'] = data[:, i*block_num:(i+1)*block_num]
            input_params['index_list'] = index_list
            input_params['is_category'] = is_category[i*block_num:(i+1)*block_num]
            input_params['is_fillna'] = is_fillna[i*block_num:(i+1)*block_num]
            input_params['tag'] = i
            input_params['en_label'] = en_label[i]
            input_params['split_points'] = split_points[i*block_num:(i+1)*block_num]
            p = Process(target=host_exchange,
                        args=(
                            input_params,
                            federal_info,
                            sec_param,
                            algo_param,
                            results
                        )
                    )

            p.start()
            p_ls.append(p)

        for p in p_ls:
            p.join()

        infos = self._get_msg_result(results, fea_cols, is_category, is_fillna)
        return infos

    def run_guest(self, length: int, data: np.ndarray, federal_info: dict, sec_param: dict, algo_param: dict, key: list):
        p_ls = []
        idx_list, block_num = self._cal_block(length)
        logger.info(f'****** block_num {block_num} {length}')

        for i in range(self.process_num):
            index_list = idx_list[i*block_num: min(length,(i+1)*block_num)]
            if len(index_list) == 0:
                break
            input_params = dict()
            input_params['data'] = data
            input_params['index_list'] = index_list
            input_params['tag'] = i
            input_params['key'] = key[i]

            # print(f'******** index_list {index_list}')
            p = Process(target=guest_exchange,
                        args=(
                            input_params,
                            federal_info,
                            sec_param,
                            algo_param
                        )
                    )

            p.start()
            p_ls.append(p)

        for p in p_ls:
            p.join()

    def run_coord(self, length: int, federation_info: dict, sec_param: dict=None):
        idx_list, block_num = self._cal_block(length)

        p_ls = []
        for i in range(self.process_num):
            index_list = idx_list[i*block_num: min(length,(i+1)*block_num)]
            if len(index_list) == 0:
                break
            p = Process(target=coord_exchange, 
                        args=(federation_info, i)
                    )

            p.start()
            p_ls.append(p)

        for p in p_ls:
            p.join()

    def run_local(self, instance: object, name: str, input_params: Dict):
        results = Manager().dict()
        msg_queue = multiprocessing.Queue()
        process_list = []
        length = input_params['data'].shape[1]
        idx_list, block_num = self._cal_block(length)

        logger.info(f'****** block_num {block_num} {length}')
        func = getattr(instance, name)
        for i in range(self.process_num):
            d = input_params.copy()
            d['label'] = input_params['label']
            d['data'] = input_params['data'][:, i*block_num:(i+1)*block_num]
            d['is_category'] = input_params['is_category'][i*block_num:(i+1)*block_num]
            d['is_fillna'] = input_params['is_fillna'][i*block_num:(i+1)*block_num]
            d['fea_cols'] = input_params['fea_cols'][i*block_num:(i+1)*block_num]
            d['index'] = idx_list[i*block_num:(i+1)*block_num]
            p = multiprocessing.Process(target=func,
                                        args=(
                                            d,
                                            results,
                                            i
                                        )
                                    )

            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()

        model_info = OrderedDict()
        ivs = []
        for k in sorted(results.keys()):
            model_info.update(results[k][0])
            ivs.extend(results[k][1])
        return model_info, ivs

    def _get_msg_result(self, results, fea_cols, is_category, is_fillna):
        model_info = OrderedDict()
        ivs = []
        for i in sorted(results.keys()):
            v = results[i]
            model_info[fea_cols[i]] = {
                'woe': v[0],
                'iv': v[1]
            }
            ivs.append(v[1])
        
        return model_info, ivs


def host_exchange(input_params, federation_info, sec_param, algo_param, results):
    index_list = input_params['index_list']
    tag = input_params['tag']
    federal_info = deepcopy(federation_info)
    federal_info['session']['job_id'] += "_" + str(tag)
    en_label = input_params['en_label']
    split_points = input_params['split_points']

    protocol = make_protocol(
            IV_FFS,
            federal_info,
            sec_param.get(IV_FFS),
            algo_param
    )
    if en_label is None:
        en_label = protocol.pre_exchange()

    data = input_params['data']
    is_category = input_params['is_category']
    is_fillna = input_params['is_fillna']

    for i in range(len(index_list)):
        tag1 = str(federal_info['session']['job_id']) +"_"+str(i)
        if i < data.shape[1]:
            woes, iv, _ = protocol.exchange(data[:, i], 
                                                en_labels=en_label, 
                                                is_category=is_category[i], 
                                                data_null=bool(1-is_fillna[i]),
                                                split_list=split_points[i],
                                                tag=tag1
                                            )
            results[index_list[i]] = (woes, iv)
        else:
            protocol.exchange(None, 
                                en_labels=en_label, 
                                is_category=None, 
                                data_null=None,
                                split_list=None,
                                tag=tag1
                            )
        logger.info(f'******* processed index {tag} {i} {index_list[i]}')


def guest_exchange(input_params, federation_info, sec_param, algo_param):
    index_list = input_params['index_list']
    tag = input_params['tag']   
    priv_pub_key = input_params['key']

    federal_info = deepcopy(federation_info)
    federal_info['session']['job_id'] += "_" + str(tag)

    protocol = make_protocol(
            IV_FFS,
            federal_info,
            sec_param.get(IV_FFS),
            algo_param
    )
    label = input_params['data']
    if priv_pub_key is None:
        protocol.pre_exchange(label)
    else:
        protocol.over_loading_first_pub_private_key(priv_pub_key)

    for i in range(len(index_list)):
        tag1 = str(federal_info['session']['job_id']) +"_"+str(i)
        protocol.exchange(label, tag=tag1)
        logger.info(f'******* processed index {tag} {i} {index_list[i]}')


def coord_exchange(federation_info, tag):
    federal_info = deepcopy(federation_info)
    federal_info['session']['job_id'] += "_" + str(tag)
    commu.init(federal_info)
    