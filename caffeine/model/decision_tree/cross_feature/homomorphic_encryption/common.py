#!/usr/bin/python3
#
#  Copyright 2020 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#                                                                                              
#  Project name: iBond                                                                         
#                                                                                              
#  File name:                                                                           
#                                                                                              
#  Create date:  2020/12/29
import os
import json
import queue
from typing import List, Dict, Union, Optional, Tuple

import numpy as np
import pandas as pd
from flex.constants import HE_GB_FT, HE_GB_FP, OTP_PN_FL
from flex.api import make_protocol
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

from caffeine.model.base_model import AbstractModel, JsonModel, JsonModule, ModelInfo
from caffeine.model.mixins import ModelSaver, FLEXUser
from caffeine.utils.dataframe import parse_ibonddf
from caffeine.utils import ClassMethodAutoLog, FunctionAutoLog, IBondDataFrame, Context
from caffeine.utils.exceptions import ArgumentError, ParseError, DataMismatchError, LossTypeError, DataCheckError
from caffeine.utils.federal_commu import FederalInfoParser, Radio
from caffeine.utils.config import security_config
from caffeine.utils.loss import *
from caffeine.utils.metric import bcl_metrics

class HeteroDecisionTreeParams(BaseModel):
    '''
    Params:
        nodes: Dict. Node dict with HeteroDecisionTreeNodeParams.
        federal_info: Dict. Federal info for this tree.
    '''
    nodes: Dict
    feature_columns: List[str]
    loss_type: str
    federal_info: Dict
    min_samples_leaf: float
    reg_lambda: float
    min_gain: float
    gain: str
    feature_importance: Dict

class HeteroHEDTTrainParams(BaseModel):
    gain: str = 'grad_hess'
    max_depth: PositiveInt = 5
    min_samples_leaf: PositiveInt = 50
    reg_lambda: PositiveFloat = 0.1
    loss_type: str = 'BCELoss'
    feature_category_thres: PositiveInt = 10
    min_gain: PositiveFloat = 1e-3
    bin_num: PositiveInt = 10

class HeteroHEDTPredictParams(BaseModel):
    batch_size: PositiveInt = 4096

class HeteroHEDecisionTreeMetaParams(BaseModel):
    train_param: Optional[HeteroHEDTTrainParams]
    predict_param: Optional[ HeteroHEDTTrainParams]
    security_param: Optional[List[List]]
    federal_info: Optional[Dict]

class HeteroDTBase(AbstractModel, ModelSaver, FLEXUser):
    _algo_info = {
        "algo_name": "HeteroHEDecisionTree",
        "model_type": "algo_ml_2c_eval",
        "federate_type": 0  # 0 for federated model, 1 for local model
    }
    meta_param_model = HeteroHEDecisionTreeMetaParams
    meta_param_model_dict = {
        'train_param': HeteroHEDTTrainParams,
        'predict_param': HeteroHEDTPredictParams
    }

    @ClassMethodAutoLog()
    def __init__(self, meta_param: Dict, context: Context, param: Optional[Dict] = None):
        """
        Init decision tree.
        Args:
            meta_params: a dict of meta parameters:
            {
                'train_param': {
                    'max_depth': int, max depth of the tree
                    'reg_lambda': float,
                    'gamma': float,
                    'min_sample_leaf': int,
                    'min_sample_split': int,
                    'ensemble': bool,
                    'gain': str
                },
                OR
                'predict_param': {
                'sec_param': {
                    'he_algo': str, he encrypt method,
                    'he_key_length': int, he encrypt key length
                },
                'federal_info': dict, federal info
            }
        Returns:

        """
        super().__init__(meta_param, context)

        # check meta_param
        self._parse_meta_param()

        self.nodes = dict()
        self.feature_columns = None
        self._train_data_info = {}
        self.feature_importance = {}

        if 'federal_info' not in meta_param and param is None:
            raise ArgumentError('Argument meta_param should contain key "federal_info" if param is None.')

        if param is not None:
            self.logger.info(f'********final load:{param}')
            self.load_params(param)

        if 'train_param' in self._meta_param:
            train_param = self._meta_param['train_param']
            self.gain: str = train_param.get('gain', 'grad_hess')
            self.max_depth: int = train_param.get('max_depth', 5)
            self.min_samples_leaf: int = train_param.get('min_samples_leaf', 50)
            self.reg_lambda: float = train_param.get('reg_lambda', 0.1)
            self.loss_type: str = train_param.get('loss_type', 'BCELoss')
            self.feature_category_thres: int = train_param.get('feature_category_thres', 10)
            self.min_gain: float = train_param.get('min_gain', 1e-3)
            self.bin_num: int = train_param.get('bin_num', 10)
        else:
            self.dt_train_params = {}

        self.init_protocols([
            OTP_PN_FL
        ])

        algo_params = {
            'min_sample_leaf': self.min_samples_leaf,
            'lambda_': self.reg_lambda,
            'gain': self.gain,
        }
        self._protocols[HE_GB_FT] = make_protocol(HE_GB_FT, self._meta_param['federal_info'], security_config.get(HE_GB_FT, None), algo_params)
        self._protocols[HE_GB_FP] = make_protocol(HE_GB_FP, self._meta_param['federal_info'], security_config.get(HE_GB_FP, None), None)

        fed_info = self._meta_param['federal_info']
        self.logger.info(f'***********fed_info_{fed_info}')

        self._federal_info_parser = FederalInfoParser(self._meta_param['federal_info'])
        self.party_id = self._federal_info_parser.local_id
        self.local_role = self._federal_info_parser.local_role
        self.major_guest = self._federal_info_parser.major_guest
        self.job_id = self._federal_info_parser.job_id

        self._radio = Radio(
            station_id=self.major_guest,
            federal_info=self._federal_info_parser.federal_info,
            channels=['hetero_he_dt_predictions', 'hetero_he_dt_training_loss']
        )

    def get_feature_category(self, data: pd.DataFrame) -> Dict:
        '''
        Get feature category dict for input data.
        Args:
            data: shape[n,m], pd.DataFrame. Data for training and testing.

        Returns:
            feature_category: Dict. Dict of feature determined by category feature or not. like: ['xx':True,'xxx':False]
        '''
        feature_category = {}.fromkeys(data.columns, False)
        if self._meta_param.get('category_feature'):
            category_feature_list = self._meta_param['train_param']['category_feature']
            feature_category.update({}.fromkeys(category_feature_list, True))
        return feature_category

    @ClassMethodAutoLog()
    def _before_train(self, data: Optional[IBondDataFrame] = None):
        """
        Init train cross feature decision tree, includes: total feature_number feature_type and loss init.

        Args:
            data: ibond dataframe, train data.

        """
        # check federation info and data
        if len(self._federal_info_parser.participants) < 2:
            raise ArgumentError("Federation participants less than 2")
        if self.local_role == 'guest' and data is None:
            raise DataCheckError("Guest data should be not None")

        # update train data info
        self._train_data_info.update(parse_ibonddf(data))
        self.feature_columns = self._train_data_info['feat_cols'] if data is not None else []
        self.id_cols = self._train_data_info['id_cols'] if data is not None else []
        self.y_cols =  self._train_data_info['y_cols'] if data is not None else []
        # self.local_sample_number = data.shape[0] if data is not None else 0
        # self.local_feature_number = len(self.feature_columns)
        # self._train_data_info.update({
        #     "local_num_id": len(self.id_cols),
        #     "local_num_x": self.local_feature_number,
        #     "local_num_y": len(self.y_cols),
        #     "fed_num_id": len(self.id_cols),
        #     "fed_num_x": self.total_feature_number,
        #     "fed_num_y": 1
        # })

        # guest init label and pred
        if self.local_role == 'guest':
            self.y = data[self.y_cols].to_numpy()[:, 0].reshape(-1, 1)
            self.y_pred = np.zeros_like(self.y) + 0.5
            self.logger.info(f'********guest_y_shape:{self.y.shape}')

        # init loss
        if self.loss_type == 'BCELoss':
            self.loss = BCELoss()
        elif self.loss_type == 'MSELoss':
            self.loss = MSELoss()
        elif self.loss_type == 'HingeLoss':
            self.loss = HingeLoss()
        else:
            raise LossTypeError("loss type not support yet")

        self.data_bin={}
        for col_name in self.feature_columns:
            data_col = data.toSeries(col_name)
            unique_value_and_count = data_col.value_counts(ascending=True).to_dict()
            unique_value = list(unique_value_and_count.keys())
            unique_value.sort()
            self.data_bin[col_name] = unique_value
            self.logger.info(f'feature_{col_name}_unique_value:{self.data_bin[col_name]}')
        self.feature_category = self.get_feature_category(data)
        self.feature_importance = {}.fromkeys(self.feature_columns, 0.0)

    @ClassMethodAutoLog()
    def compute_continious_feature_split_data(self, col_data: np.array) -> List:
        """
        Compute continious feature split data.

        Args:
            col_data: np.array, continious feature column data.

        Returns:
            split_data: list, all split data for col_data, like [split_data1, split_data2, ...]
        """
        split_data = []
        col_data = np.sort(col_data)
        index_list = [int(len(col_data) * (float(i + 1) / self.bin_num) - 1) for i in range(self.bin_num)]
        if index_list[-1] != len(col_data) - 1:
            index_list[-1] = len(col_data) - 1

        new_index_list = [index_list[0]]
        for i in range(1, self.bin_num):
            if col_data[index_list[i]] != col_data[new_index_list[-1]]:
                new_index_list.append(index_list[i])

        for i in range(len(new_index_list)):
            split_data.append(col_data[new_index_list[i]])
        if split_data[-1] < col_data[-1]:
            split_data.append(col_data[-1])

        del col_data

        return split_data

    @ClassMethodAutoLog()
    def _before_one_tree(self, data: Optional[IBondDataFrame] = None, tree_id: Optional[int]= 0):
        """
        Guest and hosts secure sharing g/h/index for each sample and get sample index in each bin.

        Args:
            data: ibond dataframe, input training data.
            tree_id: int, id of current training decision tree, default: 0.
        """

        self.logger.info(f'********************************{self.local_role}_gain:{self.gain}')

        if self.local_role == 'guest':
            if self.gain == 'grad_hess':
                grad = self.loss.gradient(self.y_pred, self.y).reshape(-1, 1)
                hess = self.loss.hessian(self.y_pred, self.y).reshape(-1, 1)
                self.hist_base = np.concatenate([grad,hess], axis=1)
            elif self.gain == 'gini':
                self.hist_base = copy.deepcopy(self.y)

            self._protocols[HE_GB_FT].pre_exchange(self.hist_base, tree_id=tree_id)  # todo 这里flex需要修改，不要把guest的内容都加密了，这样guest计算hist能快很多。
        elif self.local_role == 'host':
            self.hist_base = self._protocols[HE_GB_FT].pre_exchange(tree_id=tree_id)

        self.tree_node_idx = 0

    @ClassMethodAutoLog()
    def train_dt(self, data: Optional[IBondDataFrame]=None, tree_id: Optional[int] = 0):
        """
        Guest train cross feature decision tree and update its nodes.
        Args:
            data: ibond dataframe, input training data.
            tree_id: int, tree id of current training decision tree, default: 0.
        """

        q = queue.Queue()
        q.put((0, 1, np.array(list(range(data.shape[0])))))
        while not q.empty():
            nid, depth, node_data_id = q.get()
            self.logger.info(f'Start {nid + 1}-th node')
            histogram = self.calculate_histogram(data, node_data_id)
            if self.local_role == 'guest':
                self.logger.info(f'*********{self.local_role}_{self.party_id}_node_id_{nid+1}_hist:{histogram}')
            tag = f'tree_id_{tree_id}_nid_{nid}'
            max_gain, best_pid, best_feature_name, best_bid, weight = self._protocols[HE_GB_FT].exchange(histogram, self.feature_category, self.gain, tag=tag)

            self.logger.info(f'*********{self.local_role}_{self.party_id}_node_id_{nid+1}_FLEX_max_gain_best_pid_best_feature_name,best_bid:{max_gain, best_pid, best_feature_name, best_bid}')

            if best_pid is None:
                self.nodes[str(nid)] ={'nid': nid,'party_id':best_pid, 'is_leaf': True, 'weight':weight}
                self.logger.info(f'Complete {nid + 1}-th node')
                continue

            if best_pid == self.party_id:
                node_cnt = len(node_data_id)
                node_cnt_l = np.sum(histogram[best_feature_name]['count'][:best_bid+1])
                node_cnt_r = node_cnt - node_cnt_l
                split = node_cnt_l >= self.min_samples_leaf and \
                        node_cnt_r >= self.min_samples_leaf and \
                        max_gain > self.min_gain and \
                        depth < self.max_depth
                self.logger.info(f'*********{self.local_role}_{self.party_id}__node_id_{nid+1}_node_lnode_rnode_data_nums_and_split:{node_cnt, node_cnt_l, node_cnt_r, split}')
                if split:
                    col_data_numpy = data[best_feature_name].to_numpy()
                    split_data = self.data_bin[best_feature_name][best_bid]
                    left_id_mask = np.where(col_data_numpy <= split_data)[0]
                    left_data_id = np.intersect1d(node_data_id, left_id_mask)
                    right_data_id = np.setdiff1d(node_data_id, left_data_id)
                    left_nid = self.tree_node_idx + 1
                    right_nid = self.tree_node_idx + 2
                    self._protocols[HE_GB_FP].exchange(best_pid, left_data_id, tag=tag)
                    self.nodes[str(nid)] = {'nid': nid, 'party_id': best_pid, 'is_leaf': False, 'split_feature': best_feature_name, 'split_data': split_data, 'left_nid': left_nid, 'right_nid': right_nid}
                    self.tree_node_idx += 2
                    self.feature_importance[best_feature_name] += max_gain
                    q.put((left_nid, depth + 1, left_data_id))
                    q.put((right_nid, depth + 1, right_data_id))
                else:
                    self._protocols[HE_GB_FP].exchange(best_pid, None, tag=tag)
                    self.nodes[str(nid)] = {'nid': nid, 'party_id': best_pid, 'is_leaf': True, 'weight': weight}
            else:
                left_data_id = self._protocols[HE_GB_FP].exchange(best_pid, None, tag=tag)
                if left_data_id is not None:
                    left_nid = self.tree_node_idx + 1
                    right_nid = self.tree_node_idx + 2
                    right_data_id = np.setdiff1d(node_data_id, left_data_id)
                    self.nodes[str(nid)] = {'nid': nid, 'party_id': best_pid, 'is_leaf': False, 'left_nid': left_nid, 'right_nid': right_nid}
                    self.tree_node_idx += 2
                    q.put((left_nid, depth + 1, left_data_id))
                    q.put((right_nid, depth + 1, right_data_id))
                else:
                    self.nodes[str(nid)] = {'nid': nid, 'party_id': best_pid, 'is_leaf': True, 'weight': weight}
            self.logger.info(f'Complete {nid + 1}-th node')

    @ClassMethodAutoLog()
    def predict_dt(self, data: Optional[IBondDataFrame] = None, predict_tree_id: Optional[str] = '0') -> np.array:
        """
        Predict SS_result on all leaf nodes for input data.

        Args:
            data: ibond dataframe, input data of prediction.

        Return:
            SS_result: np.array, secure_sharing prediction of input data.
        """
        tree_tag = 'predict_tree_' + predict_tree_id

        self.logger.info(f'********{self.local_role}_{self.party_id}_predict_nodes:{self.nodes}')

        q = queue.Queue()
        q.put((0, np.array(list(range(data.shape[0])))))
        leaf_node_ids = {}
        while not q.empty():
            nid, node_data_id = q.get()
            node = self.nodes[str(nid)]
            if node['is_leaf']:
                leaf_node_ids.update({str(nid): node_data_id})
                continue
            if node['party_id'] == self.party_id:
                col_data_numpy = data[node['split_feature']].to_numpy()
                left_id_mask = np.where(col_data_numpy <= node['split_data'])[0]
                left_data_id = np.intersect1d(node_data_id, left_id_mask)
                right_data_id = np.setdiff1d(node_data_id, left_data_id)
            else:
                left_data_id = node_data_id.copy()
                right_data_id = node_data_id.copy()
            q.put((node['left_nid'], left_data_id))
            q.put((node['right_nid'], right_data_id))
        self.logger.info(f'predcit decision tree complete')

        if self.local_role == 'host':
            self._protocols[OTP_PN_FL].host_param_broadcast(leaf_node_ids, tag=tree_tag)
            return None

        if self.local_role == 'guest':
            predictions = np.zeros((data.shape[0], 1))
            host_leaf_node_ids_list = self._protocols[OTP_PN_FL].host_param_broadcast(tag=tree_tag)
            for host_leaf_node_ids in host_leaf_node_ids_list:
                assert len(leaf_node_ids) == len(host_leaf_node_ids)
            for key, value in leaf_node_ids.items():
                for idx in range(len(host_leaf_node_ids_list)):
                    value = np.intersect1d(value, host_leaf_node_ids_list[idx][key])
                predictions[value] = self.nodes[key]['weight']
            return predictions

    def calculate_histogram(self, data: IBondDataFrame, node_data_id: np.array) -> Dict:
        histogram = dict()
        if self.gain == 'grad_hess':
            hist_base = self.hist_base[node_data_id]
            g = hist_base[:, 0]
            h = hist_base[:, 1]
            for feature in self.feature_columns:
                col_data = data[feature].to_numpy()
                node_data = col_data[node_data_id]
                bin_num = len(self.data_bin[feature])
                his = np.zeros((bin_num, 3)).astype(object)
                for num in range(bin_num):
                    bin_idx = np.where(node_data == self.data_bin[feature][num])[0]
                    # bin_idx = np.where(node_data > self.data_bin[feature][num][0]) & np.where(node_data <= self.data_bin[feature][num][1])
                    his[num] = (len(bin_idx), g[bin_idx].sum(), h[bin_idx].sum())
                histogram[feature] = dict()
                histogram[feature]['count'] = his[:, 0]
                histogram[feature]['grad'] = his[:, 1]
                histogram[feature]['hess'] = his[:, 2]
        elif self.gain == 'gini':
            pass
        else:
            raise ValueError('Invalid gain type')
        return histogram

    @ClassMethodAutoLog()
    def _after_train(self, val_data: Optional[IBondDataFrame] = None):
        """
        Participant save models.
        Return:
             model_infos: List[str], information of saved models.
        """

        preds = self.predict(val_data, predict_id='after_train_dt_')

        if val_data and val_data.has_y() and preds is not None:
            val_metrics = bcl_metrics(
                preds.get_pred().to_numpy(),
                val_data.get_y(first_only=True).to_numpy()
            )
            self.logger.info(f'********Part_{self.party_id}_val_metrics:{val_metrics}')
            self._context.report("validation_metrics", json.dumps(val_metrics))
        else:
            val_metrics = None

        model_info = ModelInfo(
            model_id=None,
            model_attributes={
                "metrics": val_metrics,
                "verify_result": val_metrics,
                "epoch_times": 0,
            }
        )

        model_infos = []
        model_info = self._save_params(model_info)
        if model_info is not None:
            model_infos.append(model_info)
        return model_infos

    @ClassMethodAutoLog()
    def _save_params(self, model_info: Optional[ModelInfo] = None):
        """
        Save parameters to middleware.
        """
        hedt_mod, _ = self.make_module(
            params=HeteroDecisionTreeParams(
                nodes=self.nodes,
                feature_columns=self.feature_columns,
                loss_type=self.loss_type,
                federal_info=self._meta_param['federal_info'],
                min_samples_leaf=float(self.min_samples_leaf),
                reg_lambda=self.reg_lambda,
                min_gain=self.min_gain,
                gain = self.gain,
                feature_importance = self.feature_importance,
            )
        )

        if model_info is None:
            model_info = ModelInfo(
                None, {}
            )

        model_info.update_attributes(self._algo_info)
        model_info.update_attributes(self._train_data_info)
        model_id = self.save_modules([hedt_mod], self._meta_param['federal_info'], model_info)
        model_info.update_id(model_id)

        return model_info

    @ClassMethodAutoLog()
    def load_params(self, params: Dict):
        self.logger.info(f'Parsing params: {params}...')
        model = JsonModel.parse_obj(params)

        self.nodes = model.modules[0].params['nodes']
        self.logger.info(f'Loading nodes: {self.nodes}...')
        self.feature_columns = model.modules[0].params['feature_columns']
        self.logger.info(f'Loading feature_columns: {self.feature_columns}...')
        self.loss_type = model.modules[0].params['loss_type']
        federal_info = model.modules[0].params['federal_info']
        self._meta_param['federal_info'] = federal_info
        self.min_samples_leaf = int(model.modules[0].params['min_samples_leaf'])
        self.reg_lambda = model.modules[0].params['reg_lambda']
        self.min_gain = model.modules[0].params['min_gain']
        self.gain = model.modules[0].params['gain']
        self.feature_importance = model.modules[0].params['feature_importance']
        self.logger.info(f'Model loaded.')