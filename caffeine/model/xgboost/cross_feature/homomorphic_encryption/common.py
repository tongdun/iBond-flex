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
#
#  File name: train
#
#  Create date: 2021/07/29
import os
import json
import numpy as np
from flex.utils.check_security_params import validator, valid_he_gb_ft
from pydantic import BaseModel, PositiveInt, PositiveFloat
from typing import Optional, Dict, List, Union
from caffeine.IO.wafer.util.hooks import Time_hook, Loss_hook

from caffeine.model.base_model import AbstractModel, JsonModel, ModelInfo
from caffeine.model.mixins import FLEXUser, ModelSaver
from caffeine.utils import ClassMethodAutoLog, Context, IBondDataFrame
from caffeine.utils.dataframe import parse_ibonddf
from caffeine.utils.exceptions import DataMismatchError, ArgumentError
from caffeine.utils.federal_commu import FederalInfoParser, Radio
from caffeine.utils.metric import bcl_metrics

HOOKS = {
    'time': Time_hook,
    'loss': Loss_hook
}



class HeteroXGBParams(BaseModel):
    trees: Dict
    feature_columns: List[str]
    lr: float
    loss_type: str
    federal_info: Dict

class HeteroHEXGBTrainParams(BaseModel):
    lr: PositiveFloat = 0.1
    tree_nums: PositiveInt = 10
    max_depth: PositiveInt = 5
    min_samples_leaf: PositiveInt = 50
    min_gain: PositiveFloat = 1e-5
    reg_lambda: PositiveFloat = 0.1
    feature_category_thres: PositiveInt = 10
    loss_type: str = 'BCELoss'
    bin_num: PositiveInt = 10
    gain: str = 'grad_hess'
    early_stop_param: dict = None

    class Config:
        schema_extra = {
            'expose': ['lr', 'max_depth', 'tree_nums']
        }


class HeteroHEXGBPredictParams(BaseModel):
    batch_size: PositiveInt = 4096


class HeteroXGBSecurityParams(BaseModel):
    HE_GB_FT: Optional[List] = [["paillier", {"key_length": 1024}]]

    _ht_gb_ft = validator('HE_GB_FT', allow_reuse=True)(valid_he_gb_ft)


class HeteroHEXGBoostMetaParams(BaseModel):
    train_param: Optional[HeteroHEXGBTrainParams]
    predict_param: Optional[HeteroHEXGBPredictParams]
    security_param: Optional[HeteroXGBSecurityParams]
    federal_info: Optional[Dict]


class HeteroXGBBase(AbstractModel, ModelSaver, FLEXUser):
    _algo_info = {
        "algo_name": "HeteroHEXGBooost",
        "model_type": "algo_ml_2c_eval",
        "federate_type": 0  # 0 for federated model, 1 for local model
    }
    meta_param_model = HeteroHEXGBoostMetaParams
    meta_param_model_dict = {
        'train_param': HeteroHEXGBTrainParams,
        'predict_param': HeteroHEXGBPredictParams,
        'security_param': HeteroXGBSecurityParams
    }

    @ClassMethodAutoLog()
    def __init__(self, meta_param: Dict, context: Context, param: Optional[Dict] = None):
        """
        Common init operations for all participants.

        Args:
            meta_params, dict, a dict of meta parameters:
            {
                'train_param': {
                    'tree_nums': int, optional, total tree nums. Default: 10
                    'max_depth': int, optional,  max depth of tree. Default: 5
                    'min_samples_leaf': int, optional, min number of samples on a leaf node. Default: 50
                    'reg_lambda': float, optional, reg_lambda for leaf value compute. Default: 0.1
                    'bin_num': int, optional, number of binning. Default: 10
                    'loss_type': str, optional, loss function name. Default: 'BCELoss',
                    'lr': float, optional, learning rate. Default: 0.1
                },
                'predict_param': {

                },
                'security': {
                    'key_exchange_size': int, secure aggregation encrypt key length
                },
                'federal_info': dict, federal info
            }
            context: Context, context, e.g. wafer session.
            param: optional dict, if not None, load model from this dict.
        """
        super().__init__(meta_param, context)

        self._parse_meta_param()

        self.trees = dict()
        self._feat_cols = self.feature_columns = None
        self._train_data_info ={}
        self.feature_importance = {}
        self._current_epoch = 0

        #add security review switch
        self.is_review = True if os.environ.get('IBOND_SECURITY_REVIEW') == 'on' else False

        if 'federal_info' not in meta_param and param is None:
            raise ArgumentError('Argument meta_param should contain key "federal_info" if param is None.')

        if param is not None:
            self.logger.info(f'********final load:{param}')
            self.load_params(param)

        if 'train_param' in self._meta_param:
            train_param = self._meta_param['train_param']
            self.lr: float = train_param.get('lr', 0.1)
            self.tree_nums: int = train_param.get('tree_nums', 10)
            self.max_depth: int = train_param.get('max_depth', 5)
            self.min_samples_leaf: int = train_param.get('min_samples_leaf', 50)
            self.reg_lambda: float = train_param.get('reg_lambda', 0.1)
            self.loss_type: str = train_param.get('loss_type', 'BCELoss')
            self.feature_category_thres: int = train_param.get('feature_category_thres', 10)
            self.bin_num: int = train_param.get('bin_num', 10)
            self.min_gain:float = train_param.get('min_gain', 1e-5)
            self.gain: str = train_param.get('gain', 'grad_hess')

            self._check_early_stop_param()

            self.dt_train_params = {
                'max_depth': self.max_depth,
                'min_samples_leaf': self.min_samples_leaf,
                'reg_lambda': self.reg_lambda,
                'loss_type': self.loss_type,
                'feature_category_thres': self.feature_category_thres,
                'bin_num': self.bin_num,
                'min_gain': self.min_gain,
                'gain': self.gain
            }
        else:
            self.dt_train_params = {}

        fed_info = self._meta_param['federal_info']
        self.logger.info(f'federal_info:{fed_info}')

        self.local_id = fed_info['session']['local_id']

        self.dt_meta_param = {
            'train_param': self.dt_train_params,
            'predict_param': {},
            'security': self._meta_param['security_param'] if 'security_param' in self._meta_param.keys() else None,
            'federal_info': fed_info
        }

        self._federal_info_parser = FederalInfoParser(
            self._meta_param['federal_info'])
        self._radio = Radio(
            station_id=self._federal_info_parser.major_guest,
            federal_info=self._federal_info_parser.federal_info,
            channels=['hetero_hexgb_predictions', 'hetero_hexgb_training_loss', 'hetero_hexgb_train_continue']
        )

        self.job_id = self._federal_info_parser.job_id

    @property
    def _federation(self) -> Dict:
        return self._meta_param['federal_info']['federation']

    @property
    def coord_id(self) -> str:
        """
        Return:
            one coordinator id
        """
        coord_id = self._federation['coordinator'][0]
        return coord_id

    @property
    def part_ids(self) -> List[str]:
        """
        Return:
            all participants ids
        """
        part_ids =  self._federation['guest']
        return part_ids

    @ClassMethodAutoLog()
    def before_train(self, train_data: Optional[IBondDataFrame]=None):
        """
        Init training of secure sharing cross feature XGB.

        Args:
            train_data: ibond dataframe, input training data.
        """

        if self.local_id == self._federal_info_parser.major_guest:
            train_continue = (self._current_epoch < self.tree_nums)
            self._radio._hetero_hexgb_train_continue_chan.broadcast(train_continue)
            if self.is_review:
                self.logger.debug(f'SEC_REVIEW | COMMU | SEND-BROADCAST | FROM: guest | TO: all |'
                                  f'jobid: {self.job_id} | tag: {None} | content: is_train_continue is {train_continue}')

        else:
            train_continue = self._radio._hetero_hexgb_train_continue_chan.broadcast()
            if self.is_review:
                self.logger.debug(f'SEC_REVIEW | COMMU | RECV | FROM: guest | TO: all |'
                                  f'jobid: {self.job_id} | tag: {None} | content: is_train_continue is {train_continue}')


        if not train_continue:
            return train_continue

        self._report_callback = self._context.report
        self._report_hooks = ['time', 'loss']
        if train_data is not None:
            self._train_data_info.update(parse_ibonddf(train_data))
            self._feat_cols = self.feature_columns = self._train_data_info['feat_cols']
            if self._report_callback is not None:
                train_data.register_hooks(
                    [HOOKS[hook_name](self._report_callback) for hook_name in self._report_hooks]
                )
            train_data.set_context("max_num_epoch", self.tree_nums)

            self.feature_importance = {}.fromkeys(self.feature_columns, 0.0)

        self.dt._before_train(train_data)

        return train_continue

    @ClassMethodAutoLog()
    def before_one_tree(self, tree_id: int, train_data: Optional[IBondDataFrame]=None):
        """
        Init one decision tree before training.

        Args:
            tree_id: int, tree id of current training decision tree.
            train_data: ibond dataframe, input training data.
        """
        if train_data is not None:
            train_data.set_context("current_epoch", tree_id)
            train_data.set_context("iteration_num", 1)
            train_data.set_context('current_iteration',1)
            [hook.pre_hook(train_data._context) for hook in train_data._hooks]
            # [hook.iter_hook(train_data._context) for hook in train_data._hooks]
            # [hook.post_hook(train_data._context) for hook in train_data._hooks]


        self.dt.nodes = dict()
        self.dt._before_one_tree(train_data, tree_id)


    @ClassMethodAutoLog()
    def update_mode_info(self, tree_id: int, val_data: Optional[IBondDataFrame] = None, model_infos= list):

        preds = self.predict(val_data)
        if val_data and val_data.has_y() and preds is not None:
            val_metrics = bcl_metrics(
                preds.get_pred().to_numpy(),
                val_data.get_y(first_only=True).to_numpy()
            )
            self.logger.info(f'********Part_{self.local_id}_val_metrics:{val_metrics}')
            val_metrics['epoch'] = tree_id
            self._context.report(
                "validation_metrics",
                json.dumps(val_metrics)
            )
        else:
            val_metrics = {'epoch': tree_id}

        model_info = ModelInfo(
            model_id=None,
            model_attributes={
                "metrics": val_metrics,
                "verify_result": val_metrics,
                "epoch_times": tree_id,
                "feature_importance": self.feature_importance,
            }
        )

        model_info = self._save_params(model_info)
        if model_info is not None:
            model_infos.append(model_info)

        return model_infos


    # @ClassMethodAutoLog()
    # def after_train(self, val_data: Optional[IBondDataFrame] = None):
    #     """
    #     Participant save models.
    #     Return:
    #          model_infos: List[str], information of saved models.
    #     """
    #
    #     model_infos = []
    #     model_infos = self.update_mode_info(val_data, model_infos)
    #     return model_infos

    @ClassMethodAutoLog()
    def predict_xgb(self, data: Optional[IBondDataFrame], predict_xgb_id: Optional[str] = 'xgb_tree_') -> np.array:
        """
        Participant compute cross_sample predict of input data.

        Args:
            data: ibond dataframe, input predict data.

        Returns:
            total_tree_preds: np.array. the prediction of cross feature cross_sample.
        """

        all_tree_preds_list = []
        for tree_id, tree_nodes in self.trees.items():
            self.dt.nodes = tree_nodes
            one_tree_preds = self.dt.predict_dt(data, predict_tree_id= predict_xgb_id+'tree_'+tree_id)
            all_tree_preds_list.append(one_tree_preds)

        if self.dt.local_role == 'guest':
            total_tree_preds = np.zeros((data.shape[0], 1))
            for one_tree_preds in all_tree_preds_list:
                total_tree_preds += self.lr * one_tree_preds
            return total_tree_preds
        else:
            return None

    @ClassMethodAutoLog()
    def _save_params(self, model_info: Optional[ModelInfo] = None):
        """
        Save parameters to middleware.
        """
        self._feat_cols = self.feature_columns = self.dt.feature_columns if self.dt.feature_columns is not None else []
        self.logger.info(f'feat_cols: {self.feat_cols}')
        xgb_mod, _ = self.make_module(
            params = HeteroXGBParams(
                trees = self.trees,
                feature_columns = self.feature_columns,
                lr = self.lr,
                loss_type = self.loss_type,
                federal_info = self._meta_param['federal_info']
            )
        )

        if model_info is None:
            model_info = ModelInfo(
                None, {}
            )

        model_info.update_attributes(self._algo_info)
        model_info.update_attributes(self._train_data_info)
        model_id = self.save_modules([xgb_mod],self._meta_param['federal_info'], model_info)
        model_info.update_id(model_id)

        return model_info

    @ClassMethodAutoLog()
    def load_params(self, params: Dict):
        """
        Load parameters into this model.

        Args:
            params, the dict representation of the model.

        Example:
        >>> params = {
                'model_type': 'HeteroXGBGuest',
                'versions': {
                    'caffeine': '0.1',
                    'flex': '1.1',
                    'wafer': '1.0'
                },
                'modules': [
                    {
                        'module_id': 'Module-HeteroXGBGuest-2020-01-01-10-12-33-12345',
                        'module_type': 'Module-HeteroXGBGuest',
                        'params': {
                            'trees': {'0'：{'0'：{'layer_id':0, 'node_id':0,'is_leaf': False, 'split_party_id': 10001}, '1':{...},...},'1':{...}},
                            'feature_columns': ['Age', 'Education-Num',...,'Hours_per_week'],
                            'lr':0.1,
                            'loss_type':'BCELoss',
                        }
                    }
                ]
            }
        >>> model = HomoXGBGuest(meta_param)
        >>> model.load_params(params)
        """
        self.logger.info(f'Parsing params: {params}...')
        model = JsonModel.parse_obj(params)

        trained_epoches = model.attributes.get('epoch_times', 0)
        self._current_epoch = trained_epoches

        self.trees = model.modules[0].params['trees']
        self.logger.info(f'Loading nodes: {self.trees}...')
        self._feat_cols = self.feature_columns = model.modules[0].params['feature_columns']
        self.logger.info(f'Loading feature_columns: {self.feature_columns}...')
        self.lr = model.modules[0].params['lr']
        self.loss_type = model.modules[0].params['loss_type']
        if 'federal_info' not in self._meta_param:
            federal_info = model.modules[0].params['federal_info']
            self.logger.info(
                f'Loading federal_info from model: {federal_info}...')
            self._meta_param['federal_info'] = federal_info
            self.logger.info(
                f'Use loaded federal_info: {self._meta_param["federal_info"]}')
        else:
            self.logger.info(
                f'Use custom federal_info: {self._meta_param["federal_info"]}')
