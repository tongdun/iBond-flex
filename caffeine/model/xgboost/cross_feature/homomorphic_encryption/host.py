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
#  Create date: 2021/07/30
from typing import Optional, Dict, List


from caffeine.model.decision_tree.cross_feature.homomorphic_encryption.host import HeteroDTHost
from caffeine.model.xgboost.cross_feature.homomorphic_encryption.common import HeteroXGBBase
from caffeine.utils import ClassMethodAutoLog, Context, IBondDataFrame
from caffeine.utils.exceptions import NotTrainedError

class HeteroXGBHost(HeteroXGBBase):
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
        super().__init__(meta_param, context, param)

        self.dt = HeteroDTHost(self.dt_meta_param, self._context)

    @ClassMethodAutoLog()
    def train(self, train_data:  Optional[IBondDataFrame] = None, val_data: Optional[IBondDataFrame] = None) -> List[str]:
        """
        Train cross sample cross_sample.

        Args:
            train_data, ibond dataframe, train data.
            val_data, optional ibond dataframe, validation data.

        Returns:
            model_infos: List[str], information of saved models.
        """
        model_infos = []
        train_continue = self.before_train(train_data)
        if not train_continue:
            return model_infos

        for tree_id in range(self._current_epoch+1, self.tree_nums+1):
            self.logger.info(f'********Host_train_tree_{tree_id}_start')
            self.before_one_tree(tree_id, train_data)
            self.dt.train_dt(train_data, tree_id)
            self.after_one_tree(tree_id, train_data)
            model_infos = self.update_mode_info(tree_id, val_data, model_infos)
            self.logger.info(f'********Host_train_tree_{tree_id}_end')

            if self.early_stop_param['early_stop'] is True and self.early_stop_param['early_stop_step'] == 'iter' and self.converge_flag is True:
                break


        # model_infos = self.after_train(val_data)
        return model_infos

    @ClassMethodAutoLog()
    def after_one_tree(self, tree_id: int, data: IBondDataFrame):
        """
        Update models and training data predictions after one tree trained.

        Args:
            tree_id: int, id of current train tree.
            data: ibond dataframe, input training data.
        """
        self.trees[str(tree_id)] = self.dt.nodes
        self.predict_xgb(data,predict_xgb_id='xgb_after_tree_'+str(tree_id)+'_trained_')

        remote_loss = self._radio._hetero_hexgb_training_loss_chan.broadcast()

        if self.is_review:
            self.logger.debug(f'SEC_REVIEW | COMMU | RECV | FROM: guest | \
                                       TO: all | desc: Cross-feat XGB guest send loss | \
                                       job_id: {self.job_id} | tag: {None} | content: Guest_after_train_{tree_id}_trees_loss:{remote_loss}')
        data.set_context("loss", remote_loss)

        if self.early_stop_param['early_stop'] is True and self.early_stop_param['early_stop_step'] == 'iter':
            self.converge_flag = self.converge_func.is_converge(remote_loss)

        if data is not None:
            # [hook.pre_hook(data._context) for hook in data._hooks]
            [hook.iter_hook(data._context) for hook in data._hooks]
            [hook.post_hook(data._context) for hook in data._hooks]

        for column in self.feature_columns:
            self.feature_importance[column]  += self.dt.feature_importance[column]

    @ClassMethodAutoLog()
    def predict(self, data: Optional[IBondDataFrame] = None, predict_id: Optional[str] = '*'):
        """
        Model prediction interface.

        Args:
            data: ibond dataframe.
            predict_id: str, predict id for broadcast channel.

        Return:
            prediction: ibond dataframe. predicts of input data.
        """
        #check guest/host val data shape is equal or not
        data_shape = data.shape[0] if data is not None else 0

        if data_shape == 0:
            self.logger.info(f'host predict data is None')
            return

        self.logger.info(f'Host predict current trees:{self.trees}')
        ### check training status
        if self.trees is None:
            raise NotTrainedError('Host cross feature cross_sample model has not been trained.')

        self.dt.feature_columns = self.feature_columns

        self.predict_xgb(data,predict_xgb_id=predict_id)

        prediction = self.dt._radio._hetero_he_dt_predictions_chan.broadcast(tag=predict_id)

        return prediction