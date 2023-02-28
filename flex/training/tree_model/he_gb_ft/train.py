#
#  Copyright 2020 The FLEX Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import os
import time
from typing import Optional, Dict, List, Union

import numpy as np

from flex.cores.base_model import BaseModel, send_pubkey, get_pubkey
from ..split_model import SplitModel
from flex.cores.parser import parse_algo_param, AlgoParamParser
from flex.algo_config import HE_GB_FT_PARAM
from flex.utils import ClassMethodAutoLog, FunctionAutoLog


class NodeBaseModel(BaseModel):
    """
        Hetero tree node split
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List],
                 algo_param: Optional[Dict]):
        """
        Tree node split protocol param inits
        inits of federation information for communication and secure params for security calculation

        Args:
            federal_info: dict, federal info
            sec_param: list, params for security calc
            algo_param: dict, params for algo
        ----

        **Example**:
        >>> federal_info = {
        >>>    "server": "localhost:6001",
        >>>    "session": {
        >>>        "role": "guest",
        >>>        "local_id": "zhibang-d-014011",
        >>>        "job_id": 'test_job',
        >>>    },
        >>>    "federation": {
        >>>        "host": ["zhibang-d-014010"],
        >>>        "guest": ["zhibang-d-014011"],
        >>>        "coordinator": ["zhibang-d-014012"]
        >>>    }
        >>> }

        >>> sec_parma = [['paillier', {"key_length": 1024}], ]

        >>> algo_param = {
        >>>     'min_sample_leaf': 20,
        >>>     'lambda_': 0.1,
        >>>     'gain': 'gini'   # grad_hess of xgboost model
        >>> }
        """
        BaseModel.__init__(self, federal_info, sec_param)
        self._init_encrypt()
        self.algo_param = _parse_algo_param(algo_param)

        self.job_id = federal_info["session"]["job_id"]

        # channel inits
        self.data_channel = self.commu.guest2host_broadcast_channel('data')
        self.label_channel = self.commu.guest2host_broadcast_channel('label')
        self.key = self.commu.guest2host_broadcast_channel('pubkey')

    @ClassMethodAutoLog()
    def _feature_encrypt(self, data: Dict) -> Dict:
        """
        This method mainly update encrypt value of histgram

        Args:
            data: dict, save feature msg, full of counts, gradients and hessian

        Returns:
            dict, all encrypt value
        ----
        """
        for feature in data:
            index = np.where(data[feature]['count'] == 0)[0]

            # values encrypt
            if self.algo_param.gain == 'gini':
                data[feature]['y'][index] = self.pf_ecc.encrypt(data[feature]['y'][index])
            elif self.algo_param.gain == 'grad_hess':
                data[feature]['grad'][index] = self.pf_ecc.encrypt(data[feature]['grad'][index])
                data[feature]['hess'][index] = self.pf_ecc.encrypt(data[feature]['hess'][index])
        return data


class HEGBFTGuest(NodeBaseModel):
    """
    Hetero tree node split protocol, guest
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List],
                 algo_param: Optional[Dict]):
        NodeBaseModel.__init__(self,
                               federal_info=federal_info,
                               sec_param=sec_param,
                               algo_param=algo_param)
        # send pubkey to host
        send_pubkey(self.key, self.pf_ecc.en)

    @ClassMethodAutoLog()
    def pre_exchange(self, label: np.ndarray, tree_id=0, *args, **kwargs) -> None:
        """
        First protocol of hetero tree split protocol, send encrypt label mess to host

        Args:
            label: samples label

        Returns:
             None
        ----

        **Example:**
        >>> label = np.array([0, 1, 0, 1, 0, 0])
        >>> self.pre_exchange(label)
        """
        # encrypt label
        en_label = self.pf_ecc.encrypt(label)

        # send label to host
        self.label_channel.broadcast(en_label)

    @ClassMethodAutoLog()
    def exchange(self, data: Dict,
                 is_category: Optional[Dict],
                 *args, **kwargs) -> Optional[tuple]:
        """
        Second protocl of hetero tree split protocol, mialy find best split point

        Args:
            data: dict, main contain all mess of count/grad/hessian in each feature:
            is_category: Dict, judge continuous(False)/category(True) of feature

        Returns:
            max_gain: gain of this split
            party_id: party msg of optimal classification
            best_feature: best split feature
            best_split_point: split point of best feature
            weight: value of weight for split node
        ----

        **Example**:
        >>> data = {
        >>>     'k1':{
        >>>         'count': np.array([10, 25, 25]),
        >>>         'grad': np.array([0.6, 0.8, -1.2]),
        >>>         'hess': np.array([0.9, 0.8, 0.5])
        >>>     },
        >>>     'k2':{
        >>>         'count': np.array([20, 10, 15, 15]),
        >>>         'grad': np.array([1.1, 1.5, -2.3, -0.5]),
        >>>         'hess': np.array([1.3, -0.8, 2.2, -0.5])
        >>>     }
        >>> }

        >>> is_category = {
        >>>     'k1': False, 'k2': True
        >>> }

        >>> self.exchange(data, is_category)
        """
        # check data type
        is_category = _data_type_check(data, is_category)

        # save feature msg
        feature_list = list()
        feature_list.append(list(data.keys()))

        # get encrypt data from host
        host_data_gather = self.data_channel.gather()

        self.logger.info(f'guest complete get split message from host')

        # append all host data to guest data
        for i, host_data in enumerate(host_data_gather):
            host_data, is_category_host = host_data


            # host data decrypt
            for feature in host_data:
                for key in host_data[feature]:
                    if key != 'count':
                        host_data[feature][key] = self.pf_ecc.decrypt(host_data[feature][key])

            # merge guest and host data
            data.update(host_data)
            is_category.update(is_category_host)

            # add host feature msg
            feature_list.append(list(host_data.keys()))
        self.logger.info('guest complete guest/host combine')

        # find best split point and index
        best_split = SplitModel(lambda_=self.algo_param.lambda_,
                                gain=self.algo_param.gain,
                                min_samples_leaf=self.algo_param.min_samples_leaf)
        weight, max_gain, best_feature, best_split_point, max_left_weight, max_right_weight \
            = best_split.calc_best_split(data=data, is_category=is_category)
        self.logger.info('guest complete best split message calculation')

        # node can't split return all None value
        if best_feature is None:
            split_info = [(None, None, None, None, weight)] * len(self.federal_info.guest_host)
        else:
            # best split feature party msg
            for i, feature in enumerate(feature_list):
                if best_feature in feature:
                    party_id = self.commu.guest_host[i]

            # generate send msg
            split_info = []
            for i, feature in enumerate(feature_list):
                if best_feature in feature:
                    split_info.append((max_gain, party_id, best_feature, best_split_point, weight))
                else:
                    split_info.append((None, party_id, None, None, weight))
        result = split_info[0]
        split_info.pop(0)

        # guest broadcast split msg
        self.data_channel.scatter(split_info)

        self.logger.info('send split result to host')

        return result


class HEGBFTHost(NodeBaseModel):
    """
    Hetero tree node split protocol, host
    """

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List],
                 algo_param: Optional[Dict]):
        NodeBaseModel.__init__(self,
                               federal_info=federal_info,
                               sec_param=sec_param,
                               algo_param=algo_param)
        # get pubkey from guest
        self.pf_ecc.en = get_pubkey(self.key)


    @ClassMethodAutoLog()
    def pre_exchange(self, tree_id=0, *args, **kwargs) -> List:
        """
        First protocol of hetero tree split protocol, get encrypt label mess from guest

        Returns:
             encrypt message of label
        ----

        **Example**
        >>> self.pre_exchange()
        """
        # get encrypt label from guest
        label_en = self.label_channel.broadcast()
        return label_en

    @ClassMethodAutoLog()
    def exchange(self, data: Dict,
                 is_category: Optional[Dict],
                 *args, **kwargs) -> Union[None, tuple]:
        """
        Second protocl of hetero tree split protocol, mialy find best split point

        Args:
            data: dict, main contain all mess of count/grad/hessian in each feature:
            is_category: Dict, judge continuous(False)/category(True) of feature

        Returns:
            max_gain: gain of this split
            party_id: party msg of optimal classification
            best_feature: best split feature
            best_split_point: split point of best feature
            weight: value of weight for split node
        ----

        **Example:**
        >>> data = {
        >>>     'f1':{
        >>>         'count': np.array([10, 30, 20]),
        >>>         'grad': np.array([0.6, 0.8, -1.2]),
        >>>         'hess': np.array([0.9, 0.8, 0.5])
        >>>     },
        >>>     'f2':{
        >>>         'count': np.array([20, 10, 15, 15]),
        >>>         'grad': np.array([1.1, 1.5, -2.3, -0.5]),
        >>>         'hess': np.array([1.3, -0.8, 2.2, -0.5])
        >>>     }
        >>> }

        >>> is_category = {
        >>>     'f1': True, 'f2': True
        >>> }

        >>> self.exchange(data, is_category)
        """
        # check data type
        is_category = _data_type_check(data, is_category)

        # data encrypt
        data = self._feature_encrypt(data)

        # send host encrypt bin mess to guest
        self.data_channel.gather((data, is_category))

        self.logger.info('host send encrypted histogram data to guest')

        # get split result from guest
        split_info = self.data_channel.scatter()

        self.logger.info('host get split result from guest')

        return split_info


@FunctionAutoLog(__file__)
def _parse_algo_param(algo_param: Dict) -> AlgoParamParser:
    """
    Parse algorithm parameters
    Args:
        algo_param: dict, params for algo

    Returns:
        AlgoParamParser object
    ----

    **Example**:
    >>> algo_param = {
    >>>     'min_sample_leaf': 20,
    >>>     'lambda_': 0.1,
    >>>     'gain': 'gini'   # grad_hess of xgboost model
    >>> }

    >>> parse_algo_param(algo_param)
    """
    # algo param inits
    algo_param = parse_algo_param(algo_param)

    # params
    if algo_param.empty:
        raise ValueError('algo param can not be null')
    else:
        # gain information
        if not hasattr(algo_param, 'gain'):
            raise ValueError('must given the method about calculate information gain')

        # min sample leaf inits
        algo_param.min_samples_leaf = algo_param.min_samples_leaf \
            if hasattr(algo_param, 'min_samples_leaf') else HE_GB_FT_PARAM['min_samples_leaf']

        # lambda inits
        algo_param.lambda_ = algo_param.lambda_ \
            if hasattr(algo_param, 'lambda_') else HE_GB_FT_PARAM['lambda_']

    return algo_param


@FunctionAutoLog(__file__)
def _data_type_check(data: Dict, is_category: Dict) -> Dict:
    """
    This method mainly check the description of feature about continuous/category

    Args:
        data: dict, origin data
        is_category: dict, continuous/category of feature

    Returns:
        dict, fill the description of feature
    """
    if is_category is not None:
        # not None, return
        return is_category

    else:
        # data type is None, set all features are continuous
        is_category = dict()

        # generate feature message
        for key in data:
            is_category[key] = False

        return is_category
