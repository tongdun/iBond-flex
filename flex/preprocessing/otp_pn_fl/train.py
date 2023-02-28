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

from typing import Dict, Optional, Union, List, Set
import functools
import os
import pickle

import numpy as np

from flex.cores.base_model import BaseModel
from flex.utils import ClassMethodAutoLog


class PNBaseModel(BaseModel):
    """
        Logistic regression(No coordinate)  base model
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[Dict] = None,
                 algo_param: Optional[Dict] = None):
        """
        Protocol for parameters negotiation
        inits of federation information for communication and secure params for security calculation
        ----

        Args:
            federal_info: dict, federal info
            sec_param: dict, params for security calc
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

        >>> sec_param = None

        >>> algo_param = None

        >>> PNBaseModel(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self, federal_info=federal_info,
                           sec_param=sec_param)

        self.job_id = federal_info["session"]["job_id"]
        # inits sec review
        self.sec_review = True if os.getenv('IBOND_SECURITY_REVIEW') == 'on' else False
        if self.sec_review:
            default_review_path = os.path.join(os.getcwd(), "sec_review_output")
            self.sec_review_path = os.environ.get(
                'IBOND_SECURITY_REVIEW_PATH',
                default_review_path
            )
            self.sec_review_path = os.path.join(
                self.sec_review_path,
                self.job_id,
                'OTP_PN_FL'
            )
            os.makedirs(self.sec_review_path, exist_ok=True)

    @ClassMethodAutoLog()
    def _save_tofile(self, param, filename):
        try:
            output_file = os.path.join(self.sec_review_path, filename)
            with open(output_file, 'wb') as f:
                pickle.dump(param, f)
        except:
            self.logger.error(f'!!!Open file {output_file} failed!!!')
            raise Exception(f'Open file {output_file} failed.')

class OTPPNCoord(PNBaseModel):
    """
        Parameters negotiation, coordinate
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[Dict] = None,
                 algo_param: Optional[Dict] = None):
        PNBaseModel.__init__(self,
                             federal_info=federal_info,
                             sec_param=sec_param,
                             algo_param=algo_param)

    @ClassMethodAutoLog()
    def param_negotiate(self, param: str, tag: str = '*',
                        *args, **kwargs) -> Union[int, float]:
        """
        Used to negotiate some parameters such as the number of iterations

        Args:
            param: The statistics you need, currently supported:
                   max, min, equal, sum, average
            tag: run time name for commu

        Return:
            The statistics you need
        ----

        **Example**:
        >>> param = 'min'
        >>> OTPPNCoord.param_negotiate(param)
        """
        self.logger.info('coordinate start parameter negotiate protocol')

        # coord inits channel
        data_channel = self.commu.coord_broadcast_channel('data')

        # get msg from parties
        data_list = data_channel.gather(tag=tag)
        if self.sec_review:
            self.logger.debug(f'SEC_REVIEW | COMMU | RECEIVE-GATHER | FROM: all | TO: coord | desc: Parameters negotiation coord receive param | jobid: {self.job_id} | content: {self.job_id}_negotiation_param.pkl')
            self._save_tofile(data_list, f'{self.job_id}_negotiation_param.pkl')

        if param == 'equal':
            if len(set(data_list)) == 1:
                result = data_list[0]
            else:
                raise Exception('Data is not equal')

        elif param == 'max':
            result = np.max(data_list, 0)

        elif param == 'min':
            result = np.min(data_list, 0)

        elif param == 'sum':
            result = np.sum(data_list, 0)

        elif param == 'average':
            result = np.average(data_list, 0)

        elif param == 'union':
            data_set = [set(x) for x in data_list]
            result = list(functools.reduce(set.union, data_set))

        elif param == 'intersection':
            data_set = [set(x) for x in data_list]
            result = list(functools.reduce(set.intersection, data_set))

        else:
            raise Exception('Unknown param')

        data_channel.broadcast(result, tag=tag)
        self.logger.info('coordinate end parameter negotiate protocol')
        if self.sec_review:
            self.logger.debug(f'SEC_REVIEW | COMMU | SEND-BROADCAST | FROM: coord | TO: all | desc: Parameters negotiation coord send negotiation result | jobid: {self.job_id} | content: {self.job_id}_negotiation_result.pkl')
            self._save_tofile(result, f'{self.job_id}_negotiation_result.pkl')

        return result

    @ClassMethodAutoLog()
    def coord_param_broadcast(self, data: Union[int, float, np.ndarray],
                              tag: str = '*',
                              *args, ** kwargs) -> None:
        """
        This method mainly using for coord share param of init model
        """
        self.logger.info('coordinate start init param and share to all participant')

        # coord inits channel
        data_cp = self.commu.coord_broadcast_channel('data')

        data_cp.broadcast(data, tag=tag)
        if self.sec_review:
            self.logger.debug(f'SEC_REVIEW | COMMU | SEND-BROADCAST | FROM: coord | TO: all | desc: Parameters negotiation coord broadcast share param | jobid: {self.job_id} | content: {self.job_id}_share_param.pkl')
            self._save_tofile(data, f'{self.job_id}_share_param.pkl')


class OTPPNGuest(PNBaseModel):
    """
        Parameters negotiation, guest
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[Dict] = None,
                 algo_param: Optional[Dict] = None):
        PNBaseModel.__init__(self,
                             federal_info=federal_info,
                             sec_param=sec_param,
                             algo_param=algo_param)

    @ClassMethodAutoLog()
    def param_negotiate(self, data: Union[int, float, np.ndarray],
                        param: str, tag: str = '*',
                        *args, **kwargs) -> Union[int, float, np.ndarray]:
        """
        Used to negotiate some parameters such as the number of iterations

        Args:
            param: The statistics you need, currently supported:
                   max, min, equal, sum, average, uniopython, intersection
            data: some parameters such as the number of iterations，coord default is None
            tag: run time name for commu

        Return:
            The statistics you need
        ----

        **Example**:
        >>> data = 10
        >>> param = 'min'
        >>> OTPPNGuest.param_negotiate(param, data)
        """
        self.logger.info('guest start parameter negotiate protocol')

        # coord inits channel
        data_channel = self.commu.coord_broadcast_channel('data')

        data_channel.gather(data, tag=tag)
        if self.sec_review:
            self.logger.debug(f'SEC_REVIEW | COMMU | SEND-GATHER | FROM: guest | TO: coord | desc: Parameters negotiation guest send negotiation param | jobid: {self.job_id} | content: {self.job_id}_negotiation_param.pkl')

        result = data_channel.broadcast(tag=tag)
        self.logger.info('guest end parameter negotiate protocol')
        if self.sec_review:
            self.logger.debug(f'SEC_REVIEW | COMMU | RECEIVE-BROADCAST | FROM: coord | TO: guest | desc: Parameters negotiation guest receive negotiation result | jobid: {self.job_id} | content: {self.job_id}_negotiation_result.pkl')

        return result

    @ClassMethodAutoLog()
    def param_broadcast(self, data: Union[int, float, np.ndarray, List],
                        tag: str = '*', *args, **kwargs) -> None:
        """
        Guest broadcast data msg to host

        Args:
            data: dataset, needed transfer
            tag: run time name for commu
        """
        # channel inits
        data_gh = self.commu.guest2host_broadcast_channel('data')

        data_gh.broadcast(data, tag=tag)
        self.logger.info('guest end parameter broadcast protocol')
        if self.sec_review:
            self.logger.debug(f'SEC_REVIEW | COMMU | SEND-BROADCAST | FROM: guest | TO: all | desc: Parameters negotiation guest broadcast data msg | jobid: {self.job_id} | content: {self.job_id}_broadcast_data.pkl')
            self._save_tofile(data, f'{self.job_id}_broadcast_data.pkl')

    @ClassMethodAutoLog()
    def host_param_broadcast(self, tag: str = '*',
                             *args, **kwargs) -> List[Dict]:
        """
        Guest receive host data msg

        Args:
            tag: run time name for commu
        """
        # channel inits
        data_gh = self.commu.guest2host_broadcast_channel('data')

        data = data_gh.gather(tag=tag)
        self.logger.info('Guest complete parameter broadcast protocol')
        if self.sec_review:
            self.logger.debug(f'SEC_REVIEW | COMMU | RECEIVE-GATHER | FROM: guest | TO: host | desc: Parameters negotiation host receive data msg | jobid: {self.job_id} | content: {self.job_id}_guest2host_data.pkl')
            self._save_tofile(data, f'{self.job_id}_guest2host_data.pkl')

        return data

    @ClassMethodAutoLog()
    def coord_param_broadcast(self, tag: str = '*',
                              *args, **kwargs) -> Union[int, float, np.ndarray]:
        """
        This method mainly using for coord share param of init model
        """
        self.logger.info('coordinate start init param and share to all participant')

        # coord inits channel
        data_cp = self.commu.coord_broadcast_channel('data')

        data = data_cp.broadcast(tag=tag)
        if self.sec_review:
            self.logger.debug(f'SEC_REVIEW | COMMU | RECEIVE-BROADCAST | FROM: coord | TO: all | desc: Parameters negotiation guest receive share param | jobid: {self.job_id} | content: {self.job_id}_share_param.pkl')

        return data


class OTPPNHost(PNBaseModel):
    """
        Parameters negotiation, host
    """
    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[Dict] = None,
                 algo_param: Optional[Dict] = None):
        PNBaseModel.__init__(self,
                             federal_info=federal_info,
                             sec_param=sec_param,
                             algo_param=algo_param)

    @ClassMethodAutoLog()
    def param_negotiate(self, data: Union[int, float, np.ndarray],
                        param: str, tag: str = '*',
                        *args, **kwargs) -> Union[int, float, np.ndarray]:
        """
        Used to negotiate some parameters such as the number of iterations

        Args:
            param: The statistics you need, currently supported:
                   max, min, equal, sum, average, union, intersection
            data: some parameters such as the number of iterations，coord default is None
            tag: run time name for commu

        Return:
            The statistics you need
        ----

        **Example**:
        >>> data = 10
        >>> param = 'min'
        >>> OTPPNHost.param_negotiate(param, data)
        """
        self.logger.info('host start parameter negotiate protocol')

        # coord inits channel
        data_channel = self.commu.coord_broadcast_channel('data')

        data_channel.gather(data, tag=tag)
        if self.sec_review:
            self.logger.debug(f'SEC_REVIEW | COMMU | SEND-GATHER | FROM: host | TO: coord | desc: Parameters negotiation host send negotiation param | jobid: {self.job_id} | content: {self.job_id}_negotiation_param.pkl')

        result = data_channel.broadcast(tag=tag)
        self.logger.info('host start parameter negotiate protocol')
        if self.sec_review:
            self.logger.debug(f'SEC_REVIEW | COMMU | RECEIVE-BROADCAST | FROM: coord | TO: host | desc: Parameters negotiation host receive negotiation result | jobid: {self.job_id} | content: {self.job_id}_negotiation_result.pkl')

        return result

    @ClassMethodAutoLog()
    def param_broadcast(self, tag: str = '*',
                        *args, **kwargs) -> Union[int, float, np.ndarray, List]:
        """
        Host get broadcast data from guest

        Args:
            tag: run time name for commu

        Returns:
            data: dataset, get from host
        """
        # channel inits
        data_gh = self.commu.guest2host_broadcast_channel('data')

        data = data_gh.broadcast(tag=tag)
        self.logger.info('host end parameter broadcast protocol')
        if self.sec_review:
            self.logger.debug(f'SEC_REVIEW | COMMU | RECEIVE-BROADCAST | FROM: guest | TO: all | desc: Parameters negotiation host receive data msg | jobid: {self.job_id} | content: {self.job_id}_broadcast_data.pkl')

        return data

    @ClassMethodAutoLog()
    def host_param_broadcast(self, data: Union[Dict, List],
                             tag: str = '*', *args, **kwargs) -> None:
        """
        Guest receive host data msg

        Args:
            data: dict/list, origin data broadcast to guest
            tag: run time name for commu
        """
        # channel inits
        data_gh = self.commu.guest2host_broadcast_channel('data')

        data_gh.gather(data, tag=tag)
        self.logger.info('Host complete parameter broadcast protocol')
        if self.sec_review:
            self.logger.debug(f'SEC_REVIEW | COMMU | SEND-GATHER | FROM: guest | TO: all | desc: Parameters negotiation guest send data msg | jobid: {self.job_id} | content: {self.job_id}_guest2host_data.pkl')