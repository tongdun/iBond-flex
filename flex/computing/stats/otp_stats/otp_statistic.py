"""OTP_STATS Protocol
"""
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
import secrets
import numpy as np

from flex.cores.base_model import BaseModel, send_pubkey, get_pubkey
from flex.cores.check import CheckMixin
from flex.utils import ClassMethodAutoLog

from typing import List, Dict, Optional


class OTPStatisticGuest(BaseModel):
    """
    in federation count calc
        n1 + n2
    desc: n1 not null numbers of feature(same as n2)
    """
    @property
    def num_of_guests(self) -> int:
        return len(self.federal_info.guest)

    @ClassMethodAutoLog()
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        OTP_STATS Protocol guest init

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

        >>> sec_param = [['onetime_pad', {'key_length': 512}], ]
        >>> algo_param = { }
        >>> protocol = OTPStatisticGuest(federal_info, sec_param, algo_param)
        """

        BaseModel.__init__(self,
                           federal_info=federal_info,
                           sec_param=sec_param)

        # inits encrypt
        self._init_encrypt(share_party=self.commu.guest_host,
                           local_party=self.commu.local_id)

        self.check = CheckMixin
        self.data_sta = dict()
        self.broadcast_chan = self.commu.coord_broadcast_channel('guests_coord')
        self.check = CheckMixin

    def exchange(self,
                 data: np.ndarray,
                 stats: List = ['std'],
                 *args, **kwargs) -> Dict:
        """
        exchange: computing stats

        Args:
            data: numpy.ndarray, local data
            stats: list, stats method to be calc

        Returns:
            data_sta: Dict, result of the stats
        ----

        **Example**:
        >>> data = np.random.rand(100000)
        >>> protocol.exchange(data)
        """
        self.check.array_type_check(data)

        if 'std' in stats:
            self.calc_count(data)
            self.calc_mean(data)
            self.calc_std(data)
        elif 'mean' in stats:
            self.calc_count(data)
            self.calc_mean(data)
        else:
            self.calc_count(data)

        return self.data_sta

    def calc_count(self, data: np.ndarray) -> None:
        """
        computing count stats

        Args:
            data: numpy.ndarray, local data
        ----

        **Example**:
        >>> data = np.random.rand(100000)
        >>> protocol.calc_count(data)
        """
        # send data columns
        data_count = data.size
        self.broadcast_chan.gather(data_count, tag='stats_count_data_shape')

        # get some of data shape
        sum_data_count = self.broadcast_chan.broadcast(tag='stats_count_sum_data_shape')

        # calc not null values of each columns
        not_null_cal = data_count - np.isnan(data).sum()
        enc_local_null_cal = self.pf_ecc.encrypt(np.asarray([not_null_cal]), alpha=1)

        # send encode null calc value to arbiter
        self.broadcast_chan.gather(enc_local_null_cal, tag='stats_count_not_null_local')

        # get global mess of null calc value from guest
        enc_global_null_cal = self.broadcast_chan.broadcast(tag='stats_count_enc_not_null_col_global')

        global_null_cal = self.pf_ecc.decrypt(enc_global_null_cal, alpha=self.num_of_guests)[0]

        self.data_sta['local_count'] = int(data_count)
        self.data_sta['count'] = int(sum_data_count)
        self.data_sta['local_not_null_count'] = int(not_null_cal)
        self.data_sta['not_null_count'] = int(global_null_cal)

    def calc_mean(self, data: np.ndarray) -> None:
        """
        computing mean stats

        Args:
            data: numpy.ndarray, local data
        ----

        **Example**:
        >>> data = np.random.rand(100000)
        >>> protocol.calc_mean(data)
        """
        # requires: calc_count
        local_not_null_count = self.data_sta['local_not_null_count']
        self.data_sta['local_mean'] = float(np.nansum(data) / local_not_null_count)

        normal_mean = self.data_sta['local_mean'] * local_not_null_count / self.data_sta['not_null_count']
        # normal_mean = np.asarray([normal_mean])

        # generate random data, encode mean value
        enc_normal_mean = self.pf_ecc.encrypt(np.asarray([normal_mean], dtype=np.float32), alpha=1)

        # send encode null calc value to arbiter
        self.broadcast_chan.gather(enc_normal_mean, tag='stats_mean_local_mean')

        # get global mess of mean value from arbiter
        enc_global_mean = self.broadcast_chan.broadcast(tag='stats_mean_global_mean')

        # decode global mess of mean value
        global_mean = self.pf_ecc.decrypt(enc_global_mean, alpha=self.num_of_guests)[0]
        # global_mean = enc_global_mean
        self.data_sta['mean'] = float(global_mean)

    def calc_std(self, data: np.ndarray) -> None:
        """
        computing std stats

        Args:
            data: numpy.ndarray, local data
        ----

        **Example**:
        >>> data = np.random.rand(100000)
        >>> protocol.calc_std(data)
        """
        # requires: mean, count
        # calc local std value in normalization method
        normal_std = np.nansum((data - self.data_sta['mean'])**2) / self.data_sta['not_null_count']

        # generate random data, encode std value
        enc_normal_std = self.pf_ecc.encrypt(np.asarray([normal_std], dtype=np.float32), alpha=1)
        # send encode std value to arbiter
        self.broadcast_chan.gather(enc_normal_std, tag='stats_std_local_std')

        # get global mess of std value from arbiter
        enc_global_std = self.broadcast_chan.broadcast(tag='stats_std_global_std')

        # decode global mess of std value
        global_std = self.pf_ecc.decrypt(enc_global_std, alpha=self.num_of_guests)[0]
        global_std = np.sqrt(global_std)
        self.data_sta['local_std'] = float(np.nanstd(data))
        self.data_sta['std'] = float(global_std)


class OTPStatisticCoord(BaseModel):
    """
    in federation count calc
        n1 + n2
    desc: n1 not null numbers of feature(same as n2)
    """
    def __init__(self,
                 federal_info: Dict,
                 sec_param: Optional[List] = None,
                 algo_param: Optional[Dict] = None):
        """
        OTP_STATS Protocol coordinator init

        Args:
            federal_info: dict, federal info
            sec_param: list, params for security calc
            algo_param: dict, params for algo
        ----

        **Example**:
        >>> federal_info = {
        >>>    "server": "localhost:6001",
        >>>    "session": {
        >>>        "role": "coordinator",
        >>>        "local_id": "zhibang-d-014012",
        >>>        "job_id": 'test_job',
        >>>    },
        >>>    "federation": {
        >>>        "host": ["zhibang-d-014010"],
        >>>        "guest": ["zhibang-d-014011"],
        >>>        "coordinator": ["zhibang-d-014012"]
        >>>    }
        >>> }

        >>> sec_param = [['onetime_pad', {'key_length': 512}], ]
        >>> algo_param = { }
        >>> protocol = OTPStatisticGuest(federal_info, sec_param, algo_param)
        """
        BaseModel.__init__(self,
                           federal_info=federal_info,
                           sec_param=sec_param)

        # inits encrypt
        # self._init_encrypt()
        self.check = CheckMixin
        self.broadcast_chan = self.commu.coord_broadcast_channel('guests_coord')

    def exchange(self, stats: Optional[List] = ['std'],
                 *args, **kwargs) -> None:
        """
        exchange: computing stats

        Args:
            stats: list, stats method to be calc
        
        Returns:
            None
        ----

        **Example**:
        >>> data = np.random.rand(100000)
        >>> protocol.exchange(data)
        """
        if 'std' in stats:
            self.calc_count()
            self.calc_mean()
            self.calc_std()
        elif 'mean' in stats:
            self.calc_count()
            self.calc_mean()
        else:
            self.calc_count()

    def calc_count(self) -> None:
        """
        computing count stats
        ----

        **Example**:
        >>> protocol.calc_count()
        """
        # get row of all
        # calc sum shape of all client
        sum_data_count = sum(self.broadcast_chan.gather(tag='stats_count_data_shape'))
        # send sum shape to each client
        self.broadcast_chan.broadcast(sum_data_count, tag='stats_count_sum_data_shape')

        # get encode null calc value to arbiter
        # calc global col count
        enc_global_null_cal = sum(self.broadcast_chan.gather(tag='stats_count_not_null_local'))

        # save sum value to guest
        self.broadcast_chan.broadcast(enc_global_null_cal, tag='stats_count_enc_not_null_col_global')

    def calc_mean(self):
        """
        computing mean stats
        ----

        **Example**:
        >>> protocol.calc_mean()
        """
        # get encode mean value from each server
        enc_local_mean = self.broadcast_chan.gather(tag='stats_mean_local_mean')

        # sum mean value
        enc_global_mean = sum(enc_local_mean)

        # send sum value to guest
        self.broadcast_chan.broadcast(enc_global_mean, tag='stats_mean_global_mean')

    def calc_std(self):
        """
        computing std stats
        ----

        **Example**:
        >>> protocol.calc_std()
        """
        # get encode std value from each server
        enc_local_std = self.broadcast_chan.gather(tag='stats_std_local_std')

        # sum std value
        # enc_global_std = reduce(lambda x, y: x + y, en_local_std)
        enc_global_std = sum(enc_local_std)

        # send sum std value to guest
        self.broadcast_chan.broadcast(enc_global_std, tag='stats_std_global_std')
