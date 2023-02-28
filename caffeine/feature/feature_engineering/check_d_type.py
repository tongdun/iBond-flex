from typing import Optional, Dict
from logging import getLogger
from caffeine.feature.mixins import FLEXUser, IonicUser
from caffeine.feature.parallel_module import ParallelWorkerBase

logger = getLogger('Check_d_type_')

class CheckDTypeCommon(IonicUser):
    def __init__(self, meta_params: Dict):
        self._meta_params = meta_params
        self.d_data = self._meta_params.get('d_data')
        self.process_method = meta_params.get('process_method')
        self.p_worker = ParallelWorkerBase(1)

    def check_d_type(self, n: Optional[int]=None):
        pass


class CheckDTypeCoord(CheckDTypeCommon):
    def __init__(self, meta_params: Dict):
        super().__init__(meta_params)

    def check_d_type(self):
        if self.process_method == 'hetero':
            if self.d_data is None:
                self.d_data = self.p_worker.run_coord_broadcast_channel(self, 'd_data', 'broadcast')
            self._meta_params['d_data'] = self.d_data
            logger.info(f"******** coord d_data {self.d_data}")
        return self.d_data


class CheckDTypeHost(CheckDTypeCommon):
    def __init__(self, meta_params: Dict):
        super().__init__(meta_params)

    def check_d_type(self):
        if self.process_method == 'hetero':
            if self.d_data is None:
                self.d_data = self.p_worker.run_host_broadcast_channel(self, 'd_data', 'broadcast')
            self._meta_params['d_data'] = self.d_data
            logger.info(f"******** host d_data {self.d_data}")
        return self.d_data


class CheckDTypeGuest(CheckDTypeCommon):
    def __init__(self, meta_params: Dict):
        super().__init__(meta_params)

    def check_d_type(self, n: int):
        if self.process_method == 'hetero':
            if self.d_data is None:
                if n == 0:
                    self.d_data = True
                else:
                    self.d_data = False
                self.p_worker.run_guest_broadcast_channel(self, 'd_data', 'broadcast', self.d_data)
            self._meta_params['d_data'] = self.d_data
        return self.d_data
