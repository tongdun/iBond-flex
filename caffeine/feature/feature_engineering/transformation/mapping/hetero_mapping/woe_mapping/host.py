from typing import Dict, Optional, Tuple, List, Tuple, Union

import numpy as np
from flex.constants import OTP_PN_FL, IV_FFS

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame, Context
from caffeine.feature.feature_engineering.ibond_feature_df import FeatureDataFrame
from caffeine.feature.utils import get_column_data
from .common import HeteroMapToWOEBase

class HeteroMapToWOEHost(HeteroMapToWOEBase):
    @ClassMethodAutoLog()
    def __init__(self, meta_params: Dict, config: Dict, context, model_info: Optional[Dict]=None):
        """
        Init dtbin base.

        Args: 
            meta_params: dict, containing meta params.
            config: dict, configs for dtbinning.
            model_info: dict, optional and contains model infos for test (transformation).
            e.g.
                meta_params = {

                },
                config = {
                    'equal_num_bin': 50,
                    'map_to_woe': True,
                    'map_to_int': False
                },
                model_info = {
                    'x1': {
                        'split_points': [1, 2, 3],
                        'woe': [1.2, 3.3, 4.8, 5.6],
                        'iv': 0.34,
                        'is_category': False,
                        'is_fillna': True
                    },
                    'x2': {
                        ...
                    },
                    ...
                }
        """
        super().__init__(meta_params, config, context, model_info)
        if model_info is None and self.use_multiprocess is False:
            self._init_method(IV_FFS)

    @ClassMethodAutoLog()
    def fit(self, data: IBondDataFrame, data_attributes: FeatureDataFrame) -> \
            Tuple[IBondDataFrame, FeatureDataFrame]:          
        fea_num = data_attributes.shape
        if self.use_multiprocess:
            max_fea_num = self.p_worker.run_otp_pn_tl_protocol(fea_num, "max", "bin-fea-num", self.federal_info, self._meta_params['security_param'])
        else:
            max_fea_num = self._protocols[OTP_PN_FL].param_negotiate(data=fea_num, param='max', tag='bin-fea-num')
        self.logger.info(f'max_fea_num {max_fea_num}')

        self._model_info = data_attributes.bin_info

        if self.use_multiprocess:
            fea_cols = data_attributes.names
            fea_data = data[fea_cols].to_numpy()
            is_category = data_attributes.select('is_category').tolist()
            is_fillna = data_attributes.select('is_fillna').tolist()
            split_points = [self._model_info[col]['split_points'] for col in fea_cols]
            model_info, ivs = self.p_worker.run_host(
                                                    max_fea_num, 
                                                    fea_data, 
                                                    fea_cols,
                                                    is_category,
                                                    is_fillna,
                                                    split_points,
                                                    self.federal_info,
                                                    self._meta_params['security_param'],
                                                    self.algo_param,
                                                    self.Label
                                                )
            for col, v in model_info.items():
                self._model_info[col].update(v)

        else:
            if self.Label is not None:
                en_label = self.Label
                self.logger.info(f'Saved label loaded.')
            else:
                en_label = self._protocols[IV_FFS].pre_exchange()
                self.logger.info(f'Exchange label done.')

            ivs = []
            for i in range(max_fea_num):
                tag = '_'.join([str(self.job_id), str(i)])

                if i >= fea_num:
                    self._protocols[IV_FFS].exchange(None, en_label, None, None, None, tag=tag)                                         
                    continue

                col, is_category, is_fillna = data_attributes[i]
                data_col = get_column_data(data, col)
                split_points = self._model_info[col]['split_points']
                woes, iv, _ = self._protocols[IV_FFS].exchange(data_col, en_label,
                                                            is_category, bool(1-is_fillna),
                                                            split_points, tag=tag)

                is_mono = self.woe_mono(woes, is_category, is_fillna)
                if not is_mono:
                    iv = 0.0

                self._model_info[col].update({'woe': woes, 'iv': iv})
                ivs.append(iv)
                self.logger.info(f'host col {col} woes {woes}')
                self.logger.info(f'Host {self.local_id} has finished {i+1} features.')

        # woe mapping
        data_attributes['iv'] = ivs
        data_attributes.update_bin(self._model_info)

        data = self.fit_transform(data)
        self.logger.info(f"data_attributes {data_attributes.to_pandas()}")
        self.logger.info(f'host {self.local_id} model {self._model_info}')
        return data, data_attributes
        