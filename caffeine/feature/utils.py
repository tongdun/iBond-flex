from typing import Dict, Union, List, Tuple
from functools import partial
from logging import getLogger

import pandas as pd
import numpy as np
from flex.constants import *

from caffeine.utils import ClassMethodAutoLog, IBondDataFrame
from .config import security_config, Relief_FMC, Stepwise_FMC


logger = getLogger("Feature_utils_")

def get_column_data(data: IBondDataFrame, col_name: str):
    return data[col_name].to_numpy().flatten()

def make_security(meta_params):
    security = meta_params.get('security_param')
    logger.info(f"meta_params {meta_params}")
    if security is None:
        sec_params = security_config
    else:
        sec_params = security
        sec_params = {
            eval(p): v for p, v in sec_params.items()
        }

    meta_params['security_param'] = sec_params
    logger.info(f"meta_params {meta_params}")

def make_pipeline(meta_params):
    pipeline = []
    configs = meta_params['train_param']['configs']
    logger.info(f'configs : {configs}')
    pipe = meta_params['train_param']['pipeline']
    if pipe is not None:
        for s in pipe:
            s_config = configs.get(s, {})
            pipeline.append([s, s_config])

    meta_params['train_param'].pop('configs')
    meta_params['train_param']['pipeline'] = pipeline
    make_security(meta_params)


