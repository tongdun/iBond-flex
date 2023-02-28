from enum import Enum
from pydantic import BaseModel, conint, Field
from typing import List, Dict, Union, Optional, Any

from .local_mapping.map_to_int import IntegerMapConfig
from .local_mapping.onehot_map import OneHotMapConfig


class HeteroMapMethod(str, Enum):
    woe_map = 'woe_map'

class LocalMapMethod(str, Enum):
    integer_map = 'integer_map'
    onehot_map = 'onehot_map'

class WOEMethod(str, Enum):
    chi_bin = 'chi_bin'
    dt_bin = 'dt_bin'

class WOEConfig(BaseModel):
    map_to_woe: bool = True
    # method: WOEMethod = 'chi_bin'
    check_monotone: bool = True

    class Config:
        schema_extra = {
            'expose': ["check_monotone"]
        }
