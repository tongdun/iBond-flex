from enum import Enum
from pydantic import BaseModel, conint, Field
from typing import List, Dict, Union, Optional, Any


class HeteroBinMethod(str, Enum):
    dt_bin = 'dt_bin'
    # TODO  chi_bin

class LocalBinMethod(str, Enum):
    equifrequent_bin = 'equifrequent_bin'
    equidist_bin = 'equidist_bin'


class BinMethod(str, Enum):
    equifrequent_bin = 'equifrequent_bin'
    equidist_bin = 'equidist_bin'
    dt_bin = 'dt_bin'


class BinConfig(BaseModel):
    equal_num_bin: conint(gt=0, le=200) = 50
    max_bin_num: conint(ge=2, le=20) = 4
    min_bin_num: conint(ge=2, le=20) = 3

    class Config:
        schema_extra = {
            'expose': ["equal_num_bin", "max_bin_num", "min_bin_num"]
        }


class EquifrequentBinConfig(BaseModel):
    equal_num_bin: conint(gt=0, le=200) = 50
    max_bin_num: conint(ge=2, le=20) = 4
    min_bin_num: conint(ge=2, le=20) = 3

    class Config:
        schema_extra = {
            'expose': ["equal_num_bin"]
        }
