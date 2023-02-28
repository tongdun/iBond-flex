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

import re
from enum import Enum
from typing import Dict, List, Union, Optional

import numpy as np
from pydantic import BaseModel, Field, validator, ValidationError, PositiveInt, NonNegativeInt


#########################################################


def hump2underline(hunp_str):
    '''
    驼峰形式字符串转成下划线形式
    :param hunp_str: 驼峰形式字符串
    :return: 字母全小写的下划线形式字符串
    '''
    # 匹配正则，匹配小写字母和大写字母的分界位置
    p = re.compile(r'([a-z]|\d)([A-Z])')
    # 这里第二个参数使用了正则分组的后向引用
    sub = re.sub(p, r'\1_\2', hunp_str).lower()
    return sub


def underline2hump(underline_str):
    '''
    下划线形式字符串转成驼峰形式
    :param underline_str: 下划线形式字符串
    :return: 驼峰形式字符串
    '''
    # 这里re.sub()函数第二个替换参数用到了一个匿名回调函数，回调函数的参数x为一个匹配对象，返回值为一个处理后的字符串
    # sub = re.sub(r'(_\w)',lambda x:x.group(1)[1].upper(),underline_str)
    arr = filter(None, underline_str.lower().split('_'))
    res = ''
    j = 0
    for i in arr:
        res = res + i[0].upper() + i[1:]
        j += 1
    return res


######################################### Code in FLEX ##########################################
security_method_list = ['paillier', 'onetime_pad', 'secret_sharing', 'okamoto_uchiyama', 'ckks', 'fake']
# Tips：以上安全方法的名称均需要以下划线命名法进行命名
protocol_security_method_dict = {
    'he_otp_lr_ft2': [security_method_list[0], security_method_list[3], security_method_list[4], security_method_list[5]],
    'he_otp_sr_ft1': [security_method_list[0], security_method_list[4]],
    'he_lr_fp': [security_method_list[0], security_method_list[5]],
    'sal': [],
    'otp_pn_fl': [security_method_list[0], security_method_list[1]],
    'iv_ffs': [security_method_list[0]],
    'he_dt_fb': [security_method_list[0]],
}  # flex需要将所有能够支持的协议和方法拓展在这里吧


# Tips: 以下所有的pydantic校验类名称需要以驼峰命名法进行命名
############################## Fake BaseModel ######################################
class Fake(BaseModel):
    pass


############################## CKKS BaseModel ######################################
class Ckks(BaseModel):
    key_length: NonNegativeInt = 2048


############################## OT BaseModel ######################################
class ObliviousTransfer(BaseModel):
    k: int
    n: int
    remote_id: str


############################## OU BaseModel ######################################
class OkamotoUchiyama(BaseModel):
    key_size: PositiveInt = 256


############################## FPE BaseModel ######################################
class Fpe(BaseModel):
    key: Union[int, bytes]
    n: int
    t: Union[int, bytes] = b''
    method: str = 'ff1'
    encrypt_algo: str = 'aes'


############################## BloomFilter BaseModel ######################################
class BloomFilter(BaseModel):
    log2_bitlength: int = 31
    src_filter: Optional[np.ndarray] = None

    class Config:
        arbitrary_types_allowed = True


############################## Affine BaseModel ######################################
class AffineKeySize(int, Enum):
    one = 1024
    two = 512


class Affine(BaseModel):
    key_size: AffineKeySize = 1024
    key_round: PositiveInt = 5
    encode_precision: int = 2 ** 100  # 要不要保留这个入口？

    class Config:
        use_enum_values = True


############################## One Time Pad BaseModel ######################################

class OneTimePadKeyLength(int, Enum):
    one = 512
    two = 256


class OnetimePad(BaseModel):
    key_length: OneTimePadKeyLength = Field('512')

    class Config:
        use_enum_values = True


############################## Paillier BaseModel ######################################

class PaillierKeyLength(int, Enum):
    one = 512
    two = 1024
    three = 2048


class Paillier(BaseModel):
    key_length: PaillierKeyLength = Field('1024')

    class Config:
        use_enum_values = True


############################## Secret Sharing BaseModel ######################################

class SecretSharingKeyLength(int, Enum):
    one = 64
    two = 128


class SecretSharing(BaseModel):  # todo 为什么这个有两个版本？需要确认。
    precision: NonNegativeInt = 4
    shift: NonNegativeInt = 16
    length: SecretSharingKeyLength = 64
    p_value: PositiveInt = 67

    class Config:
        use_enum_values = True


################################# Validator function for BaseModel in FLEX ###################################

def valid_security_param(params, protocol_security_method):
    output = []
    if not protocol_security_method:
        if not params:
            return None
        else:
            raise Exception(f'Expected security parmas is None, get {params}')
    for param in params:
        if isinstance(param, list):
            if len(param) == 2:
                tmp_out = []
                assert param[0] in protocol_security_method
                tmp_out.append(param[0])
                tmp_param = eval(underline2hump(param[0])).parse_obj(param[1])
                tmp_out.append(tmp_param.dict())
            else:
                raise ValidationError(f'Length of {param} should be 2, get {len(param)}')
        elif isinstance(param, dict):
            tmp_out = []
            assert param['method'] in protocol_security_method
            tmp_out.append(param['method'])
            tmp_param = eval(underline2hump(param['method'])).parse_obj(param['params'])
            tmp_out.append(tmp_param.dict())
        else:
            raise ValidationError(f'Unsupported type {type(param)}')
        output.append(tmp_out)
    return output


def valid_sal(params: List) -> List:
    output = valid_security_param(params, protocol_security_method_dict['sal'])
    return output



def valid_otp_pn_fl(params: List) -> List:
    output = valid_security_param(params, protocol_security_method_dict['otp_pn_fl'])
    return output

def valid_he_dt_fb(params: List) -> List:
    output = valid_security_param(params, protocol_security_method_dict['he_dt_fb'])
    return output

# def valid_otp_sa_ft(params: List) -> List:
#     output = valid_security_param(params, protocol_security_method_dict['otp_sa_ft'])
#     return output

def valid_he_otp_lr_ft2(params: List) -> List:
    output = valid_security_param(params, protocol_security_method_dict['he_otp_lr_ft2'])
    return output


def valid_he_lr_fp(params: List) -> List:
    output = valid_security_param(params, protocol_security_method_dict['he_lr_fp'])
    return output


def valid_he_gb_ft(params: List) -> List:
    output = valid_security_param(params, protocol_security_method_dict['he_gb_ft'])
    return output

# feature selection for iv
def valid_iv_ffs(params: List) -> List:
    output = valid_security_param(params, protocol_security_method_dict['iv_ffs'])
    return output

##################################### Code Template in Caffeine ##########################################

# class Security(BaseModel):
#     OTP_PN_FL: Optional[List] = [["onetime_pad", {"key_length": 512}]]
#     HE_GB_FT: Optional[List] = [["paillier", {"key_length": 1024}]]
#     SAL: Optional[List] = None
#
#     _otp_pn_fl = validator('OTP_PN_FL', allow_reuse=True)(valid_otp_pn_fl)
#     _ht_gb_ft = validator('HE_GB_FT', allow_reuse=True)(valid_he_gb_ft)
#     _sal = validator('SAL', allow_reuse=True)(valid_sal)
#
#     class Config:  # Tips: 如果不需要限制输入参数中不带多余参数，这部分可以去掉。
#         extra = 'forbid'
#
#
# class Params(BaseModel):
#     security_param: Security = Security()


##################################################################################################
#
#
# if __name__ == '__main__':
#     ceshi = {'security_param': {'OTP_PN_FL': [{'method': 'onetime_pad', 'params': {'key_length': 512}}],
#                                 'HE_GB_FT': [["paillier", {"key_length": 1024}]],
#                                 'SAL': None}}
#     # ceshi = {'security_param': {'OTP_PN_FL': [{'method': 'onetime_pad', 'params': {'key_length': 512}}],
#     #                             'HE_GB_FT': [{'method': 'paillier', 'params': {"key_length": 1024}}]}}
#     c = Params.parse_obj(ceshi)
#     a = np.ones([1, 2])
#     b = {'log2_bitlength': 12, 'src_filter': a}
#     d = BloomFilter.parse_obj(b)
#     print(c)
