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
from flex.constants import *


def make_protocol(protocol_name: str, federal_info: dict, sec_param: dict, algo_param: dict = None):
    session = federal_info.get("session")
    role = session.get("role")
    identity = session.get("identity")

    # federated_prediction: logistic_regression
    if protocol_name == HE_LR_FP:
        from .federated_prediction.logistic_regression.he_lr_fp.predict import HELRFPCoord, HELRFPGuest, HELRFPHost
        if role == COORD:
            return HELRFPCoord(federal_info, sec_param)
        if role == GUEST:
            return HELRFPGuest(federal_info, sec_param)
        if role == HOST:
            return HELRFPHost(federal_info, sec_param)
        raise ValueError(f"Role {role} is not supported in {protocol_name} protocol.")

    # federated_preprocessing: federated_feature_selection
    if protocol_name == IV_FFS:
        from .federated_preprocessing.federated_feature_selection.iv_ffs.compute import IVFFSGuest, IVFFSHost
        if role == GUEST:
            return IVFFSGuest(federal_info, sec_param, algo_param)
        if role == HOST:
            return IVFFSHost(federal_info, sec_param, algo_param)
        raise ValueError(f"Role {role} is not supported in {protocol_name} protocol.")

    # federated_sharing: invisible_inquiry
    if protocol_name == OT_INV:
        from .federated_sharing.invisible_inquiry.ot_inv.inquiry import OTINVServer, OTINVClient
        if identity == SERVER:
            return OTINVServer(federal_info, sec_param, algo_param)
        if identity == CLIENT:
            return OTINVClient(federal_info, sec_param, algo_param)
        raise ValueError(f"Role {role} is not supported in {protocol_name} protocol.")

    # federated_sharing: sample_alignment
    if protocol_name == SAL:
        from .federated_sharing.sample_alignment.secure_alignment.align import SALCoord, SALGuest, SALHost
        if role == COORD:
            return SALCoord(federal_info, sec_param, algo_param)
        if role == GUEST:
            return SALGuest(federal_info, sec_param, algo_param)
        if role == HOST:
            return SALHost(federal_info, sec_param, algo_param)
        raise ValueError(f"Role {role} is not supported in {protocol_name} protocol.")

    # federated_training: linear_regression
    if protocol_name == HE_LINEAR_FT:
        from .federated_training.linear_regression.he_linear_ft.train import HELinearFTCoord, HELinearFTGuest, \
            HELinearFTHost
        if role == COORD:
            return HELinearFTCoord(federal_info, sec_param)
        if role == GUEST:
            return HELinearFTGuest(federal_info, sec_param)
        if role == HOST:
            return HELinearFTHost(federal_info, sec_param)
        raise ValueError(f"Role {role} is not supported in {protocol_name} protocol.")

    # federated_training: logistic_regression
    if protocol_name == HE_OTP_LR_FT1:
        from .federated_training.logistic_regression.he_otp_lr_ft1.train import HEOTPLR1Guest, HEOTPLR1Host
        if role == GUEST:
            return HEOTPLR1Guest(federal_info, sec_param)
        if role == HOST:
            return HEOTPLR1Host(federal_info, sec_param)
        raise ValueError(f"Role {role} is not supported in {protocol_name} protocol.")

    if protocol_name == HE_OTP_LR_FT2:
        from .federated_training.logistic_regression.he_otp_lr_ft2.train import HEOTPLRCoord, HEOTPLRGuest, HEOTPLRHost
        if role == COORD:
            return HEOTPLRCoord(federal_info, sec_param)
        if role == GUEST:
            return HEOTPLRGuest(federal_info, sec_param)
        if role == HOST:
            return HEOTPLRHost(federal_info, sec_param)
        raise ValueError(f"Role {role} is not supported in {protocol_name} protocol.")

    # federated_training: secure_aggregation
    if protocol_name == HE_SA_FT:
        from .federated_training.secure_aggregation.he_sa_ft.train import HESAFTCoord, HESAFTGuest, HESAFTHost
        if role == COORD:
            return HESAFTCoord(federal_info, sec_param)
        if role == GUEST:
            return HESAFTGuest(federal_info, sec_param)
        if role == HOST:
            return HESAFTHost(federal_info, sec_param)
        raise ValueError(f"Role {role} is not supported in {protocol_name} protocol.")

    if protocol_name == OTP_SA_FT:
        from .federated_training.secure_aggregation.otp_sa_ft.train import OTPSAFTCoord, OTPSAFTGuest, OTPSAFTHost
        if role == COORD:
            return OTPSAFTCoord(federal_info, sec_param)
        if role == GUEST:
            return OTPSAFTGuest(federal_info, sec_param)
        if role == HOST:
            return OTPSAFTHost(federal_info, sec_param)
        raise ValueError(f"Role {role} is not supported in {protocol_name} protocol.")

    raise NotImplementedError(f"Protocol {protocol_name} is not implemented.")
