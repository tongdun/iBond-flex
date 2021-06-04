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

from typing import Dict, Optional, List

from flex.constants import *
from flex.utils import ClassMethodAutoLog


@ClassMethodAutoLog()
def make_protocol(protocol_name: str,
                  federal_info: Dict,
                  sec_param: List = None,
                  algo_param: Optional[Dict] = None):
    session = federal_info.get("session")
    role = session.get("role")
    identity = session.get("identity")

    # ================  computing  ===================
    # computing: multi_loan
    if protocol_name == HE_ML:
        from .computing.multi_loan.he_ml.compute import HEMLCoord, HEMLGuest, HEMLHost

        all_class = {"guest": HEMLGuest, "host": HEMLHost, 'coordinator': HEMLCoord}

    elif protocol_name == SS_ML:
        from .computing.multi_loan.ss_ml.compute import SSMLCoord, SSMLGuest, SSMLHost

        all_class = {"guest": SSMLGuest, "host": SSMLHost, 'coordinator': SSMLCoord}

    elif protocol_name == SS_COMPUTE:
        from .computing.ss_computing.compute_host import SSComputeHost
        from .computing.ss_computing.compute_guest import SSComputeGuest
        from .computing.ss_computing.compute_coord import SSComputeCoord

        all_class = {"guest": SSComputeGuest, "coordinator": SSComputeCoord, "host": SSComputeHost}

    elif protocol_name == OTP_STATS:
        from .computing.stats.otp_stats.otp_statistic import OTPStatisticGuest
        from .computing.stats.otp_stats.otp_statistic import OTPStatisticCoord

        all_class = {"guest": OTPStatisticGuest, "coordinator": OTPStatisticCoord}

    # ================  prediction  ===================
    # prediction: logistic_regression
    elif protocol_name == HE_LR_FP:
        from .prediction.logistic_regression.he_lr_fp.predict import HELRFPCoord, HELRFPGuest, HELRFPHost

        all_class = {"guest": HELRFPGuest, "host": HELRFPHost, 'coordinator': HELRFPCoord}

    elif protocol_name == HE_LR_FP2:
        from flex.prediction.logistic_regression.he_lr_fp2.predict import HELRFPGuest, HELRFPHost

        all_class = {"guest": HELRFPGuest, "host": HELRFPHost}

    # factorization machines
    elif protocol_name == HE_FM_FP:
        from flex.prediction.factorization_machines.he_fm_fp.predict import HEFMFPHost, HEFMFPGuest

        all_class = {"guest": HEFMFPGuest, "host": HEFMFPHost}

    # ================  preprocessing  ===================
    # binning
    elif protocol_name == HE_DT_FB:
        from .preprocessing.binning.he_dt_fb.guest import HEDTFBGuest
        from .preprocessing.binning.he_dt_fb.host import HEDTFBHost

        all_class = {"guest": HEDTFBGuest, "host": HEDTFBHost}

    # feature_selection
    # IV
    elif protocol_name == IV_FFS:
        from .preprocessing.feature_selection.iv_ffs.guest import IVFFSGuest
        from .preprocessing.feature_selection.iv_ffs.host import IVFFSHost

        all_class = {"guest": IVFFSGuest, "host": IVFFSHost}

    # ================  sharing  ===================
    # invisible_inquiry
    elif protocol_name == OT_INV:
        from .sharing.invisible_inquiry.ot_inv.inquiry import OTINVServer, OTINVClient

        all_class = {"server": OTINVServer, "client": OTINVClient}

        my_class = all_class[identity]
        return my_class(federal_info, sec_param, algo_param)

    # sample_alignment
    elif protocol_name == BF_SF:
        from .sharing.sample_alignment.sample_filtering.filter import BFSFCoord, BFSFParty

        all_class = {"guest": BFSFParty, "host": BFSFParty, 'coordinator': BFSFCoord}

    elif protocol_name == SAL:
        from .sharing.sample_alignment.secure_alignment.align import SALCoord, SALParty

        all_class = {"guest": SALParty, "host": SALParty, 'coordinator': SALCoord}

    # ================  training  ===================
    # linear_regression
    elif protocol_name == HE_LINEAR_FT:
        from .training.linear_regression.he_linear_ft.train import HELinearFTCoord, HELinearFTGuest, \
            HELinearFTHost

        all_class = {"guest": HELinearFTGuest, "host": HELinearFTHost,
                     'coordinator': HELinearFTCoord}

    # logistic_regression
    elif protocol_name == HE_OTP_LR_FT1:
        from .training.logistic_regression.he_otp_lr_ft1.train import HEOTPLR1Guest, HEOTPLR1Host

        all_class = {"guest": HEOTPLR1Guest, "host": HEOTPLR1Host}

    elif protocol_name == HE_OTP_LR_FT2:
        from .training.logistic_regression.he_otp_lr_ft2.train import HEOTPLRCoord, HEOTPLRGuest, HEOTPLRHost

        all_class = {"guest": HEOTPLRGuest, "host": HEOTPLRHost,
                     'coordinator': HEOTPLRCoord}

    # neural_network
    elif protocol_name == OTP_NN_FT:
        from .training.neural_network.otp_nn_ft.train import OTPNNFTGuest, OTPNNFTHost

        all_class = {"guest": OTPNNFTGuest, "host": OTPNNFTHost}

    elif protocol_name == OTP_SA_FT:
        from .training.secure_aggregation.otp_sa_ft.train import OTPSAFTCoord, OTPSAFTParty

        all_class = {"guest": OTPSAFTParty, "coordinator": OTPSAFTCoord, "host": OTPSAFTParty}

    # factorization machines
    elif protocol_name == HE_FM_FT:
        from flex.training.factorization_machines.he_fm_ft.train import HEFMFTGuest, HEFMFTHost

        all_class = {"guest": HEFMFTGuest, "host": HEFMFTHost}

    else:
        raise NotImplementedError(f"Protocol {protocol_name} is not implemented.")
    my_class = all_class[role]
    return my_class(federal_info, sec_param, algo_param)
