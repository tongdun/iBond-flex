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

# ============================Crypto
CRYPTO_AES = 'aes'
CRYPTO_SM4 = 'sm4'

CRYPTO_SM3 = 'sm3'
CRYPTO_MD5 = 'md5'

CRYPTO_KEY_EXCHANGE = 'key_exchange'

CRYPTO_PAILLIER = 'paillier'

CRYPTO_ECC = 'secp256k1'

CRYPTO_SECRET_SHARING = 'secret_sharing'

CRYPTO_ONETIME_PAD = 'onetime_pad'

CRYPTO_OT = 'ot'

CRYPTO_SHA256 = 'sha256'
CRYPTO_SHA1 = 'sha1'

CRYPTO_HMAC_DRBG = "hmac_drbg"

CRYPTO_FF1 = 'ff1'

# crypto method needed param: key_length
CRYPTO_KEY_LENGTH = [CRYPTO_PAILLIER, CRYPTO_SM4, CRYPTO_AES, CRYPTO_ECC, CRYPTO_ONETIME_PAD]

# crypto method needed no params
CRYPTO_NONE_PARAM = [CRYPTO_MD5, CRYPTO_SM3]

# crypto method needed param: precision
CRYPTO_PRECISION = [CRYPTO_SECRET_SHARING]

# crypto method needed param: n&k
CRYPTO_NK = [CRYPTO_OT]


# ==========================Role
GUEST = 'guest'
HOST = 'host'
COORD = 'coordinator'
SERVER = 'server'
CLIENT = 'client'


# ==========================Phase
TRAIN = 'train'
PREDICTION = 'prediction'


# ==========================Computing
# Multi Loan
HE_ML = 'he_ml'
SS_ML = 'ss_ml'

# Dot mul
DOT_MUL = 'dot_mul'

# Contrib measure
CONTRIB_MEASURE = 'contrib_measure'

# SS Compute
SS_COMPUTE = 'ss_compute'

# OTP Statstics
OTP_STATS = 'otp_stats'

# SS Statsics
SS_STATS = 'ss_stats'

# ===========================prediction
# lr prediction
HE_LR_FP = 'he_lr_fp'
HE_LR_FP2 = 'he_lr_fp2'

# Factorization Machines
HE_FM_FP = 'he_fm_fp'

# tree model
HE_GB_FP = 'he_gb_fp'


# ===========================preprocessing
# Bin
HE_CHI_FB = 'he_chi_fb'
HE_DT_FB = 'he_dt_fb'

# FMC
FMC = 'fmc'
FMC2 = 'fmc2'
DEFAULT_MPC_PRECISION = 2

# Iv FFS
IV_FFS = 'iv_ffs'

# Kolmogorov-Smirnov
KS_FFS = 'ks_ffs'

# Feature Correlation
CORR_FFS = 'corr_ffs'

# parameters negotiate
OTP_PN_FL = 'otp_pn_fl'


# =============================sharing
# Invisible Inquiry
OT_INV = 'ot_inv'

# Secure Alignment
SAL = 'sal'

# Sample Filtering
BF_SF = 'bf_sf'


# =============================training
# Neural Network
OTP_NN_FT = 'otp_nn_ft'

# Linear Regression
HE_LINEAR_FT = 'he_linear_ft'

# Logistic Regression
HE_OTP_LR_FT1 = 'he_otp_lr_ft1'
HE_OTP_LR_FT2 = 'he_otp_lr_ft2'

# Secure Aggregation
OTP_SA_FT = 'otp_sa_ft'
HE_SA_FT = 'he_sa_ft'

# Tree
CS_GB_FT = 'cs_gb_ft'
HE_GB_FT = 'he_gb_ft'

# Factorization Machines
HE_FM_FT = 'he_fm_ft'


