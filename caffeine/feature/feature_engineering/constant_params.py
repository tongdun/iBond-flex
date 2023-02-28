"""
The default params for feature filteration.

    down_feature_num -- the min number of features for each party ;
    max_num_col -- the max number of features for each party ;
    ks_truncated_num -- the max number of features for each party truncated by ks_thres ;

    threshold --  ;
    max_bin_num -- the max number of bins for binning methods (Chi/DT now) ;
    min_bin_num -- the min number of bins for binning methods (Chi/DT now) ;
    adjust_value_woe -- adjusted value for woe when bad_num or good_num of the bin is zero ;
    equal_num_bin -- the param for equvifrequent binning method ;

    ks_thres -- the threshold for ks filteration ;
    iv_thres -- the threshold for iv filteration ;
    corr_coe -- the threshold for correlation-coefficient filteration ;

    f_value_thres -- the threshold for stepwise f_value ;
"""

feature_params = {
    # commmon
    'down_feature_num': 2,
    'max_num_col': 200,
    'max_feature_num_col': 500,

    # local gbdt
    'threshold': 1e-3,
    'set_term': 5, 
    'set_ratio': 0.9,

    # binning
    'max_bin_num': 6, 
    'min_bin_num': 4,
    'equal_num_bin': 50,
    'bin_ratio': 0.05,

    # ks 
    'ks_thres': 0.01,
    'split_bin': True,
    'ks_top_k': None,

    # iv
    'iv_thres': 0.02,
    'iv_top_k': None,

    # woe
    'adjust_value_woe': 1.0,

    # stepwise
    'f_value_thres': 3.84,

    # correlation coefficient
    'corr_coe': 0.7,
    'vif_thres': 10,

    # spearman coefficient
    'spearman_coe': 0.7,

}

relief_params = {
    'num_samples': 1000,
    'num_subsets': 1500,
    'relief_method': 'mpc',
    'mpc_col_size': 500,
    'distance_metric': 'norm',
    'metric_order': 2,
    'random_seed': 911,
    'k_neighbor': 25,
    'smoothing_p': 1,
    'top_k': 500
}

TRUNC_SIZE = 500
NUM_DIGITS = 4
EPS = 1e-8
DEFAULT_MAX = 1e8
FMC_EPS = 1e-6

# multiprocess related
PARALLEL_NUM = 5
USE_MULTIPROCESS = False
