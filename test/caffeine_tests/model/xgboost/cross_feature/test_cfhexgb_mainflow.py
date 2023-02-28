from flex.constants import OTP_PN_FL

from caffeine.model.xgboost.cross_feature.homomorphic_encryption.guest import HeteroXGBGuest
from caffeine.model.xgboost.cross_feature.homomorphic_encryption.host import HeteroXGBHost
from caffeine_tests.model.hetero_mainflow import case_template_no_coord, run_processes

##############################################################################
case_prefix = 'HeteroXGB_'

local_path = '' #os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
tree_nums = 3
max_depth = 5
min_samples_leaf=50
reg_lambda= 0.1
min_gain = 1e-5
gamma =0
bin_num = 10
lr = 0.1
gain = 'grad_hess'

security = {
}


learners = [
    HeteroXGBGuest,
    HeteroXGBHost
]

test_cases = {
    'shap_finance_hetero_he_xgb': case_template_no_coord(
        train_files=[
            '/mnt/nfs/datasets/shap_finance/train_guest.csv',
            '/mnt/nfs/datasets/shap_finance/train_host.csv'
        ],
        test_files=[
            '/mnt/nfs/datasets/shap_finance/test_guest.csv',
            '/mnt/nfs/datasets/shap_finance/test_host.csv'
        ],
        descs=[
            {'id_desc': ['ID'], 'y_desc': ['Label']},
            {'id_desc': ['ID']}
        ],
        meta_params = {
            'train_param': {
                'tree_nums': tree_nums,
                'max_depth': max_depth,
                'min_sample_leaf': min_samples_leaf,
                'reg_lambda': reg_lambda,
                'bin_num':bin_num,
                'min_gain': min_gain,
                'lr': lr,
                'loss_type': 'BCELoss',
                'feature_category_thres': 10,
                'gain': gain,
            },
            'predict_param': {
             },
            'security': security
        },
        learners = learners,
        metrics = {
            'ks': lambda x: x >= 0.4,
            'auc': lambda x: x >= 0.7
        },
        scale = False,
        timeout = 2400,
    ),
    'shap_finance_hetero_he_xgb_no_val': case_template_no_coord(
        train_files=[
            '/mnt/nfs/datasets/shap_finance/train_guest.csv',
            '/mnt/nfs/datasets/shap_finance/train_host.csv'
        ],
        test_files=[
        ],
        descs=[
            {'id_desc': ['ID'], 'y_desc': ['Label']},
            {'id_desc': ['ID']}
        ],
        meta_params = {
            'train_param': {
                'tree_nums': 2,
                'max_depth': 3,
                'min_sample_leaf': min_samples_leaf,
                'reg_lambda': reg_lambda,
                'bin_num':10,
                'min_gain':min_gain,
                'lr': 0.1,
                'loss_type': 'BCELoss',
                'feature_category_thres': 10,
            },
            'predict_param': {
             },
            'security': security
        },
        learners = learners,
        scale = False,
        timeout = 1200,
        check_predict = False
    ),
    # 'lending_club_hetero_dt_xgb': case_template_no_coord(
    #     train_files=[
    #         '/mnt/nfs/datasets/lendingclub101/lendingclub_aa_train_minmax.csv',
    #         '/mnt/nfs/datasets/lendingclub101/lendingclub_bb_train_minmax.csv'
    #     ],
    #     test_files=[
    #         '/mnt/nfs/datasets/lendingclub101/lendingclub_aa_test_minmax.csv',
    #         '/mnt/nfs/datasets/lendingclub101/lendingclub_bb_test_minmax.csv'
    #     ],
    #     descs=[
    #         {'id_desc': ['ID','newID'], 'y_desc': ['Status']},
    #         {'id_desc': ['ID','newID']}
    #     ],
    #     meta_params={
    #         'train_param': {
    #             'tree_nums': 5,
    #             'max_depth': 5,
    #             'min_sample_leaf': min_samples_leaf,
    #             'reg_lambda': reg_lambda,
    #             'bin_num': 20,
    #             'min_gain':min_gain,
    #             'lr': 0.2,
    #             'loss_type': 'BCELoss',
    #             'feature_category_thres': 10,
    #         },
    #         'predict_param': {
    #         },
    #         'security': security
    #     },
    #     learners=learners,
    #     metrics={
    #         'ks': lambda x: x >= 0.0,
    #         'auc': lambda x: x >= 0.0
    #     },
    #     scale=False,
    #     timeout=120000000,
    #     check_predict=False
    # ),

}
##############################################################################

from functools import partial
for case_name, case in test_cases.items():
    locals()[f'test_{case_prefix}{case_name}'] = partial(run_processes, {case_name: case})

if __name__ == '__main__':
    for case_name, case in test_cases.items():
        eval(f'test_{case_prefix}{case_name}()')

