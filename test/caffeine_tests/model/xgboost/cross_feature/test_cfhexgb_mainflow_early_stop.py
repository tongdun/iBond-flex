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
gamma = 0
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
    'shap_finance_hetero_he_xgb_early_stop_iter_abs': case_template_no_coord(
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
                'early_stop_param': {'early_stop': True, 'early_stop_step': 'iter', 'early_stop_method': 'abs',
                                     'early_stop_eps': 0.6},
            },
            'predict_param': {
             },
            'security': security
        },
        learners = learners,
        metrics = {
            'ks': lambda x: x >= 0.0,
            'auc': lambda x: x >= 0.0
        },
        scale = False,
        timeout = 2400,
    ),
}
##############################################################################

from functools import partial
for case_name, case in test_cases.items():
    locals()[f'test_{case_prefix}{case_name}'] = partial(run_processes, {case_name: case})

if __name__ == '__main__':
    for case_name, case in test_cases.items():
        eval(f'test_{case_prefix}{case_name}()')

