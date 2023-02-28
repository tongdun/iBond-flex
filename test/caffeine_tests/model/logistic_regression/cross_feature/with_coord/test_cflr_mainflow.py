from functools import partial

from flex.constants import HE_OTP_LR_FT2, HE_LR_FP

from caffeine.model.logistic_regression.cross_feature.with_coord.coordinator import HeteroLogisticRegressionCoord
from caffeine.model.logistic_regression.cross_feature.with_coord.guest import HeteroLogisticRegressionGuest
from caffeine.model.logistic_regression.cross_feature.with_coord.host import HeteroLogisticRegressionHost
from caffeine_tests.model.hetero_mainflow import case_template, run_processes

##############################################################################
case_prefix = 'hetero_lr_'

security =  {
    'HE_OTP_LR_FT2': [["paillier", {"key_length": 1024}]],
}

learners = [
    HeteroLogisticRegressionGuest,
    HeteroLogisticRegressionHost,
    HeteroLogisticRegressionCoord
]

test_cases = {
    'uci_credit': case_template(
        train_files = [
            '/mnt/nfs/datasets/UCI_credit/hetero/train_2_0.csv',
            '/mnt/nfs/datasets/UCI_credit/hetero/train_2_1.csv'
        ],
        test_files = [
            '/mnt/nfs/datasets/UCI_credit/hetero/test_2_0.csv',
            '/mnt/nfs/datasets/UCI_credit/hetero/test_2_1.csv'
        ],
        descs = [
            {'id_desc': ['id'], 'y_desc': ['default.payment.next.month']},
            {'id_desc': ['id']}
        ],
        meta_params = {
            'train_param': {
                'learning_rate': 0.01, 
                'batch_size': 512, 
                'num_epoches': 2
            },
            'predict_param': {
                'batch_size': 512
            },
            'security' : security
        },
        learners = learners,
        metrics = {
            'ks': lambda x: x > 0.4,
            'auc': lambda x: x > 0.7
        },
        timeout = 12000
    ),
    'shap_finance': case_template(
        train_files = [
            '/mnt/nfs/datasets/shap_finance/train_guest.csv',
            '/mnt/nfs/datasets/shap_finance/train_host.csv'
        ],
        test_files = [
            '/mnt/nfs/datasets/shap_finance/test_guest.csv',
            '/mnt/nfs/datasets/shap_finance/test_host.csv'
        ],
        descs = [
            {'id_desc': ['ID'], 'y_desc': ['Label']},
            {'id_desc': ['ID']}
        ],
        meta_params = {
            'train_param': {
                'learning_rate': 0.01, 
                'batch_size': 512, 
                'num_epoches': 2
            },
            'predict_param': {
                'batch_size': 512
            },
            'security' : security
        },
        learners = learners,
        metrics = {
            'ks': lambda x: x > 0.55,
            'auc': lambda x: x > 0.8
        },
        timeout = 12000
    )
}
##############################################################################

from functools import partial
for case_name, case in test_cases.items():
    locals()[f'test_{case_prefix}{case_name}'] = partial(run_processes, {case_name: case})

if __name__ == '__main__':
    for case_name, case in test_cases.items():
        eval(f'test_{case_prefix}{case_name}()')
