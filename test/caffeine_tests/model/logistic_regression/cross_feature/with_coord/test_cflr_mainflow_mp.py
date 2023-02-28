from functools import partial

from caffeine.model.logistic_regression.cross_feature.with_coord.coordinator import HeteroLogisticRegressionCoord
from caffeine.model.logistic_regression.cross_feature.with_coord.guest import HeteroLogisticRegressionGuest
from caffeine.model.logistic_regression.cross_feature.with_coord.host import HeteroLogisticRegressionHost
from caffeine_tests.model.hetero_mainflow_mp import case_template, run_processes

##############################################################################
case_prefix = 'hetero_lr_multipart_'

# security = {
#    'he_algo': 'paillier',
#    'he_key_length': 1024,
#    'key_exchange_size': 2048
# }
security = [
    ['paillier', {'key_length': 1024}],
]

learners = [
    HeteroLogisticRegressionGuest,
    HeteroLogisticRegressionHost,
    HeteroLogisticRegressionHost,
    HeteroLogisticRegressionCoord
]

test_cases = {
    'shap_finance': case_template(
        roles = [
            'guest',
            'host',
            'host',
            'coordinator',
        ],
        train_files = [
            '/mnt/nfs/datasets/shap_finance/hetero/train_5_0.csv',
            '/mnt/nfs/datasets/shap_finance/hetero/train_5_1.csv',
            '/mnt/nfs/datasets/shap_finance/hetero/train_5_2.csv',
        ],
        test_files = [
            '/mnt/nfs/datasets/shap_finance/hetero/test_5_0.csv',
            '/mnt/nfs/datasets/shap_finance/hetero/test_5_1.csv',
            '/mnt/nfs/datasets/shap_finance/hetero/test_5_2.csv',
        ],
        descs = [
            {'id_desc': ['ID'], 'y_desc': ['Label']},
            {'id_desc': ['ID']},
            {},
        ],
        meta_params = {
            'train_param': {
                'learning_rate': 0.01, 
                'batch_size': 128, 
                'num_epoches': 2
            },
            'predict_param': {
                'batch_size': 128
            },
            'security' : security
        },
        learners = learners,
        metrics = {
            'ks': lambda x: x > 0.55,
            'auc': lambda x: x > 0.8
        },
        timeout = 2400
    )
}
##############################################################################

from functools import partial
for case_name, case in test_cases.items():
    locals()[f'test_{case_prefix}{case_name}'] = partial(run_processes, {case_name: case})

if __name__ == '__main__':
    for case_name, case in test_cases.items():
        eval(f'test_{case_prefix}{case_name}()')
