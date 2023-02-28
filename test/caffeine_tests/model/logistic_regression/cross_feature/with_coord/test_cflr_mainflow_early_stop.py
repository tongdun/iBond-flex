from functools import partial

from caffeine.model.logistic_regression.cross_feature.with_coord.coordinator import HeteroLogisticRegressionCoord
from caffeine.model.logistic_regression.cross_feature.with_coord.guest import HeteroLogisticRegressionGuest
from caffeine.model.logistic_regression.cross_feature.with_coord.host import HeteroLogisticRegressionHost
from caffeine_tests.model.hetero_mainflow import case_template, run_processes

##############################################################################
case_prefix = 'hetero_lr_'

learners = [
    HeteroLogisticRegressionGuest,
    HeteroLogisticRegressionHost,
    HeteroLogisticRegressionCoord
]

#security = {
#    'he_algo': 'paillier',
#    'he_key_length': 1024,
#    'key_exchange_size': 2048
#}
security = [
    ['paillier', {'key_length': 1024}],
]

test_cases = {
    'fate_breast_early_stop_iter_abs': case_template(
        train_files = [
            '/mnt/nfs/datasets/fate_breast/hetero/breast_a.csv',
            '/mnt/nfs/datasets/fate_breast/hetero/breast_b.csv'
        ],
        test_files = [
        ],
        descs = [
            {'id_desc': ['id'], 'y_desc': ['y']},
            {'id_desc': ['id']}
        ],
        meta_params = {
            'train_param': {
                'learning_rate': 0.01,
                'batch_size': 32,
                'num_epoches': 2,
                'with_bias': True,
                'early_stop_param':{
                    'early_stop': True,
                    'early_stop_step': 'iter',
                    'early_stop_method': 'abs',
                    'early_stop_eps': 0.40,
                },
            },
            'predict_param': {
                'batch_size': 512
            },
            'security': security
        },
        learners = learners,
        metrics = {
            'ks': lambda x: x > 0.0,
            'auc': lambda x: x > 0.0
        },
        timeout = 300
    ),
    # 'fate_breast_early_stop_iter_diff': case_template(
    #     train_files = [
    #         '/mnt/nfs/datasets/fate_breast/hetero/breast_a.csv',
    #         '/mnt/nfs/datasets/fate_breast/hetero/breast_b.csv'
    #     ],
    #     test_files = [
    #     ],
    #     descs = [
    #         {'id_desc': ['id'], 'y_desc': ['y']},
    #         {'id_desc': ['id']}
    #     ],
    #     meta_params = {
    #         'train_param': {
    #             'learning_rate': 0.01,
    #             'batch_size': 32,
    #             'num_epoches': 2,
    #             'with_bias': True,
    #             'early_stop_param':{
    #                 'early_stop': True,
    #                 'early_stop_step': 'iter',
    #                 'early_stop_method': 'diff',
    #                 'early_stop_eps': 0.40
    #             },
    #         },
    #         'predict_param': {
    #             'batch_size': 512
    #         },
    #         'security': security
    #     },
    #     learners = learners,
    #     metrics = {
    #         'ks': lambda x: x > 0.0,
    #         'auc': lambda x: x > 0.0
    #     },
    #     timeout = 300
    # ),
    # 'fate_breast_early_stop_epoch_abs': case_template(
    #     train_files = [
    #         '/mnt/nfs/datasets/fate_breast/hetero/breast_a.csv',
    #         '/mnt/nfs/datasets/fate_breast/hetero/breast_b.csv'
    #     ],
    #     test_files = [
    #     ],
    #     descs = [
    #         {'id_desc': ['id'], 'y_desc': ['y']},
    #         {'id_desc': ['id']}
    #     ],
    #     meta_params = {
    #         'train_param': {
    #             'learning_rate': 0.01,
    #             'batch_size': 32,
    #             'num_epoches': 2,
    #             'with_bias': True,
    #             'early_stop_param':{
    #                 'early_stop': True,
    #                 'early_stop_step': 'epoch',
    #                 'early_stop_method': 'abs',
    #                 'early_stop_eps': 0.45
    #             },
    #         },
    #         'predict_param': {
    #             'batch_size': 512
    #         },
    #         'security': security
    #     },
    #     learners = learners,
    #     metrics = {
    #         'ks': lambda x: x > 0.0,
    #         'auc': lambda x: x > 0.0
    #     },
    #     timeout = 300
    # ),
    # 'fate_breast_early_stop_epoch_diff': case_template(
    #     train_files = [
    #         '/mnt/nfs/datasets/fate_breast/hetero/breast_a.csv',
    #         '/mnt/nfs/datasets/fate_breast/hetero/breast_b.csv'
    #     ],
    #     test_files = [
    #     ],
    #     descs = [
    #         {'id_desc': ['id'], 'y_desc': ['y']},
    #         {'id_desc': ['id']}
    #     ],
    #     meta_params = {
    #         'train_param': {
    #             'learning_rate': 0.01,
    #             'batch_size': 32,
    #             'num_epoches': 2,
    #             'with_bias': True,
    #             'early_stop_param':{
    #                 'early_stop': True,
    #                 'early_stop_step': 'epoch',
    #                 'early_stop_method': 'diff',
    #                 'early_stop_eps': 0.35
    #             },
    #         },
    #         'predict_param': {
    #             'batch_size': 512
    #         },
    #         'security': security
    #     },
    #     learners = learners,
    #     metrics = {
    #         'ks': lambda x: x > 0.0,
    #         'auc': lambda x: x > 0.0
    #     },
    #     timeout = 300
    # ),

}
##############################################################################

from functools import partial
for case_name, case in test_cases.items():
    locals()[f'test_{case_prefix}{case_name}'] = partial(run_processes, {case_name: case})

if __name__ == '__main__':
    for case_name, case in test_cases.items():
        eval(f'test_{case_prefix}{case_name}()')
