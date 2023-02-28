from functools import partial

from caffeine.model.logistic_regression.cross_feature.no_coord.guest import HeteroLogisticRegressionNoCoordGuest
from caffeine.model.logistic_regression.cross_feature.no_coord.host import HeteroLogisticRegressionNoCoordHost
from caffeine_tests.model.hetero_mainflow import case_template_no_coord, run_processes

##############################################################################
case_prefix = 'hetero_lr_nocoord_'

learners = [
    HeteroLogisticRegressionNoCoordGuest,
    HeteroLogisticRegressionNoCoordHost,
]

test_cases = {
    'fate_breast_early_stop_iter_abs': case_template_no_coord(
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
                'optimizer': {
                    'type': 'SGD',
                    'parameters': {
                        'learning_rate': 0.0001, 
                        'batch_size': 32, 
                        'regularization': 'L2',
                        'alpha': 0.1,
                        'max_epoch': 2
                    }
                },
                'encryptor': {
                    'type': 'Paillier',
                    'parameters': {
                        'key_length': 1024,
                    }
                },
                'initializer': {
                    'type': 'zeros',
                },
                'early_stop_param':{
                    'early_stop': True,
                    'early_stop_step': 'iter',
                    'early_stop_method': 'abs',
                    'early_stop_eps': 0.69
                },
            },
        },
        learners = learners,
        metrics = {
            'ks': lambda x: x > 0.0,
            'auc': lambda x: x > 0.0
        },
        timeout = 300
    ),
    'fate_breast_early_stop_iter_diff': case_template_no_coord(
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
                'optimizer': {
                    'type': 'SGD',
                    'parameters': {
                        'learning_rate': 0.0001,
                        'batch_size': 32,
                        'regularization': 'L2',
                        'alpha': 0.1,
                        'max_epoch': 2
                    }
                },
                'encryptor': {
                    'type': 'Paillier',
                    'parameters': {
                        'key_length': 1024,
                    }
                },
                'initializer': {
                    'type': 'zeros',
                },
                'early_stop_param':{
                    'early_stop': True,
                    'early_stop_step': 'iter',
                    'early_stop_method': 'diff',
                    'early_stop_eps': 0.69
                },
            },
        },
        learners = learners,
        metrics = {
            'ks': lambda x: x > 0.0,
            'auc': lambda x: x > 0.0
        },
        timeout = 300
    ),
    'fate_breast_early_stop_epoch_abs': case_template_no_coord(
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
                'optimizer': {
                    'type': 'SGD',
                    'parameters': {
                        'learning_rate': 0.0001,
                        'batch_size': 32,
                        'regularization': 'L2',
                        'alpha': 0.1,
                        'max_epoch': 2
                    }
                },
                'encryptor': {
                    'type': 'Paillier',
                    'parameters': {
                        'key_length': 1024,
                    }
                },
                'initializer': {
                    'type': 'zeros',
                },
                'early_stop_param':{
                    'early_stop': True,
                    'early_stop_step': 'epoch',
                    'early_stop_method': 'abs',
                    'early_stop_eps': 0.69
                },
            },
        },
        learners = learners,
        metrics = {
            'ks': lambda x: x > 0.0,
            'auc': lambda x: x > 0.0
        },
        timeout = 300
    ),
    'fate_breast_early_stop_epoch_diff': case_template_no_coord(
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
                'optimizer': {
                    'type': 'SGD',
                    'parameters': {
                        'learning_rate': 0.0001,
                        'batch_size': 32,
                        'regularization': 'L2',
                        'alpha': 0.1,
                        'max_epoch': 2
                    }
                },
                'encryptor': {
                    'type': 'Paillier',
                    'parameters': {
                        'key_length': 1024,
                    }
                },
                'initializer': {
                    'type': 'zeros',
                },
                'early_stop_param':{
                    'early_stop': True,
                    'early_stop_step': 'epoch',
                    'early_stop_method': 'diff',
                    'early_stop_eps': 0.69
                },
            },
        },
        learners = learners,
        metrics = {
            'ks': lambda x: x > 0.0,
            'auc': lambda x: x > 0.0
        },
        timeout = 300
    ),


}
##############################################################################

from functools import partial
for case_name, case in test_cases.items():
    locals()[f'test_{case_prefix}{case_name}'] = partial(run_processes, {case_name: case})

if __name__ == '__main__':
    for case_name, case in test_cases.items():
        eval(f'test_{case_prefix}{case_name}()')
