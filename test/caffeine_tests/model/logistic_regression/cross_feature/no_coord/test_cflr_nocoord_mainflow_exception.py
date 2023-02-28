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
    'small_data_no_coord_lr': case_template_no_coord(
        train_files=[
            '/mnt/nfs/base/datasets/abnormal_data/common/small_data/a.csv',
            '/mnt/nfs/base/datasets/abnormal_data/common/small_data/b.csv'
        ],
        test_files=[
            '/mnt/nfs/base/datasets/abnormal_data/common/small_data/a.csv',
            '/mnt/nfs/base/datasets/abnormal_data/common/small_data/b.csv'
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
                        'learning_rate': 0.01, 
                        'batch_size': 128, 
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
                    'type': 'zeros'
                }
            },
        },
        learners = learners,
        metrics = {
            'ks': lambda x: x > 0.0,
            'auc': lambda x: x > 0.0
        },
        timeout = 1200
    ),

    'dtype_no_coord_lr': case_template_no_coord(
        train_files = [
            '/mnt/nfs/datasets/fate_breast/hetero/train_d_guest.csv',
            '/mnt/nfs/datasets/fate_breast/hetero/train_d_host.csv'
        ],
        test_files = [
            '/mnt/nfs/datasets/fate_breast/hetero/test_d_guest.csv',
            '/mnt/nfs/datasets/fate_breast/hetero/test_d_host.csv'
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
                        'learning_rate': 0.01, 
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
                    'type': 'zeros'
                }
            },
        },
        learners = learners,
        metrics = {
            'ks': lambda x: x > 0.8,
            'auc': lambda x: x > 0.85
        },
        timeout = 300
    ),

    'no_val_no_coord_lr': case_template_no_coord(
        train_files = [
            '/mnt/nfs/datasets/fate_breast/hetero/breast_a.csv',
            '/mnt/nfs/datasets/fate_breast/hetero/breast_b.csv'
        ],
        test_files = [],
        descs = [
            {'id_desc': ['id'], 'y_desc': ['y']},
            {'id_desc': ['id']}
        ],
        meta_params = {
            'train_param': {
                'optimizer': {
                    'type': 'SGD',
                    'parameters': {
                        'learning_rate': 0.01, 
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
                    'type': 'zeros'
                }
            },
        },
        learners = learners,
        metrics = {
            'ks': lambda x: x > 0.8,
            'auc': lambda x: x > 0.85
        },
        timeout = 300
    ),

    'multiID_no_coord_lr': case_template_no_coord(
        train_files=[
            '/mnt/nfs/base/datasets/abnormal_data/fate_breast/multiID_breast_a.csv',
            '/mnt/nfs/base/datasets/abnormal_data/fate_breast/multiID_breast_b.csv'
        ],
        test_files=[
            '/mnt/nfs/base/datasets/abnormal_data/fate_breast/multiID_breast_a.csv',
            '/mnt/nfs/base/datasets/abnormal_data/fate_breast/multiID_breast_b.csv'
        ],
        descs = [
            {'id_desc': ['id','new_id'], 'y_desc': ['y']},
            {'id_desc': ['id','new_id']}
        ],
        meta_params = {
            'train_param': {
                'optimizer': {
                    'type': 'SGD',
                    'parameters': {
                        'learning_rate': 0.01, 
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
                    'type': 'zeros'
                }
            },
        },
        learners = learners,
        metrics = {
            'ks': lambda x: x > 0.8,
            'auc': lambda x: x > 0.85
        },
        timeout = 300
    ),

    'multiLabel_no_coord_lr': case_template_no_coord(
        train_files=[
            '/mnt/nfs/base/datasets/abnormal_data/fate_breast/multiLabel_breast_a.csv',
            '/mnt/nfs/base/datasets/abnormal_data/fate_breast/hetero/breast_b.csv'
        ],
        test_files=[
            '/mnt/nfs/base/datasets/abnormal_data/fate_breast/multiLabel_breast_a.csv',
            '/mnt/nfs/base/datasets/abnormal_data/fate_breast/hetero/breast_b.csv'
        ],
        descs = [
            {'id_desc': ['id'], 'y_desc': ['y','new_y']},
            {'id_desc': ['id']}
        ],
        meta_params = {
            'train_param': {
                'optimizer': {
                    'type': 'SGD',
                    'parameters': {
                        'learning_rate': 0.01, 
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
                    'type': 'zeros'
                }
            },
        },
        learners = learners,
        metrics = {
            'ks': lambda x: x > 0.8,
            'auc': lambda x: x > 0.85
        },
        timeout = 300
    )

}
##############################################################################

from functools import partial
for case_name, case in test_cases.items():
    locals()[f'test_{case_prefix}{case_name}'] = partial(run_processes, {case_name: case})

if __name__ == '__main__':
    for case_name, case in test_cases.items():
        eval(f'test_{case_prefix}{case_name}()')
