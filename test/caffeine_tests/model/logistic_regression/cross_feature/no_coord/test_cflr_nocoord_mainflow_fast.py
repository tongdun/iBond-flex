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
    'fate_breast': case_template_no_coord(
        train_files = [
            '/mnt/nfs/datasets/fate_breast/hetero/breast_a.csv',
            '/mnt/nfs/datasets/fate_breast/hetero/breast_b.csv'
        ],
        test_files = [
            '/mnt/nfs/datasets/fate_breast/hetero/breast_a.csv',
            '/mnt/nfs/datasets/fate_breast/hetero/breast_b.csv'
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
                        'max_epoch': 3
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
                }
            },
        },
        learners = learners,
        metrics = {
            'ks': lambda x: x > 0.8,
            'auc': lambda x: x > 0.85
        },
        timeout = 500
    )
}
##############################################################################

from functools import partial
for case_name, case in test_cases.items():
    locals()[f'test_{case_prefix}{case_name}'] = partial(run_processes, {case_name: case})

if __name__ == '__main__':
    for case_name, case in test_cases.items():
        eval(f'test_{case_prefix}{case_name}()')
