import signal
import time
import copy
import vaex
# from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger
from multiprocessing import Process

import pandas as pd
import numpy as np

from caffeine.utils import IBondDataFrame
from caffeine.utils import Middleware
from caffeine_tests.config import fed_conf_coordinator
from caffeine_tests.config import fed_conf_guest
from caffeine_tests.config import fed_conf_guest_no_coordinator
from caffeine_tests.config import fed_conf_host
from caffeine_tests.config import fed_conf_host_no_coordinator

from caffeine_tests.config import fed_conf_host_no_coordinator_2_machine
from caffeine_tests.config import fed_conf_guest_no_coordinator_2_machine

logger = getLogger('Hetero_tester')


def case_template(train_files, test_files, descs, meta_params, learners, metrics={}, scale=True, timeout=300, check_predict=True):
    return {
        'guest': {
            'input': {
                'learner': learners[0],
                'fed_conf': fed_conf_guest,
                'role': 'guest', 
                'train_csv': train_files[0],
                'test_csv': None if len(test_files) <= 0 else test_files[0],
                'csv_desc': descs[0],
                'scale': scale,
                'meta_params': meta_params,
                'metrics': metrics
            },
            'timeout': timeout,
            'check_predict': check_predict,
            'extra_flag': True if len(test_files) > 0 else False
        },
        'host': {
            'input': {
                'learner': learners[1],
                'fed_conf': fed_conf_host,
                'role': 'host',
                'train_csv': train_files[1],
                'test_csv': None if len(test_files) <= 0 else test_files[1],
                'csv_desc': descs[1],
                'scale': scale,
                'meta_params': meta_params
            },
            'timeout': timeout,
            'check_predict': check_predict,
            'extra_flag': True if len(test_files) > 0 else False
        },
        'coordinator': {
            'input': {
                'learner': learners[2],
                'fed_conf': fed_conf_coordinator,
                'role': 'coordinator',
                'meta_params': meta_params
            },
            'timeout': timeout,
            'check_predict': check_predict,
            'extra_flag': True if len(test_files) > 0 else False
        }
    }

def case_template_no_coord(train_files, test_files, descs, meta_params, learners, metrics = {}, scale = False, timeout = 300, check_predict=True):
    return {
        'guest': {
            'input': {
                'learner': learners[0],
                'fed_conf': fed_conf_guest_no_coordinator,
                'role': 'guest',
                'train_csv': train_files[0],
                'test_csv': None if len(test_files) <= 0 else test_files[0],
                'csv_desc': descs[0],
                'scale': scale,
                'meta_params': meta_params,
                'metrics': metrics
            },
            'timeout': timeout,
            'check_predict': check_predict,
            'extra_flag': True if len(test_files) > 0 else False
        },
        'host': {
            'input': {
                'learner': learners[1],
                'fed_conf': fed_conf_host_no_coordinator,
                'role': 'host',
                'train_csv': train_files[1],
                'test_csv': None if len(test_files) <= 0 else test_files[1],
                'csv_desc': descs[1],
                'scale': scale,
                'meta_params': meta_params
            },
            'timeout': timeout,
            'check_predict': check_predict,
            'extra_flag': True if len(test_files) > 0 else False
        },
    }

def case_template_no_coord_guest(train_files, test_files, descs, meta_params, learners, metrics = {}, scale = False, timeout = 300, check_predict=True):
    return {
        'guest': {
            'input': {
                'learner': learners[0],
                'fed_conf': fed_conf_guest_no_coordinator_2_machine,
                'role': 'guest',
                'train_csv': train_files[0],
                'test_csv': None if len(test_files) <= 0 else test_files[0],
                'csv_desc': descs[0],
                'scale': scale,
                'meta_params': meta_params,
                'metrics': metrics
            },
            'timeout': timeout,
            'check_predict': check_predict,
            'extra_flag': True if len(test_files) > 0 else False
        },
    }
def case_template_no_coord_host(train_files, test_files, descs, meta_params, learners, metrics = {}, scale = False, timeout = 300, check_predict=True):
    return {
        'host': {
            'input': {
                'learner': learners[0],
                'fed_conf': fed_conf_host_no_coordinator_2_machine,
                'role': 'host',
                'train_csv': train_files[0],
                'test_csv': None if len(test_files) <= 0 else test_files[0],
                'csv_desc': descs[0],
                'scale': scale,
                'meta_params': meta_params,
                'metrics': metrics
            },
            'timeout': timeout,
            'check_predict': check_predict,
            'extra_flag': True if len(test_files) > 0 else False
        },
    }

def load_data(f, desc, middleware, mean = None, std = None, scale = True):
    pdf = vaex.open(f).to_pandas_df(array_type='numpy')
    id_cols = desc['id_desc']
    for id_col in id_cols:
        pdf[id_col] = pdf[id_col].astype(str)
    # NOTE std features
    nonfeat_cols = sum(desc.values(), [])
    fc = [n for n in pdf.columns if n not in nonfeat_cols]
    if mean is None or std is None:
        mean = pdf[fc].mean()
        std = pdf[fc].std()
    if scale is True:
        pdf[fc] = ((pdf[fc] - mean)/(std+1.e-10)).fillna(0)
    data = middleware.create_dataframe(
        pdf,
        desc
    )
    return data, mean, std

def run(casename, input, expect=None, timeout=120, check_predict=True, extra_flag=True):
    fed_conf = input.get('fed_conf')
    role = input['role']

    middleware = Middleware(
        config={
            'engine': 'light',
            'warehouse': f'/mnt/nfs/tmp/caffeine/models_{role}',
            'bulletin': {
                'name': 'caffeine_test',
                'bond_dag_task_uuid': fed_conf['session']['job_id']+casename
            }
        }
    ).session

    fed_conf['session']['job_id'] = fed_conf['session']['job_id']+casename

    train_meta_params = {
        'train_param': input.get('meta_params', {}).get('train_param', {}),
        'security_param': input.get('meta_params', {}).get('security', {}),
        'federal_info': input.get('fed_conf'),
        # 'algo_param': get_algo_param(input), # new
        # 'category_feature': None # new
    }

    if extra_flag:
        train_meta_params['predict_param'] = input.get('meta_params', {}).get('predict_param', {})
        predict_federal_info = copy.deepcopy(input.get('fed_conf'))
        predict_job_id = predict_federal_info['session']['job_id'] + '_pure_predict'
        predict_federal_info['session']['job_id'] = predict_job_id
        pure_predict_meta_params = {
            'predict_param': input.get('meta_params', {}).get('predict_param', {}),
            'security_param': input.get('meta_params', {}).get('security', {}),
            'federal_info': predict_federal_info
            # 'category_feature': None # new
        }

    learner_class = input.get('learner')
    learner = learner_class(train_meta_params, context=middleware)

    if role in ['guest', 'host']:
        # get and process data
        train_data, mean, std = load_data(
            input.get('train_csv'), input.get('csv_desc'), middleware,  None, None, input.get('scale'))
        if extra_flag:
            test_data, _, _ = load_data(
                input.get('test_csv'), input.get('csv_desc'), middleware,  mean, std, input.get('scale'))
        else:
            test_data = None

        # train
        model_infos = learner.train(train_data, test_data)
        logger.info('~~~~~~~~~~~~~~~~~~~~~')
        logger.info(model_infos)
        logger.info('~~~~~~~~~~~~~~~~~~~~~')
        if extra_flag:
            # if 'metrics' in input:
            #     for metric, judge in input.get('metrics').items():
            #         logger.info(f'Start check validation metric {metric} by {judge}.')
            #         assert 'metrics' in model_infos[-1].model_attributes
            #         assert metric in model_infos[-1].model_attributes['metrics']
            #         metric_value = model_infos[-1].model_attributes['metrics'][metric]
            #         logger.info(f'Final validation metric {metric} for case {casename} is {metric_value}!!!')
            #         assert judge(metric_value)
            #         logger.info(f'Validation metric {metric} by {judge} passed.')

            # predict
            logger.info(f"{role}: train test ended, start predicting...")
            final_results = learner.predict(test_data)

            print(pure_predict_meta_params)
            print("===========================")
            # reload
            learner = learner_class(
                pure_predict_meta_params,
                middleware,
                middleware.load_model(model_infos[-1].model_id)
            )
            final_results_loaded = learner.predict(test_data)
            print(final_results_loaded)
            # # check
            # if role == 'guest':
            #     if final_results is None:
            #         assert final_results_loaded is None
            #         equals = np.array([True])
            #     else:
            #         equals = final_results.to_numpy() == final_results_loaded.to_numpy()
            #
            #     if check_predict:
            #         assert equals.all()
            #     else:
            #         not_euqals = np.where(equals == False)
            #         logger.info(f'not euqal:{not_euqals}')
            #         logger.info(f'final_result not equal:{[final_results.to_numpy()[not_euqals]]}')
            #         logger.info(f'final_result_loaded_not_euqal:{[final_results_loaded.to_numpy()[not_euqals]]}')
            #         logger.info(f'not_equals_numbers:{len(not_euqals[0])}')
            # # check continue training
            # logger.info(f'>len model_infos: {len(model_infos)}')
            # if False: #len(model_infos) > 1:
            #     logger.info('------Check continue training--------')
            #     train_meta_params['federal_info']['session']['job_id'] = fed_conf['session']['job_id']+'continue'
            #     learner = learner_class(
            #         train_meta_params,
            #         middleware,
            #         middleware.load_model(model_infos[-2].model_id)
            #     )
            #     learner.train(train_data, test_data)

    elif role == 'coordinator':
        # train
        model_infos = learner.train(None, None)
        logger.info(f'Run test: {extra_flag}')
        if extra_flag:
            # predict
            learner.predict(None)
            # reload
            '''
            predict_federal_info = input.get('fed_conf')
            predict_job_id = predict_federal_info['session']['job_id'] + '_pure_predict'
            predict_federal_info['session']['job_id'] = predict_job_id
            pure_predict_meta_params = {
                'predict_param': input.get('meta_params', {}).get('predict_param', {}),
                'security': input.get('meta_params', {}).get('security', {}),
                'federal_info': predict_federal_info
                # 'category_feature': None # new
            }
            '''
            learner = learner_class(
                pure_predict_meta_params,
                middleware,
                middleware.load_model(model_infos[-1].model_id)
            )
            learner.predict(None)
            logger.info(f'>len model_infos: {len(model_infos)}')
            if False: #len(model_infos) > 1:
                logger.info('------Check continue training--------')
                train_meta_params['federal_info']['session']['job_id'] = fed_conf['session']['job_id']+'continue'
                learner = learner_class(
                    train_meta_params,
                    middleware,
                    middleware.load_model(model_infos[-2].model_id)
                )
                learner.train(None, None)

    logger.info(f'Run ended for {casename} {role}.')


def run_processes(test_cases):
    for test_name, test_case in test_cases.items():
        logger.info(f'!!!Start to test {test_name}.')

        process_list = []
        try:
            for role, config in test_case.items():
                timeout = config.get('timeout', 300)
                config['casename'] = test_name
                p = Process(target=run, kwargs=config)
                p.start()
                process_list.append(p)

            def timeout_handler(signum, frame):
                logger.info(f'Timeout!')
                for p in process_list:
                    p.terminate()
                raise Exception('Case Error!')

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

            for p in process_list:
                p.join()

            logger.info(f'Testcase {test_name} finished!!!')
            signal.alarm(0)
        except KeyboardInterrupt:
            logger.info(f'User Interrupt!')
            for p in process_list:
                p.terminate()
            break
