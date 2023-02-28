import pandas as pd
import signal
import time
# from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger
from multiprocessing import Process

from caffeine.utils import IBondDataFrame
from caffeine.utils import Middleware
from caffeine_tests.config import fed_conf_coordinator_mp
from caffeine_tests.config import fed_conf_guest_mp
from caffeine_tests.config import fed_conf_guest_no_coordinator
from caffeine_tests.config import fed_conf_host_mp
from caffeine_tests.config import fed_conf_host_no_coordinator
from caffeine_tests.config import fed_conf_no_coordinator_guest_mp
from caffeine_tests.config import fed_conf_no_coordinator_host_mp

logger = getLogger('Hetero_tester')


def case_template(roles, train_files, test_files, descs, meta_params, learners, metrics={}, scale=True, timeout=300, check_predict=True):
    template_ret = {}
    guest_count = 0
    host_count = 0
    for index in range(0, len(roles)):
        key = ''
        fed_conf = {}
        if roles[index] == 'guest':
            key = 'guest{}'.format(guest_count)
            fed_conf = fed_conf_guest_mp
            guest_count += 1
        elif roles[index] == 'host':
            key = 'host{}'.format(host_count)
            fed_conf = fed_conf_host_mp[host_count]
            host_count += 1
        elif roles[index] == 'coordinator':
            key = 'coordinator'
            template_ret[key] = {
                'input': {
                    'learner': learners[index],
                    'fed_conf': fed_conf_coordinator_mp,
                    'role': 'coordinator',
                    'meta_params': meta_params
                },
                'timeout': timeout,
                'check_predict': check_predict,
                'extra_flag': True if len(test_files) > 0 else False
            }
            continue
        else:
            continue
        template_ret[key] = {
            'input':{
                'learner': learners[index],
                'fed_conf': fed_conf,
                'role': roles[index], 
                'train_csv': train_files[index] if index < len(train_files) else None,
                'test_csv': test_files[index] if len(test_files)>0 else None,
                'csv_desc': descs[index],
                'scale': scale,
                'meta_params': meta_params,
            },
            'timeout': timeout,
            'check_predict': check_predict,
            'extra_flag': True if len(test_files) > 0 else False
        }
        if roles[index] == 'guest':
            template_ret[key]['input']['metrics'] = metrics
    return template_ret


def case_template_no_coord(roles, train_files, test_files, descs, meta_params, learners, metrics={}, scale=True, timeout=300, check_predict=True):
    template_ret = {}
    guest_count = 0
    host_count = 0
    for index in range(0, len(roles)):
        key = ''
        fed_conf = {}
        if roles[index] == 'guest':
            key = 'guest{}'.format(guest_count)
            fed_conf = fed_conf_no_coordinator_guest_mp
            guest_count += 1
        elif roles[index] == 'host':
            key = 'host{}'.format(host_count)
            fed_conf = fed_conf_no_coordinator_host_mp[host_count]
            host_count += 1
        else:
            continue
        template_ret[key] = {
            'input':{
                'learner': learners[index],
                'fed_conf': fed_conf,
                'role': roles[index],
                'train_csv': train_files[index] if index < len(train_files) else None,
                'test_csv': test_files[index] if len(test_files)>0 else None,
                'csv_desc': descs[index],
                'scale': scale,
                'meta_params': meta_params,
            },
            'timeout': timeout,
            'check_predict': check_predict,
            'extra_flag': True if len(test_files) > 0 else False
        }
        if roles[index] == 'guest':
            template_ret[key]['input']['metrics'] = metrics
    return template_ret


def load_data(f, desc, middleware, mean = None, std = None, scale = True):
    pdf = pd.read_csv(f)
    # NOTE std features
    nonfeat_cols = sum(desc.values(), [])
    fc = [n for n in pdf.columns if n not in nonfeat_cols]
    if mean is None or std is None:
        mean = pdf[fc].mean()
        std = pdf[fc].std()
    if scale is True:
        pdf[fc] = (pdf[fc] - mean)/std
    data = middleware.create_dataframe(
        pdf,
        desc
    )
    return data, mean, std

def run(casename, input, expect=None, timeout=120, check_predict=True, extra_flag=True):
    # middleware = Middleware(
    #     config={
    #         'engine': 'light',
    #         'warehouse': '/mnt/nfs/tmp/caffeine/models/'
    #     }
    # ).session
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

    role = input['role']

    fed_conf = input.get('fed_conf')
    fed_conf['session']['job_id'] = fed_conf['session']['job_id']+casename

    # train_meta_params = {
    #     'train_param': input.get('meta_params', {}).get('train_param', {}),
    #     'predict_param': input.get('meta_params', {}).get('predict_param', {}),
    #     'security': input.get('meta_params', {}).get('security', {}),
    #     'federal_info': input.get('fed_conf')
    # }
    #
    # pure_predict_meta_params = {
    #     'predict_param': input.get('meta_params', {}).get('predict_param', {}),
    #     'security': input.get('meta_params', {}).get('security', {})
    # }


    train_meta_params = {
        'train_param': input.get('meta_params', {}).get('train_param', {}),
        'security': input.get('meta_params', {}).get('security', {}),
        'federal_info': input.get('fed_conf'),
        # 'algo_param': get_algo_param(input), # new
        # 'category_feature': None # new
    }

    if extra_flag:
        train_meta_params['predict_param'] = input.get('meta_params', {}).get('predict_param', {})
        pure_predict_meta_params = {
            'predict_param': input.get('meta_params', {}).get('predict_param', {}),
            'security': input.get('meta_params', {}).get('security', {}),
            # 'category_feature': None # new
        }



    logger.info(f'{role}: learner before create trainer!')
    learner_class = input.get('learner')
    learner = learner_class(train_meta_params, context=middleware)

    logger.info(f'{role}: learner create trainer!')
    #if role in ['guest', 'host']:
    if role in ['guest', 'host']:
        # get and process data
        train_data, mean, std = load_data(
            input.get('train_csv'), input.get('csv_desc'), middleware,  None, None, input.get('scale'))
        # test_data, _, _ = load_data(
        #     input.get('test_csv'), input.get('csv_desc'), middleware,  mean, std, input.get('scale'))

        if extra_flag:
            test_data, _, _ = load_data(
                input.get('test_csv'), input.get('csv_desc'), middleware, mean, std, input.get('scale'))
        else:
            test_data = None

        # train
        logger.info(f'{role}: learner start to train!')
        model_infos = learner.train(train_data, test_data)
        # if 'metrics' in input:
        #     for metric, judge in input.get('metrics').items():
        #         logger.info(f'Start check validation metric {metric} by {judge}.')
        #         assert 'metrics' in model_infos[-1].model_attributes
        #         assert metric in model_infos[-1].model_attributes['metrics']
        #         metric_value = model_infos[-1].model_attributes['metrics'][metric]
        #         logger.info(f'Final validation metric {metric} for case {casename} is {metric_value}!!!')
        #         assert judge(metric_value)
        #         logger.info(f'Validation metric {metric} by {judge} passed.')
        # logger.info(f"{role}: train test ended, start predicting...")
        #
        # # predict
        # final_results = learner.predict(test_data)
        # # reload
        # learner = learner_class(
        #     pure_predict_meta_params,
        #     middleware,
        #     middleware.load_model(model_infos[-1].model_id)
        # )
        # final_results_loaded = learner.predict(test_data)
        # # check
        # equals = final_results.to_numpy() == final_results_loaded.to_numpy()
        # assert equals.all()
        if extra_flag:
            if 'metrics' in input:
                for metric, judge in input.get('metrics').items():
                    logger.info(f'Start check validation metric {metric} by {judge}.')
                    assert 'metrics' in model_infos[-1].model_attributes
                    assert metric in model_infos[-1].model_attributes['metrics']
                    metric_value = model_infos[-1].model_attributes['metrics'][metric]
                    logger.info(f'Final validation metric {metric} for case {casename} is {metric_value}!!!')
                    assert judge(metric_value)
                    logger.info(f'Validation metric {metric} by {judge} passed.')

            # predict
            logger.info(f"{role}: train test ended, start predicting...")
            final_results = learner.predict(test_data)
            logger.info(f"{role}: predicting done ...")

            logger.info(f"{role}: model_infos:{model_infos}...")

            # reload
            learner = learner_class(
                pure_predict_meta_params,
                middleware,
                middleware.load_model(model_infos[-1].model_id)
            )

            logger.info(f"{role}: reload ended, start reload predicting  ...")
            final_results_loaded = learner.predict(test_data)
            logger.info(f"{role}: reload predicting done ...")
            # check
            equals = final_results.to_numpy() == final_results_loaded.to_numpy()

            if check_predict:
                assert equals.all()
            else:
                import numpy as np
                not_euqals = np.where(equals == False)
                logger.info(f'not euqal:{not_euqals}')
                logger.info(f'final_result not equal:{[final_results.to_numpy()[not_euqals]]}')
                logger.info(f'final_result_loaded_not_euqal:{[final_results_loaded.to_numpy()[not_euqals]]}')
                logger.info(f'not_equals_numbers:{len(not_euqals[0])}')

    elif role == 'coordinator':
        # train
        model_infos = learner.train(None, None)
        # # predict
        # learner.predict(None)
        # # reload
        # learner = learner_class(
        #     pure_predict_meta_params,
        #     middleware,
        #     middleware.load_model(model_infos[-1].model_id)
        # )
        # learner.predict(None)
        if extra_flag:
            # predict
            learner.predict(None)
            # reload
            pure_predict_meta_params = {
                'predict_param': input.get('meta_params', {}).get('predict_param', {}),
                'security': input.get('meta_params', {}).get('security', {}),
                # 'category_feature': None # new
            }
            learner = learner_class(
                pure_predict_meta_params,
                middleware,
                middleware.load_model(model_infos[-1].model_id)
            )
            learner.predict(None)


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
