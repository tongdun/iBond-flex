import signal
from logging import getLogger
from multiprocessing import Process

from caffeine.utils import Middleware
from caffeine_tests.config import fed_conf_guest1_no_host, fed_conf_guest2_no_host, fed_conf_coordinator_no_host
from caffeine_tests.model.hetero_mainflow import load_data

logger = getLogger('Homo_tester')


def case_template(train_files, test_files, descs, meta_params, learners, metrics={}, scale=True, timeout=300):
    return {
        'guest1': {
            'input': {
                'learner': learners[0],
                'fed_conf': fed_conf_guest1_no_host,
                'role': 'guest',
                'train_csv': train_files[0],
                'test_csv': test_files[0],
                'csv_desc': descs[0],
                'scale': scale,
                'meta_params': meta_params,
                'metrics': metrics
            },
            'timeout': timeout
        },
        'guest2': {
            'input': {
                'learner': learners[1],
                'fed_conf': fed_conf_guest2_no_host,
                'role': 'guest',
                'train_csv': train_files[1],
                'test_csv': test_files[1],
                'csv_desc': descs[1],
                'scale': scale,
                'meta_params': meta_params
            },
            'timeout': timeout
        },
        'coordinator': {
            'input': {
                'learner': learners[2],
                'fed_conf': fed_conf_coordinator_no_host,
                'role': 'coordinator',
                'meta_params': meta_params
            },
            'timeout': timeout
        }
    }


def homo_run(casename, input, expect=None, timeout=120):
    fed_conf = input.get('fed_conf')
    role = input['role']

    middleware = Middleware(
        config={
            'engine': 'light',
            'warehouse': f'/mnt/nfs/tmp/caffeine/models_{role}',
            'bulletin': {
                'name': 'caffeine_test',
                'bond_dag_task_uuid': fed_conf['session']['job_id'] + casename
            }
        }
    ).session

    fed_conf['session']['job_id'] = fed_conf['session']['job_id'] + casename

    train_meta_params = {
        'train_param': input.get('meta_params', {}).get('train_param', {}),
        # 'predict_param': input.get('meta_params', {}).get('predict_param', {}),
        'security': input.get('meta_params', {}).get('security', {}),
        'federal_info': input.get('fed_conf'),
        # 'algo_param': get_algo_param(input), # new
        # 'category_feature': None # new
    }

    # todo bug
    pure_predict_meta_params = {
        'predict_param': input.get('meta_params', {}).get('predict_param', {}),
        'security': input.get('meta_params', {}).get('security', {}),
        # 'category_feature': None # new
    }

    learner_class = input.get('learner')
    learner = learner_class(train_meta_params, context=middleware)

    if role in ['guest', 'host']:
        # get and process data
        train_data, mean, std = load_data(
            input.get('train_csv'), input.get('csv_desc'), middleware, None, None, input.get('scale'))
        test_data, _, _ = load_data(
            input.get('test_csv'), input.get('csv_desc'), middleware, mean, std, input.get('scale'))

        # train
        model_infos = learner.train(train_data, test_data)
        if 'metrics' in input:
            for metric, judge in input.get('metrics').items():
                logger.info(f'Start check validation metric {metric} by {judge}.')
                assert 'metrics' in model_infos[-1].model_attributes
                assert metric in model_infos[-1].model_attributes['metrics']
                metric_value = model_infos[-1].model_attributes['metrics'][metric]
                logger.info(f'Final validation metric {metric} for case {casename} is {metric_value}!!!')
                assert judge(metric_value)
                logger.info(f'Validation metric {metric} by {judge} passed.')
        logger.info(f"{role}: train test ended, start predicting...")

        # predict
        final_results = learner.predict(test_data)
        # reload
        learner = learner_class(
            pure_predict_meta_params,
            middleware,
            middleware.load_model(
                model_idx=model_infos[-1].model_id,
                #model_name=f'{fed_conf["session"]["local_id"]}_model' # with random id not need 
            )
        )
        final_results_loaded = learner.predict(test_data)
        # check
        equals = final_results.to_numpy() == final_results_loaded.to_numpy()
        assert equals.all()
        # check continue training
        logger.info(f'>len model_infos: {len(model_infos)}')
        if False: #len(model_infos) > 1:
            logger.info('------Check continue training--------')
            train_meta_params['federal_info']['session']['job_id'] = fed_conf['session']['job_id']+'continue'
            learner = learner_class(
                train_meta_params,
                middleware,
                middleware.load_model(model_infos[-2].model_id)
            )
            learner.train(train_data, test_data)

    elif role == 'coordinator':
        # train
        model_infos = learner.train(None, None)
        # predict
        learner.predict(None)
        # reload
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


def homo_run_processes(test_cases):
    for test_name, test_case in test_cases.items():
        logger.info(f'!!!Start to test {test_name}.')

        process_list = []
        try:
            for role, config in test_case.items():
                timeout = config.get('timeout', 300)
                config['casename'] = test_name
                p = Process(target=homo_run, kwargs=config)
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
