import signal
from logging import getLogger
from multiprocessing import Process

from caffeine_tests.config import fed_conf_guest1_no_host, fed_conf_guest2_no_host, fed_conf_coordinator_no_host
from caffeine_tests.model.hetero_mainflow import run

logger = getLogger('Homo_tester')


def case_template(train_files, test_files, descs, meta_params, learners, metrics={}, scale=True, timeout=300, check_predict=True):
    return {
        'guest1': {
            'input': {
                'learner': learners[0],
                'fed_conf': fed_conf_guest1_no_host,
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
        'guest2': {
            'input': {
                'learner': learners[1],
                'fed_conf': fed_conf_guest2_no_host,
                'role': 'guest',
                'train_csv': train_files[1],
                'test_csv': None if len(test_files) <= 0 else test_files[0],
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
                'fed_conf': fed_conf_coordinator_no_host,
                'role': 'coordinator',
                'meta_params': meta_params
            },
            'timeout': timeout,
            'check_predict': check_predict,
            'extra_flag': True if len(test_files) > 0 else False
        }
    }


def homo_run(casename, input, expect=None, timeout=120, check_predict=True, extra_flag=True):
    run(casename, input, expect, timeout, check_predict, extra_flag)


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
