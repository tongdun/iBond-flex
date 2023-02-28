import signal
import time
from flex.api import SAL, ECDH_SAL
from logging import getLogger
from multiprocessing import Process, Queue
import hashlib

from caffeine.alignment.sample_align import SampleAligner
from caffeine_tests.config import fed_conf_coordinator
from caffeine_tests.config import fed_conf_guest
from caffeine_tests.config import fed_conf_guest_no_coordinator
from caffeine_tests.config import fed_conf_host
from caffeine_tests.config import fed_conf_host_no_coordinator

logger = getLogger('Alignment')

test_cases = {
    
    #with third party
    'not_small': {
        'guest': {
            'input': {
                'fed_conf': fed_conf_guest,
                'id_cols': [list(map(lambda x: hashlib.md5(str(x).encode()).hexdigest(), range(100000000)))],
                'align_method': SAL,
                'security': [['aes', {'key_length': 128}]]
            },
            'timeout': 60,
            'expect': [['1', '333']]
        },
        'host': {
            'input': {
                'fed_conf': fed_conf_host,
                'id_cols': [list(map(lambda x: hashlib.md5(str(x).encode()).hexdigest(), range(100000)))],
                'align_method': SAL,
                'security': [['aes', {'key_length': 128}]]
            },
            'timeout': 60,
            'expect':  [list(map(lambda x: hashlib.md5(str(x).encode()).hexdigest(), range(100000)))],
        },
        'coordinator': {
            'input': {
                'fed_conf': fed_conf_coordinator,
                'id_cols': [[]],
                'align_method': SAL,
                'security': [['aes', {'key_length': 128}]]
            },
            'timeout': 3600
        },
    },
}


def run(input, role, share_queue, expect=None, casename='', timeout=60):
    fed_conf = input['fed_conf']
    job_id_prefix = fed_conf['session']['job_id']
    fed_conf['session']['job_id'] = job_id_prefix + casename

    aligner = SampleAligner(input['align_method'], input['fed_conf'], input['security'])
    out = aligner.align(input['id_cols'])
    print('===================')
    if type(out[0]) is list:
        print(role, 'output length', len(out[0]))
        print(role, 'output', out[0][:10])
    print('-------------------')
    print(role, 'expected', list(map(lambda x: hashlib.md5(str(x).encode()).hexdigest(), range(10))))
    print('===================')

    '''
    if role in ['guest', 'host']:
        print(role, 'put to share_queue')
        share_queue.put(out)
        print(role, 'put to share_queue done')
    print('done')
    '''


def test_alignment():
    def timeout_handler(signum, frame):
        logger.info(f'Timeout!')
        for p in process_list:
            p.terminate()
        raise Exception('Case Error!')

    share_queue = Queue()
    for test_name, test_case in test_cases.items():
        #if test_name in ['mock_small',
        # 'mock_small_wo_cooridinator', 'mock_multicolumn_wo_coordinator']:
        if False:
             continue
        else:
             logger.info(test_name)

        logger.info(f'!!!Start to test {test_name}.')

        process_list = []
        try:
            casename_postfix = str(time.time())
            for role, config in test_case.items():
                config['casename'] = test_name + casename_postfix
                config['share_queue'] = share_queue
                config['role'] = role
                timeout = config.get('timeout', 60)
                p = Process(target=run, kwargs=config)
                p.start()
                process_list.append(p)

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

            for p in process_list:
                p.join()
            out_list = []
            '''
            print('Start to get from share queue!')
            for i in range(0, share_queue.qsize()):
                out_list.append(share_queue.get())
            print('Start to compare all parties!')
            for index in range(1, len(out_list)):
                assert out_list[0] == out_list[index]
                #logger.info(f'ref {out_list[0]} target {out_list[index]}')
            print('Done comparing all parties!')
            '''
            logger.info(f'Testcase {test_name} finished!!!')
            signal.alarm(0)
        except KeyboardInterrupt:
            logger.info(f'User Interrupt!')
            for p in process_list:
                p.terminate()
            break

            
        logger.info(f'Testcase {test_name} finished!!!')

if __name__ == '__main__':
    test_alignment()

    """
    for i in range(20):
        logger.info(f"For {i}th iteration....")
        test_alignment()
    """
