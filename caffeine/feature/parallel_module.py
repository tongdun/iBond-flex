from typing import Dict, List
from logging import getLogger
import multiprocessing
from multiprocessing import Queue, Process, Manager
import math
from operator import itemgetter

import numpy as np
from copy import deepcopy
from flex.api import make_protocol
from flex.constants import OTP_PN_FL
from flex.commu.ionic_so import commu


class ParallelWorkerBase(object):
    def __init__(self, parallel_num):
        core_num = multiprocessing.cpu_count()
        self.process_num = min(int(core_num*0.5), parallel_num)
        self.logger = getLogger(self.__class__.__name__)
        self.logger.info(f'*** multiprocessing process_num {self.process_num}')

    def run_host(self, *args):
        pass

    def run_guest(self, *args):
        pass

    def run_coord(self, *args):
        pass

    def run_guest_broadcast_channel(self, instance, channel_name, channel_type, msg):
        p = Process(target=guest_broadcast_channel_worker,
                    args=(instance,
                    channel_name,
                    channel_type,
                    msg
                    )
                )
        p.start()
        p.join()

    def run_host_broadcast_channel(self, instance, channel_name, channel_type):
        q = Queue()
        p = Process(target=host_broadcast_channel_worker,
                    args=(instance,
                    channel_name,
                    channel_type,
                    q
                    )
                )
        p.start()
        msg = q.get()
        p.join()
        return msg

    def run_coord_broadcast_channel(self, instance, channel_name, channel_type):
        q = Queue()
        p = Process(target=coord_broadcast_channel_worker,
                    args=(instance,
                    channel_name,
                    channel_type,
                    q
                    )
                )
        p.start()
        if channel_type == 'broadcast':
            num = q.get()
        else:
            num = None
        p.join()
        return num


    def run_guest_gather_channel(self, instance, channel_name, channel_type):
        q = Queue()
        p = Process(target=guest_gather_channel_worker,
                    args=(instance,
                    channel_name,
                    channel_type,
                    q
                    )
                )
        p.start()
        msg = q.get()
        p.join()
        return msg

    def run_host_gather_channel(self, instance, channel_name, channel_type, msg):
        p = Process(target=host_gather_channel_worker,
                    args=(instance,
                    channel_name,
                    channel_type,
                    msg
                    )
                )
        p.start()
        p.join()

    def run_coord_gather_channel(self, instance, channel_name, channel_type):
        p = Process(target=coord_gather_channel_worker,
                    args=(instance,
                    channel_name,
                    channel_type,
                    )
                )
        p.start()
        p.join()

    def run_otp_pn_tl_protocol(self, num, param, tag, federal_info, sec_param):
        q = Queue()
        p = Process(target=otp_pn_fl_worker,
                    args=(num, param, tag, 
                        federal_info, 
                        sec_param,
                        q
                    )
                )
        p.start()
        num = q.get()
        p.join()
        return num
        
    def _get_msg_result(self, results: dict):
        infos = []
        for i in sorted(list(results.keys())):
            infos.append(results[i])         
        return infos

    def _cal_block(self, length: int):
        idx_list = list(range(length))
        block_num = int(math.ceil(length/float(self.process_num)))
        return idx_list, block_num


def init_channel(instance, channel_name, channel_type):
    commu.init(instance.federal_info)
    instance.init_channels({
        channel_name: channel_type
    })
    if channel_type == 'exchange':
        func = getattr(instance, "_"+channel_name+"_chan")
    elif channel_type == 'broadcast':
        func = getattr(instance, "_"+channel_name+"_broadcast")
    return func

def guest_broadcast_channel_worker(
        instance,
        channel_name,
        channel_type,
        msg
    ):
    func = init_channel(instance, channel_name, channel_type)
    func.broadcast(msg)

def host_broadcast_channel_worker(
        instance,
        channel_name,
        channel_type,
        msg_queue
    ):
    func = init_channel(instance, channel_name, channel_type)
    msg = func.broadcast()
    msg_queue.put(msg)

def coord_broadcast_channel_worker(
        instance,
        channel_name,
        channel_type,
        msg_queue,
    ):
    if channel_type != 'broadcast':
        commu.init(instance.federal_info)
    else:
        func = init_channel(instance, channel_name, channel_type)
        msg = func.broadcast()
        msg_queue.put(msg)

def guest_gather_channel_worker(
        instance,
        channel_name,
        channel_type,
        msg_queue
    ):
    func = init_channel(instance, channel_name, channel_type)
    msg = func.gather()
    msg_queue.put(msg)

def host_gather_channel_worker(
        instance,
        channel_name,
        channel_type,
        msg
    ):
    func = init_channel(instance, channel_name, channel_type)
    func.gather(msg)

def coord_gather_channel_worker(
        instance,
        channel_name,
        channel_type,
    ):
    commu.init(instance.federal_info)

def otp_pn_fl_worker(num, param, tag, federal_info, sec_param, q):
    protocol = make_protocol(
            OTP_PN_FL,
            federal_info,
            sec_param.get(OTP_PN_FL),
            {}
    )
    result = protocol.param_negotiate(param=param, data=num, tag=tag)
    q.put(result)

def init_protocol(flex_const, federation_info, sec_param, tag):
    federal_info = deepcopy(federation_info)
    federal_info['session']['job_id'] += "_" + str(tag)

    protocol = make_protocol(
            flex_const,
            federal_info,
            sec_param.get(flex_const),
            {}
    )
    return protocol, federal_info