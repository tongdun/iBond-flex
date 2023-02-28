import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from flex.ionic_bond import commu
from flex.ionic_bond.channel import VariableChannel

commu.LocalTest = True


def host_work():
    config = {
        "server": "localhost:6001",
        "session": {
            "role": "host",
            "local_id": "zhibang-d-014010",
            "job_id": 'test_job',
        },
        "federation": {
            "host": ["zhibang-d-014010"],
            "guest": ["zhibang-d-014011"],
            "coordinator": ["zhibang-d-014012"]
        }
    }
    commu.init(config)
    remote_id = "zhibang-d-014011"
    print("aaaa")
    var_chan = VariableChannel(name="Exchange_single_variable",
                               remote_id=remote_id)

    for _ in range(10):
        my_var = np.random.random((10, 10))
        var_chan.send(my_var)
        remote_var = var_chan.recv()
        assert np.all(remote_var == my_var)


def guest_work():
    config = {
        "server": "localhost:6001",
        "session": {
            "role": "guest",
            "local_id": "zhibang-d-014011",
            "job_id": 'test_job',
        },
        "federation": {
            "host": ["zhibang-d-014010"],
            "guest": ["zhibang-d-014011"],
            "coordinator": ["zhibang-d-014012"]
        }
    }
    commu.init(config)
    remote_id = "zhibang-d-014010"
    var_chan = VariableChannel(name="Exchange_single_variable",
                               remote_id=remote_id)
    for _ in range(10):
        var_chan.send(var_chan.recv())


def test_local_ion():
    ROLE = os.getenv('ROLE')
    if ROLE == 'HOST':
        host_work()
    if ROLE == 'GUEST':
        guest_work()
