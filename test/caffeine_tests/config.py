import time

# job_uuid = f'test_job_{time.time()}'
job_uuid = f'test_job_open_source_12_02——1'


party_ids = ['9999', '10000', '10001', '10002', '10003', '10004']
# Group1: guest, host, coordinator
fed_conf_host = {
    "server": "localhost:6149",
    "session": {
        "role": "host",
        "local_id": party_ids[1],
        "job_id": job_uuid, },
    "federation": {
        "host": [party_ids[1]],
        "guest": [party_ids[2]],
        "coordinator": [party_ids[0]], }
}

fed_conf_guest = {
    "server": "localhost:6149",
    "session": {
        "role": "guest",
        "local_id": party_ids[2],
        "job_id": job_uuid, },
    "federation": {
        "host": [party_ids[1]],
        "guest": [party_ids[2]],
        "coordinator": [party_ids[0]], }
}

fed_conf_coordinator = {
    "server": "localhost:6149",
    "session": {
        "role": "coordinator",
        "local_id": party_ids[0],
        "job_id": job_uuid, },
    "federation": {
        "host": [party_ids[1]],
        "guest": [party_ids[2]],
        "coordinator": [party_ids[0]], }
}

# Group 2: guest1, guest2, coordinator
fed_conf_guest1_no_host = {
    "server": "localhost:6149",
    "session": {
        "role": "guest",
        "local_id": party_ids[1],
        "job_id": job_uuid, },
    "federation": {
        "host": [],
        "guest": [party_ids[1], party_ids[2]],
        "coordinator": [party_ids[0]], }
}

fed_conf_guest2_no_host = {
    "server": "localhost:6149",
    "session": {
        "role": "guest",
        "local_id": party_ids[2],
        "job_id": job_uuid, },
    "federation": {
        "host": [],
        "guest": [party_ids[1], party_ids[2]],
        "coordinator": [party_ids[0]], }
}

fed_conf_coordinator_no_host = {
    "server": "localhost:6149",
    "session": {
        "role": "coordinator",
        "local_id": party_ids[0],
        "job_id": job_uuid, },
    "federation": {
        "host": [],
        "guest": [party_ids[1], party_ids[2]],
        "coordinator": [party_ids[0]], }
}

# Group3: guest, host
fed_conf_host_no_coordinator = {
    "server": "10.58.10.60:6149",
    "session": {
        "role": "host",
        "local_id": party_ids[1],
        "job_id": job_uuid, },
    "federation": {
        "host": [party_ids[1]],
        "guest": [party_ids[2]], }
}

fed_conf_guest_no_coordinator = {
    "server": "10.58.10.60:6149",
    "session": {
        "role": "guest",
        "local_id": party_ids[2],
        "job_id": job_uuid, },
    "federation": {
        "host": [party_ids[1]],
        "guest": [party_ids[2]], }
}

# Group4: guest, host1, host2, coordinator
fed_conf_coordinator_mp = {
    "server": "localhost:6149",
    "session": {
        "role": "coordinator",
        "local_id": party_ids[0],
        "job_id": job_uuid, },
    "federation": {
        "host": [party_ids[2], party_ids[3]],
        "guest": [party_ids[1]],
        "coordinator": [party_ids[0]],
    }
}

fed_conf_guest_mp = {
    "server": "localhost:6149",
    "session": {
        "role": "guest",
        "local_id": party_ids[1],
        "job_id": job_uuid, },
    "federation": {
        "host": [party_ids[2], party_ids[3]],
        "guest": [party_ids[1]],
        "coordinator": [party_ids[0]],
    }
}

fed_conf_host0_mp = {
    "server": "localhost:6149",
    "session": {
        "role": "host",
        "local_id": party_ids[2],
        "job_id": job_uuid, },
    "federation": {
        "host": [party_ids[2], party_ids[3]],
        "guest": [party_ids[1]],
        "coordinator": [party_ids[0]],
    }
}

fed_conf_host1_mp = {
    "server": "localhost:6149",
    "session": {
        "role": "host",
        "local_id": party_ids[3],
        "job_id": job_uuid, },
    "federation": {
        "host": [party_ids[2], party_ids[3]],
        "guest": [party_ids[1]],
        "coordinator": [party_ids[0]],
    }
}

fed_conf_host_mp = {
    0: fed_conf_host0_mp,
    1: fed_conf_host1_mp
}


# Group 5: guest1, guest2, guest3, coordinator
fed_conf_guest1_no_host_mp = {
    "server": "localhost:6149",
    "session": {
        "role": "guest",
        "local_id": party_ids[1],
        "job_id": job_uuid, },
    "federation": {
        "host": [],
        "guest": [party_ids[1], party_ids[2],party_ids[3]],
        "coordinator": [party_ids[0]], }
}

fed_conf_guest2_no_host_mp = {
    "server": "localhost:6149",
    "session": {
        "role": "guest",
        "local_id": party_ids[2],
        "job_id": job_uuid, },
    "federation": {
        "host": [],
        "guest": [party_ids[1], party_ids[2],party_ids[3]],
        "coordinator": [party_ids[0]], }
}
fed_conf_guest3_no_host_mp = {
    "server": "localhost:6149",
    "session": {
        "role": "guest",
        "local_id": party_ids[3],
        "job_id": job_uuid, },
    "federation": {
        "host": [],
        "guest": [party_ids[1], party_ids[2],party_ids[3]],
        "coordinator": [party_ids[0]], }
}
fed_conf_coordinator_no_host_mp = {
    "server": "localhost:6149",
    "session": {
        "role": "coordinator",
        "local_id": party_ids[0],
        "job_id": job_uuid, },
    "federation": {
        "host": [],
        "guest": [party_ids[1], party_ids[2],party_ids[3]],
        "coordinator": [party_ids[0]], }
}
fed_conf_guest_no_host_mp ={
    0:fed_conf_guest1_no_host_mp,
    1:fed_conf_guest2_no_host_mp,
    2:fed_conf_guest3_no_host_mp

}


# Group6: guest, host
fed_conf_host_no_coordinator_2_machine = {
    "server": "localhost:6001",
    "session": {
        "role": "host",
        "local_id": party_ids[1],
        "job_id": job_uuid, },
    "federation": {
        "host": [party_ids[1]],
        "guest": [party_ids[2]], }
}

fed_conf_guest_no_coordinator_2_machine = {
    "server": "localhost:6001",
    "session": {
        "role": "guest",
        "local_id": party_ids[2],
        "job_id": job_uuid, },
    "federation": {
        "host": [party_ids[1]],
        "guest": [party_ids[2]], }
}