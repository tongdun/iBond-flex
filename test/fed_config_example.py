import time


job_id = 'test_job'

# federal_info example group 1
federation = ["zhibang-d-011040", "zhibang-d-011041", "zhibang-d-011042", "zhibang-d-014031"]
fed_conf_host = {
    "server": "localhost:6001",
    "session": {
        "role": "host",
        "local_id": federation[0],
        "job_id": job_id},
    "federation": {
        "host": [federation[0]],
        "guest": [federation[1]],
        "coordinator": [federation[2]]}
}

fed_conf_guest = {
    "server": "localhost:6001",
    "session": {
        "role": "guest",
        "local_id": federation[1],
        "job_id": job_id},
    "federation": {
        "host": [federation[0]],
        "guest": [federation[1]],
        "coordinator": [federation[2]]}
}

fed_conf_coordinator = {
    "server": "localhost:6001",
    "session": {
        "role": "coordinator",
        "local_id": federation[2],
        "job_id": job_id},
    "federation": {
        "host": [federation[0]],
        "guest": [federation[1]],
        "coordinator": [federation[2]]}
}

# Group 2: guest1, guest2, coordinator
fed_conf_guest1 = {
    "server": "localhost:6001",
    "session": {
        "role": "guest",
        "local_id": federation[0],
        "job_id": job_id},
    "federation": {
        "host": [],
        "guest": [federation[0], federation[1]],
        "coordinator": [federation[2]]}
}

fed_conf_guest2 = {
    "server": "localhost:6001",
    "session": {
        "role": "guest",
        "local_id": federation[1],
        "job_id": job_id},
    "federation": {
        "host": [],
        "guest": [federation[0], federation[1]],
        "coordinator": [federation[2]]}
}

fed_conf_coordinator_guest12 = {
    "server": "localhost:6001",
    "session": {
        "role": "coordinator",
        "local_id": federation[2],
        "job_id": job_id},
    "federation": {
        "host": [],
        "guest": [federation[0], federation[1]],
        "coordinator": [federation[2]]}
}

# Group 3: guest1, guest2, guest3, coordinator
fed_conf_multiparty_guest1 = {
    "server": "localhost:6001",
    "session": {
        "role": "guest",
        "local_id": federation[0],
        "job_id": job_id},
    "federation": {
        "host": [],
        "guest": [federation[0], federation[1], federation[3]],
        "coordinator": [federation[2]]}
}

fed_conf_multiparty_guest2 = {
    "server": "localhost:6001",
    "session": {
        "role": "guest",
        "local_id": federation[1],
        "job_id": job_id},
    "federation": {
        "host": [],
        "guest": [federation[0], federation[1], federation[3]],
        "coordinator": [federation[2]]}
}

fed_conf_multiparty_guest3 = {
    "server": "localhost:6001",
    "session": {
        "role": "guest",
        "local_id": federation[3],
        "job_id": job_id},
    "federation": {
        "host": [],
        "guest": [federation[0], federation[1], federation[3]],
        "coordinator": [federation[2]]}
}

fed_conf_multiparty_coordinator = {
    "server": "localhost:6001",
    "session": {
        "role": "coordinator",
        "local_id": federation[2],
        "job_id": job_id},
    "federation": {
        "host": [],
        "guest": [federation[0], federation[1], federation[3]],
        "coordinator": [federation[2]]}
}

if __name__ == '__main__':
    print(fed_conf_host)
    print(fed_conf_guest)
    print(fed_conf_coordinator)
    print(fed_conf_guest1)
    print(fed_conf_guest2)
