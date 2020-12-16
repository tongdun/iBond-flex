# federal_info example group 1
federation = ["zhibang-d-014010", "zhibang-d-014011", "zhibang-d-014012"]
fed_conf_host = {
    "server": "localhost:6001",
    "session": {
        "role": "host",
        "local_id": federation[0],
        "job_id": 'test_job', },
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
        "job_id": 'test_job', },
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
        "job_id": 'test_job', },
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
        "job_id": 'test_job', },
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
        "job_id": 'test_job', },
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
        "job_id": 'test_job', },
    "federation": {
        "host": [],
        "guest": [federation[0], federation[1]],
        "coordinator": [federation[2]]}
}

if __name__ == '__main__':
    print(fed_conf_host)
    print(fed_conf_guest)
    print(fed_conf_coordinator)
    print(fed_conf_guest1)
    print(fed_conf_guest2)
