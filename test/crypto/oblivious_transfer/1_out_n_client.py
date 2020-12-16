from flex.crypto.oblivious_transfer.api import make_ot_protocol
from flex.tools.ionic import commu


if __name__ == '__main__':
    federal_info = {
        "server": "localhost:6001",
        "session": {
            "role": "host",
            "local_id": "zhibang-d-014010",
            "job_id": 'test_job',
        },
        "federation": {
            "host": ["zhibang-d-014010"],
            "guest": ["zhibang-d-014011"],
        }
    }

    commu.init(federal_info)

    ot_protocol = make_ot_protocol(1, 10, 'zhibang-d-014011')
    msg = ot_protocol.client(index=5)
    print(msg)
