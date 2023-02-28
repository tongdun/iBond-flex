from flex.cores.commu_model import commu
from flex.crypto.key_exchange.api import make_agreement
from flex_test.fed_config_example import fed_conf_host


if __name__ == '__main__':
    # inits communication
    commu.init(fed_conf_host)
    remote_id = ["zhibang-d-011040", "zhibang-d-011041", "zhibang-d-011042"]
    local_id = "zhibang-d-011040"
    # n-party key exchange
    k = make_agreement(remote_id=remote_id, local_id=local_id, key_length=2048)
    print(k)

