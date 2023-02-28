from flex.crypto.oblivious_transfer.api import make_ot_protocol
from flex.cores.commu_model import commu
from flex_test.fed_config_example import fed_conf_host

if __name__ == '__main__':
    federal_info = fed_conf_host

    commu.init(federal_info)

    ot_protocol = make_ot_protocol(1, 10, 'zhibang-d-011041')
    msg = ot_protocol.client(index=5)
    print(msg)
