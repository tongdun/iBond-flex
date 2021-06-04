from flex.crypto.oblivious_transfer.api import make_ot_protocol
from flex.cores.commu_model import commu
from test.fed_config_example import fed_conf_guest

if __name__ == '__main__':
    federal_info = fed_conf_guest

    commu.init(federal_info)

    ot_protocol = make_ot_protocol(1, 10, 'zhibang-d-011040')
    msg = [str(i) for i in range(10)]
    ot_protocol.server(msg)
