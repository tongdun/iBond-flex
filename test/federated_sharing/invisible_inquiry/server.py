
from flex.api import make_protocol
from flex.constants import OT_INV

from test.fed_config_example import fed_conf_host


def test():
    fed_conf_host['session']['identity'] = 'server'
    federal_info = fed_conf_host

    sec_param = {
        "symmetric_algo": "aes",
    }

    algo_param = {
        'n': 10,
        'k': 1
    }

    protocol = make_protocol(OT_INV, federal_info, sec_param, algo_param)

    def query_fun(in_list):
        result = [str(int(i) * 100) for i in in_list]
        return result

    protocol.exchange(query_fun)


if __name__ == '__main__':
    test()
