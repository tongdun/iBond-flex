from flex.api import make_protocol
from flex.constants import SAL

from test.fed_config_example import fed_conf_host


def test():
    federal_info = fed_conf_host

    sec_param = [['aes', {'key_length': 128}]]

    algo_param = {}

    host_data = list(range(1000))
    print(len(host_data))

    # align
    share = make_protocol(SAL, federal_info, sec_param, algo_param)
    result = share.align(host_data)
    print(result)

    # verify
    is_align = share.verify(result)
    print(is_align)


if __name__ == '__main__':
    test()

