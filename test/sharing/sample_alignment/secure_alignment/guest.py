from flex.api import make_protocol
from flex.constants import SAL

from test.fed_config_example import fed_conf_guest


def test():
    federal_info = fed_conf_guest

    sec_param = [['aes', {'key_length': 128}]]

    algo_param = {}

    guest_data = list(range(500, 1500))
    print(len(guest_data))

    # align
    share = make_protocol(SAL,  federal_info, sec_param, algo_param)
    result = share.align(guest_data)
    print(result)

    # verify
    is_align = share.verify(result)
    print(is_align)


if __name__ == '__main__':
    test()
