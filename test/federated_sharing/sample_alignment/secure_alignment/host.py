
from flex.api import make_protocol
from flex.constants import SAL

from test.fed_config_example import fed_conf_host


def test():
    federal_info = fed_conf_host
    sec_param = {
        "key_exchange_size": 2048
    }

    host_data = list(map(str, range(160)))

    #对齐
    share = make_protocol(SAL, federal_info, sec_param)
    result = share.align(host_data)
    print(result)

    #验证
    is_align = share.verify(result)
    print(is_align)


if __name__ == '__main__':
    test()
