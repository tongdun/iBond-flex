
from flex.api import make_protocol
from flex.constants import SAL

from test.fed_config_example import fed_conf_guest

def test():
    federal_info = fed_conf_guest

    sec_param = {
        "key_exchange_size": 2048
    }

    guest_data = list(map(str, range(100, 200)))

    #对齐
    share = make_protocol(SAL,  federal_info, sec_param)
    result = share.align(guest_data)
    print(result)

    # 验证
    is_align = share.verify(result)
    print(is_align)


if __name__ == '__main__':
    test()
