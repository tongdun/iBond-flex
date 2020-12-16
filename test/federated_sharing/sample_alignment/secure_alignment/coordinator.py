
from flex.api import make_protocol
from flex.constants import SAL

from test.fed_config_example import fed_conf_coordinator


def test():
    federal_info = fed_conf_coordinator
    sec_param = {
        "key_exchange_size": 2048
    }

    #对齐
    share = make_protocol(SAL, federal_info, sec_param)
    share.align()

    #验证
    share.verify()


if __name__ == '__main__':
    test()
