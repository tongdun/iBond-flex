from flex.api import make_protocol
from flex.constants import BF_SF

from test.fed_config_example import fed_conf_coordinator


def test():
    federal_info = fed_conf_coordinator
    sec_param = []

    make_protocol(BF_SF, federal_info, sec_param).intersect()
    print("done")


if __name__ == '__main__':
    test()
