from flex.api import make_protocol
from flex.constants import OT_INV

from flex_test.fed_config_example import fed_conf_guest


def test():
    fed_conf_guest['session']['identity'] = 'client'
    federal_info = fed_conf_guest

    sec_param = [['aes', {'key_length': 128}], ['ot', {'n': 10, 'k': 1}]]

    algo_param = {}

    protocol = make_protocol(OT_INV, federal_info, sec_param, algo_param)

    import random

    def obfuscator(in_list, n):
        fake_list = [random.randint(0, 100) for i in range(n - len(in_list))]
        index = random.randint(0, n - 1)
        joint_list = fake_list[:index] + in_list + fake_list[index:]
        return joint_list, index

    result = protocol.exchange('50', obfuscator)
    print(result)


if __name__ == '__main__':
    test()
