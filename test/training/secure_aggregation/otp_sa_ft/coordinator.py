
from flex.api import make_protocol
from flex.constants import OTP_SA_FT

from test.fed_config_example import fed_conf_coordinator_guest12


def test():
    federal_info = fed_conf_coordinator_guest12

    sec_param = [('onetime_pad', {"key_length": 512})]

    trainer = make_protocol(OTP_SA_FT, federal_info, sec_param, None)
    result = trainer.exchange()
    print(result)


if __name__ == '__main__':
    test()
