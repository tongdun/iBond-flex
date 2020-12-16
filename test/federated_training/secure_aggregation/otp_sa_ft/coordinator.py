
from flex.api import make_protocol
from flex.constants import OTP_SA_FT

from test.fed_config_example import fed_conf_coordinator


def test():
    federal_info = fed_conf_coordinator

    sec_param = {
        "key_exchange_size": 2048
    }

    trainer = make_protocol(OTP_SA_FT, federal_info, sec_param)
    result = trainer.exchange()
    print(result)


if __name__ == '__main__':
    test()
