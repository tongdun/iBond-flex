from flex.constants import HE_OTP_LR_FT2
from flex.api import make_protocol

from flex_test.fed_config_example import fed_conf_coordinator


def test_he_otp_lr_ft2():
    federal_info = fed_conf_coordinator

    sec_param = [['paillier', {"key_length": 1024}], ]

    trainer = make_protocol(HE_OTP_LR_FT2, federal_info, sec_param)

    trainer.exchange()


