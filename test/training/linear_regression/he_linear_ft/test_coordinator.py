from flex.api import make_protocol
from flex.constants import HE_LINEAR_FT

from test.fed_config_example import fed_conf_coordinator

def test_he_linear_ft():
    federal_info = fed_conf_coordinator

    sec_param = [['paillier', {"key_length": 1024}], ]

    trainer = make_protocol(HE_LINEAR_FT, federal_info, sec_param, algo_param=None)
    trainer.exchange()

