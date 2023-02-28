from enum import Enum

from flex import constants
from pydantic import BaseModel, Field
from caffeine.utils.config import security_config

Stepwise_FMC = "_".join(["Stepwise", constants.FMC])
Relief_FMC = "_".join(["Relief", constants.FMC])

more_configs = {
    Stepwise_FMC: [['secret_sharing', {"shift": 20, 'length': 64, 'p_value': 67}]],
    Relief_FMC: [['secret_sharing', {"shift": 20, 'length': 64, 'p_value': 67}]],
    constants.IV_FFS: [["paillier", {"key_length": 1024}],],
    constants.HE_DT_FB: [["paillier", {"key_length": 1024}],]
}

security_config.update(more_configs)
