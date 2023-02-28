"""
 Created by liwei on 2021/1/13.
"""
import json
from typing import Dict
import uuid
from datetime import datetime


def save_model(model: Dict, prefix: str = "") -> str:
    """
    Mock save json model. Save the json to current directory.

    Args:
        model, dict, the dict representation of a model.
        prefix, str, prefix of the model_id.

    Returns:
        str, the model id.

    Example:
    >>> model_id = save_model(json.dump(model.dict()))
    >>> print(model_id)
    12345abcde
    """
    model_id = (
        f'{prefix}_{uuid.uuid1().hex}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
    )
    with open(f"{model_id}.json", "wb") as f:
        f.write(json.dump(model))
    return model_id


def load_model(model_id: str) -> Dict:
    """
    Load model by model id.

    Args:
        model_id, str, the model id of the model to load.

    Returns:
        bytes, model bson bytes

    Example:
    >>> model_json = load_model(model_id)
    """
    with open(f"{model_id}.json", "rb") as f:
        b = f.read()
    return json.load(b)
