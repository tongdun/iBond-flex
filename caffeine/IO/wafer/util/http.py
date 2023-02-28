"""
 Created by liwei on 2020/12/10.
"""
import json

import requests

from caffeine.IO.wafer.util.xlog import logger


def post(url, data, files=None, headers=None, timeout=5):
    headers = headers if headers else {"Content-Type": "application/json"}
    try:
        res = requests.post(url, data, files=files, headers=headers, timeout=timeout)
    except ConnectionRefusedError as e:
        logger("服务连接失败！")
        raise e
    except Exception as e:
        raise e

    if res.status_code != 200:
        raise RuntimeError(f"服务访问异常：{res.text}")

    result = json.loads(res.text)
    if not result["success"]:
        raise RuntimeError(f"服务调用失败：{result['message']}")

    return result


def get(url, data, timeout=5, bytes_io=False):
    try:
        res = requests.get(url, data, timeout=timeout)
    except ConnectionRefusedError as e:
        logger("服务连接失败！")
        raise e
    except Exception as e:
        logger("http异常！")
        raise e

    if res.status_code != 200:
        raise RuntimeError(f"服务访问异常：{res.text}")

    if bytes_io:
        result = res.content
    else:
        result = json.loads(res.text)
        if not result["success"]:
            raise RuntimeError(f"服务调用失败：{result['message']}")

    return result
