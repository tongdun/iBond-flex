"""
 Created by liwei on 2021/1/14.
"""
from pathlib import Path
import json
import time

import requests

from .xlog import logger


class Reporter:
    report_api = "/api/monitor/operator_data/add"
    headers = {"Content-Type": "application/json"}

    def __init__(self, bond_web_server_url, report_id):
        self.bypass = False
        if bond_web_server_url is None or report_id is None:
            # Do nothing if config is empty.
            logger.warn(
                f"bond_web_server_url and report_id are not set, thus reporter is disabled. Check your config if this is not what you want."
            )
            self.bypass = True
        else:
            self.url = bond_web_server_url + self.report_api
            self.monitor_id = report_id

    def insert(self, index_code: str, index_value: str):
        if self.bypass:
            # Do nothing
            pass
        else:
            data = {
                "index_code": index_code,
                "index_value": index_value,
                "monitor_id": self.monitor_id,
                "timestamp": int(time.time() * 1000),
            }
            try:
                return requests.post(
                    self.url, json.dumps(data), headers=self.headers, timeout=5
                )
            except Exception:
                logger.warn("report failed..")


class FileReporter:
    headers = {"Content-Type": "application/json"}

    def __init__(self, reports_dir):
        self._reports_dir = Path(reports_dir)
        try:
            self._reports_dir.mkdir(parents=True)
        except:
            pass

        
    def report_path(self, url, keys):
        tmp = []
        for key in keys:
            tmp.append({
                'filepath': str(self._reports_dir),
                'filename': key,
                'tag': key
            })
        return requests.post(url, json.dumps(tmp), headers=self.headers)

    # def report_path(self, destination, keys):
    #     tmp = []
    #     for key in keys:
    #         tmp.append({
    #             'filepath': str(self._reports_dir),
    #             'filename': key,
    #             'tag': key
    #         })
    #     with Path(destination).open(mode='w') as f:
    #         json.dump(tmp, f)


    def insert(self, key: str, val: str):
        key = key+'.json'
        report_file = self._reports_dir.joinpath(key)
        with report_file.open(mode='a') as f:
            # json.dump(val, f)
            f.write(val + '\n')

