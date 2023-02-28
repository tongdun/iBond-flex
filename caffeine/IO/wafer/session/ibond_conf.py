"""
 Created by liwei on 2020/12/9.
"""
import json
from typing import Dict


class IBondConf:
    def __init__(self, uri: str = None, config: dict = None):
        self._conf = {}

        if uri is not None:
            with open(uri) as f:
                self._conf = json.load(f)

        if config is not None:
            self._conf.update(config)

    def __getstate__(self):
        """Return state values to be pickled."""
        return self._conf

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self._conf = state

    def __getattr__(self, item: str):
        # to avoid RecursionError cause of copy the object
        _conf = object.__getattribute__(self, "_conf")

        if item in _conf:
            value = _conf.get(item)
            if isinstance(value, Dict):
                return IBondConf(config=value)
            return value
        elif item == "value":
            return _conf
        else:
            raise AttributeError(f"Unable to get the configuration item: '{item}'")

    def __iter__(self):
        return iter(self._conf)

    def __getitem__(self, item):
        return self._conf[item]

    def __repr__(self):
        return str(self._conf)

    def set(self, key, value):
        self._conf[key] = value

    def getAll(self):
        return self._conf.items()

    @property
    def runtime_conf(self):
        conf = self._conf["workflow"][0]["parameter"]
        federal_info = {"federal_info": self._conf["communication"]}
        conf.update(federal_info)

        return IBondConf(config=conf)


if __name__ == "__main__":
    c = IBondConf("/Users/zipee/Downloads/aa.py.json")
    c.set("c", 1)
    # print(c._conf)
    for k, v in c.getAll():
        print(k, v)
    print(c.d.aa)
