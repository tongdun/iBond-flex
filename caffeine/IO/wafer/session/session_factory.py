"""
 Created by liwei on 2020/12/8.
"""
from caffeine.IO.wafer.session.light_session import LightSession
from caffeine.IO.wafer.session.moonshot_session import MoonshotSession
from caffeine.IO.wafer.session.ibond_conf import IBondConf


class SessionFactory:
    def __init__(self):
        raise NotImplementedError()

    class EngineEnum:
        SPARK = "spark"
        LIGHT = "light"
        MOONSHOT = "moonshot"

    class Builder:
        _conf = IBondConf()

        def app_name(self, name: str):
            return self.config("ibond.app.name", name)

        def config(self, key: str = None, value=None, conf: IBondConf = None):
            if conf is None:
                self._conf.set(key, str(value))
            elif isinstance(conf, IBondConf):
                for k, v in conf.getAll():
                    self._conf.set(k, v)
            return self

        def get_or_create(self):
            if self._conf.engine == SessionFactory.EngineEnum.LIGHT:
                return LightSession(self._conf)
            elif self._conf.engine == SessionFactory.EngineEnum.MOONSHOT:
                return MoonshotSession(self._conf)
            raise RuntimeError(f"Illegal engine type: {self._conf.engine}")

    builder = Builder()


if __name__ == "__main__":
    conf = IBondConf("/Users/zipee/Downloads/aa.py.json")
    session = (
        SessionFactory.builder.app_name("spark_session")
        .config(conf=conf)
        .config("spark.sql.execution.arrow.enabled", "true")
        .get_or_create()
    )

    import pandas as pd

    mydict = [
        {"a": 1, "b": 2, "c": 3, "d": 4},
        {"a": 100, "b": 200, "c": 300, "d": 400},
        {"a": 1000, "b": 2000, "c": 3000, "d": 4000},
    ]
    pdf = pd.DataFrame(mydict)
    df = session.create_dataframe(pdf)
    print(df.count())

    print(df)
