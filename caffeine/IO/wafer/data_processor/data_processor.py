"""
 Created by liwei on 2020/7/6.
"""
import json
import re
from typing import Dict, List

import numpy as np

# from pyspark.sql.functions import isnan, col, mean, stddev, max, min, desc

from .. import Wafer
from .. import TableTypeEnum
from ..util.xlog import logger
from ..util.decorator import timeit
from ..util.http import post
from ..session.ibond_session import IBondSession
from ..session.ibond_conf import IBondConf
from ..dataframe.ibond_dataframe import IBondDataFrame


class DataProcessor:
    NUM_2_IMMEDIATELY_ANALYZE = 1000000

    def __init__(self, param: Dict, session: IBondSession):
        self.session = session
        self.ctx = IBondConf(config=param.get("general_param"))
        self.fed = IBondConf(config=param.get("federal_info"))

    def read(self):
        raise NotImplementedError()

    @timeit(end_msg="数据导入成功！")
    def process(self) -> None:
        read_data = self.read()
        read_data.show(5)

        data = self.do_processing(read_data)
        # header to lowercase
        # data.update_columns([col.lower() for col in data.columns])

        data.save_as_table(self.ctx.table_name)
        self.report_data_detail()

    def report_data_detail(self) -> None:
        def report(table_name, data_name):
            data_size = self.session.calc_table_size(table_name)

            url = f"{self.ctx.bond_web_server_url}/api/data/op/leftMetaData"

            data = {
                "data_name": data_name,
                "creator": self.ctx.bond_account,
                "rows_size": self.row_num,
                "data_size": data_size,
            }
            logger.info(f"report url: {url}")
            logger.info(f"report data: {json.dumps(data)}")
            post(url, json.dumps(data))

        if not self.is_A_exist and self.ctx.data_type != TableTypeEnum.A.value:
            report(self.A_table_name, self.A_data_name)
        report(self.ctx.table_name, self.ctx.data_name)

    @staticmethod
    def parse_col_desc_2_scope(desc: str) -> List:
        if not desc:
            return []

        if ":" in desc:
            head, end = desc.split(":")
            flag = 1 if int(head) < int(end) else -1
            return list(range(int(head) - 1, int(end) + flag - 1, flag))
        elif "," in desc:
            return [int(v) - 1 for v in desc.split(",")]
        return [int(desc) - 1]

    @timeit(start_msg="开始列重排", end_msg="完成列重排")
    def reorder_column(self, df: IBondDataFrame) -> (IBondDataFrame, List):
        id_desc = self.parse_col_desc_2_scope(self.ctx.primary_key)
        y_desc = self.parse_col_desc_2_scope(getattr(self.ctx, "label_key", None))

        col_names = [f"{v.get('col_name')}" for v in self.ctx.schema]
        id_names = [col_names[idx] for idx in id_desc]
        y_names = [col_names[idx] for idx in y_desc]

        non_feature_names = id_names + y_names
        feature_names = [n for n in col_names if n not in non_feature_names]
        reorder_col_names = id_names + y_names + feature_names

        logger.info(f"重排后的列顺序: {','.join(reorder_col_names)}")
        return df.select(reorder_col_names), id_names

    def ensure_group_exist(self, group_id: str) -> (str, str, str):
        url = f"{self.ctx.bond_web_server_url}/api/data/op/group/existOrCreate"

        data = {
            "data_name": self.ctx.data_name,
            "data_type": self.ctx.data_type,
            "creator": self.ctx.bond_account,
            "group_id": group_id,
        }

        result = post(url, json.dumps(data))
        return (
            result["data"]["is_exist"],
            result["data"].get("data_name"),
            result["data"].get("table_name"),
        )

    @timeit(start_msg="开始计算聚类索引", end_msg="完成计算聚类索引！")
    def calc_groupid(self, df: IBondDataFrame, id_names: List) -> str:
        df_ids = df.select(id_names)
        from hashlib import sha1

        s = sha1()
        iter = df_ids.to_local_iterator()
        for it in iter:
            m = [str(getattr(it, name)) for name in id_names]
            s.update(",".join(m).lower().encode("utf8"))
        group_id = s.hexdigest()

        logger.info(f"聚类索引为：{group_id}")
        return group_id

    def exist_or_create_group(self, df: IBondDataFrame, id_names: List) -> None:
        data_type = self.ctx.data_type
        group_id = self.calc_groupid(df, id_names)
        self.is_A_exist, self.A_data_name, self.A_table_name = self.ensure_group_exist(
            group_id
        )

        if self.is_A_exist:
            # A 存在则终止，否则继续
            if data_type == TableTypeEnum.A.value:
                raise Exception(f"数据集已存在: {self.A_data_name}")
        else:
            # B/C 存在则继续，否则先保存对应A
            if data_type != TableTypeEnum.A.value:
                self.create_group(df, self.A_table_name, id_names)

    @timeit(end_msg="聚类创建成功！")
    def create_group(self, df: IBondDataFrame, table_name: str, id_names: List) -> None:
        logger.info(f"该数据集对应聚类不存在，创建中: {self.ctx.data_name + '_key'}")
        df.select(id_names).save_as_table(table_name)

    def do_processing(self, df: IBondDataFrame) -> IBondDataFrame:
        reorder_df, id_names = self.reorder_column(df)

        self.row_num = df.count()
        logger.info(f"数据行数：{self.row_num}")
        if self.row_num >= self.NUM_2_IMMEDIATELY_ANALYZE:
            return reorder_df

        self.exist_or_create_group(reorder_df, id_names)

        replaced_pdf = self.do_data_analysis(reorder_df.to_pandas())

        return self.session.create_dataframe(replaced_pdf)

    def do_data_analysis(self, pdf: IBondDataFrame) -> IBondDataFrame:
        col_type_collection = {col["col_name"]: col["type"] for col in self.ctx.schema}

        data_analyst = DataAnalyst(pdf, col_type_collection, self)
        replaced_pdf = data_analyst.deal_with_missing_value()
        data_analyst.analyze()
        data_analyst.report(self.ctx.bond_web_server_url)
        return replaced_pdf


class DataAnalyst:
    REPORT_URL = "/api/data/op/dataAnalysisReport"

    def __init__(self, pdf, col_type_collection, data_processor):
        self.pdf = self.initialization(pdf)
        self.col_type_collection = col_type_collection
        self.data_name = data_processor.ctx.data_name
        self.creator = data_processor.ctx.bond_account
        self.data_quality = ColDataAnalyst.QUALITY_GOOD
        self.primary_data_quality = ColDataAnalyst.QUALITY_GOOD
        self.data_operator = data_processor

        self.col_analyst_collection = [
            ColDataAnalyst(item[1], col_type_collection[item[0]])
            for item in self.pdf.iteritems()
        ]

    def analyze(self):
        self.analyze_data_distribution()
        self.analyze_data_quality()

    @timeit(start_msg="开始数据分布分析", end_msg="分析完成")
    def analyze_data_distribution(self):
        for col_data_analyst in self.col_analyst_collection:
            col_data_analyst.analyze_data_distribution()

    @timeit(start_msg="开始数据质量分析", end_msg="分析完成")
    def analyze_data_quality(self):
        for col_data_analyst in self.col_analyst_collection:
            col_data_analyst.analyze_data_quality()
            if col_data_analyst.data_quality == ColDataAnalyst.QUALITY_GOOD:
                continue

            self.data_quality = col_data_analyst.data_quality
            if col_data_analyst.is_primary:
                self.primary_data_quality = col_data_analyst.data_quality

            if col_data_analyst.data_quality == ColDataAnalyst.QUALITY_POOR:
                return

    @timeit(start_msg="开始发送数据分析报告", end_msg="发送完成")
    def report(self, bond_web_server_url):
        url = f"{bond_web_server_url}{self.REPORT_URL}"
        logger.info(dict(self))
        post(url, json.dumps(dict(self)))

    def deal_with_missing_value(self):
        missing_value = getattr(self.data_operator, "missing_value", "")
        for col_data_analyst in self.col_analyst_collection:
            col = col_data_analyst.deal_with_missing_value(missing_value)
            setattr(self.pdf, col_data_analyst.col_name, col)
        return self.pdf

    def initialization(self, pdf):
        def deal_with_empty_row(pdf):
            _pdf = pdf.dropna(axis=0, how="all")
            logger.info(f"清理空行数：{len(pdf) - len(_pdf)}")
            return _pdf

        pdf = deal_with_empty_row(pdf)

        return pdf

    def __getitem__(self, item):
        if item == "data_distributions":
            return [dict(item) for item in self.col_analyst_collection]
        return getattr(self, item, "")

    @staticmethod
    def keys():
        return [
            "data_distributions",
            "data_name",
            "creator",
            "data_quality",
            "primary_data_quality",
        ]

    def __repr__(self):
        return str(dict(self))


class ColDataAnalyst:
    PRIMARY = "primary"
    LABEL = "label"
    FEATURE = "feature"
    LOSS_RATE = 0.9
    QUALITY_GOOD = "good"
    QUALITY_MEDIAM = "mediam"
    QUALITY_POOR = "poor"
    FORMAT = ".4f"
    DATA_TYPE = "float64"
    # 替换 包含字母、:、{}、空格、/、[]、'、"、,、中文字符、空字符、用户自定义的空缺符
    REPLACE_PATTERN = "[a-zA-Z:{}\s/\[\]'\",\u4e00-\u9fa5]+|^$"

    def __init__(self, col, _type):
        self.col = col
        self.col_name = self.col.name
        self.is_primary = _type == self.PRIMARY
        self.is_label = _type == self.LABEL
        self.is_feature = _type == self.FEATURE
        self.data_quality = self.QUALITY_GOOD
        self.count = len(col)

    def do_math_statistics(self):
        loss_count = self.col.isna().sum()

        if self.is_primary:
            self.is_primary_loss = loss_count > 0
            self.is_primary_duplicated = self.col.duplicated().sum() > 0
        else:
            self.unformat_max = self.col.max()
            self.unformat_min = self.col.min()
            self.max = format(self.unformat_max, self.FORMAT)
            self.min = format(self.unformat_min, self.FORMAT)
            self.average = format(self.col.mean(), self.FORMAT)
            self.median = format(self.col.median(), self.FORMAT)
            self.standard = format(self.col.std(), self.FORMAT)
            self.mode = self._calc_mode()
            self.data_distribution = []
        self.unformat_loss_rate = loss_count / len(self.col)
        self.loss_rate = format(loss_count / len(self.col), self.FORMAT)
        logger.info(self)

    def _calc_mode(self):
        # 众数个数 + nan个数 != 总行数 / 即出现次数为1，不算众数
        if (len(self.col.mode()) + self.col.isna().sum()) != self.count:
            # 最多取四个
            return ",".join([str(format(i, self.FORMAT)) for i in self.col.mode()[:4]])
        return "/"

    def deal_with_missing_value(self, missing_value):
        if self.is_primary:
            return self.col

        _missing_value = (
            missing_value.replace(",", "|") if missing_value else "not null"
        )

        pattern = re.compile(f"{self.REPLACE_PATTERN}|{_missing_value}")
        col = self.col.astype("str")
        self.col = col.replace(pattern, np.nan).astype(self.DATA_TYPE)

        logger.info(
            f"【{self.col_name}】处理空缺符数：{self.col.isna().sum() - col.isna().sum()}"
        )
        return self.col

    def analyze_data_distribution(self):
        logger.info(f"数据分布分析：{self.col_name}")
        self.do_math_statistics()

        if self.is_primary:
            return

        # 若为nan，则无需计算
        if any([self.max == "nan", self.min == "nan"]):
            return

        step = np.round((self.unformat_max - self.unformat_min) / 10, 4)
        if step == 0:
            interval = f"[{self.min}, {self.max}]"
            # count统计非nan行数, int64无法序列化，转为str
            self.data_distribution.append(
                {"interval": interval, "count": str(self.col.count())}
            )
            return

        distribution_list = list(np.arange(self.unformat_min, self.unformat_max, step))
        distribution_list.append(self.unformat_max)

        for i in range(len(distribution_list) - 1):
            left, right, unformat_left, unformat_right = (
                format(distribution_list[i], self.FORMAT),
                format(distribution_list[i + 1], self.FORMAT),
                distribution_list[i],
                distribution_list[i + 1],
            )

            if unformat_right == self.unformat_max:
                interval = f"[{left}, {right}]"
                count = str(
                    ((self.col >= unformat_left) & (self.col <= unformat_right)).sum()
                )
            else:
                interval = f"[{left}, {right})"
                count = str(
                    ((self.col >= unformat_left) & (self.col < unformat_right)).sum()
                )
            self.data_distribution.append({"interval": interval, "count": count})

    def analyze_data_quality(self):
        if all(
            [self.is_label, ((self.col == 0) | (self.col == 1)).sum() != self.count]
        ):
            self.data_quality = self.QUALITY_POOR
            logger.info(f'标签【{self.col.name}】存在非(0,1)值，故质量评价为"差"')
            return

        if self.is_feature and self.unformat_loss_rate >= self.LOSS_RATE:
            logger.info(f'特征【{self.col.name}】缺失值比例大于等于90%，故质量评价为"中"')
        elif self.is_primary and self.is_primary_loss:
            logger.info(f'主键【{self.col.name}】中出现缺失值（空字符)，故质量评价为"中"')
        elif self.is_primary and self.is_primary_duplicated:
            logger.info(f'主键【{self.col.name}】中存在重复值，故质量评价为"中"')
        else:
            return

        self.data_quality = self.QUALITY_MEDIAM

    def __getitem__(self, item):
        if item == "data_distribution":
            return json.dumps(getattr(self, item, []))
        return getattr(self, item, "")

    @staticmethod
    def keys():
        return [
            "max",
            "min",
            "average",
            "median",
            "standard",
            "mode",
            "data_distribution",
            "loss_rate",
            "col_name",
        ]

    def __repr__(self):
        return str(dict(self))
