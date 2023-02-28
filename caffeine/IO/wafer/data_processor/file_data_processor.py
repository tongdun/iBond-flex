"""
 Created by liwei on 2020/2/25.
"""
from .data_processor import DataProcessor
from ..util.xlog import logger
from ..util.decorator import timeit
from ..dataframe.ibond_dataframe import IBondDataFrame


class FileDataProcessor(DataProcessor):
    class DataFormat:
        CSV = "csv"
        PARQUET = "parquet"

    @timeit(end_msg="读取文件数据成功！")
    def read(self) -> IBondDataFrame:
        uri = self.ctx.uri
        schema = self.ctx.schema
        format = self.ctx.format

        logger.info(f"列分隔符: {self.ctx.col_delimiter}")
        logger.info(f"表字段信息: {schema}")
        logger.info(f"列长度: {len(schema)}")
        logger.info(f"编码格式: {self.ctx.encoding}")
        logger.info(f"主字段描述: {self.ctx.primary_key}")
        logger.info(f"标签字段描述: {self.ctx.label_key}")
        logger.info(f"数据集类型: {self.ctx.data_type}")
        logger.info(f"开始读取文件数据, 文件路径{uri}")
        logger.info(f"文件格式为：{format}")

        df = None
        if self.DataFormat.CSV == format:
            df = self.session.read_csv(uri)
        elif self.DataFormat.PARQUET == format:
            df = self.session.read_parquet(uri)

        if not df:
            logger.error("数据导入失败！")
            raise RuntimeError("数据导入失败!")

        return df
