from typing import List, Dict, Optional, Union

import pandas as pd 
import numpy as np


class FeatureDataFrame(object):
    def __init__(self, process_statistics: pd.DataFrame, data_desc: Optional[Dict]=None):
        """
        Create FeatureDataFrame.

        Args:
            pdf: pandas dataframe, is ibond df inner data.
            data_desc: dict, keys are fixed strings to describe the data , values are the column names, e.g. 
                {'id': ['ID'], 'y': ['Label']},
            data_attrib: pandas dataframe, contains is_category, is_fillna and statistics.
        """
        self._attrib = process_statistics
        self._desc = []
        if data_desc is not None:
            self._desc.extend(data_desc["id_desc"])
            self._desc.extend(data_desc["y_desc"])

        self.bin_dict = dict()
        self.iloc = Ilocater(self)
        self.report_dict = dict()

    def sort(self, sort_key: str, ascending=False):
        """
        Sort _attrib by key.

        Args:
            sort_key: default ascending is False.
        """
        self._attrib.sort_values(by=sort_key, ascending=ascending, inplace=True)


    def update(self, update_key: str, thres: np.double, down_feature_num: int, ascending=False):
        """
        Update _attrib by update_key according to thres and down_feature_num.

        Args:
            update_key: str, will update _attrib by the key.
            thres: np.double, threshold for the update_key.
            down_feature_num: int, the mininum number of features reserved.
        """

        self.sort(update_key, ascending)
        if self.shape > down_feature_num:
            idx_list = self._attrib[self._attrib[update_key] >= thres].index.tolist()
            if len(idx_list) < down_feature_num:
                self._attrib = self._attrib.iloc[:down_feature_num, :]
            else:
                self._attrib = self._attrib[self._attrib.index.isin(idx_list)]
        self._attrib.reset_index(drop=True, inplace=True)

    def __getitem__(self, i: int):
        """
        Generate column data attrib lines.

        Args:
            index.

        Return:
            data column: pd.Series.
            is_category: bool, True -- continuous; False -- discrete.
            is_fillna: bool, True -- contains null data; False -- no null data.
        """
        line = self._attrib.iloc[i]
        return line[0], bool(line[1]), bool(line[2])

    def __setitem__(self, key: str, values: np.ndarray):
        self._attrib[key] = values

    def update_bin(self,info_dict: dict):
        for s in list(info_dict.keys()):
            if s not in self.names:
                info_dict.pop(s)
        self.bin_dict.update(info_dict)

    def select(self, key: str):
        return self._attrib[key]
    
    def select_by_name(self, name_list: List):
        self._attrib = self._attrib[self._attrib.name.isin(name_list)].reset_index(drop=True)
        return self

    def add_report(self, name: str, d: dict):
        self.report_dict.update({name: d})

    def get_report(self):
        return self.report_dict

    @property
    def bin_info(self):
        return self.bin_dict

    def to_pandas(self):
        return self._attrib

    def get_attrib(self):
        return self._attrib[['name', 'is_category', 'is_fillna']]

    @property
    def shape(self) -> int:
        return self._attrib.shape[0]

    @property
    def get_bin_info_len(self) -> int:
        return len(self.bin_dict)

    @property
    def names(self):
        return self._attrib['name'].tolist()

    @property
    def columns(self):
        return self._attrib.columns.tolist()

    @property
    def feature_cols(self) -> List:
        return self._desc + self.names

    def append(self, pdf: pd.DataFrame):
        self._attrib.append(pdf, ignore_index=True)


class Ilocater:
    def __init__(self, df):
        self.df = df

    def __getitem__(self, x: Union[int, list, slice]) -> FeatureDataFrame:
        self.df._attrib = self.df._attrib.iloc[x].reset_index(drop=True)
        return self.df