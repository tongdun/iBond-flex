#!/usr/bin/python3
#
#  _____________             _________   ___       __      ________
#  ___(_)__  __ )__________________  /   __ |     / /_____ ___  __/____________
#  __  /__  __  |  __ \_  __ \  __  /    __ | /| / /_  __ `/_  /_ _  _ \_  ___/
#  _  / _  /_/ // /_/ /  / / / /_/ /     __ |/ |/ / / /_/ /_  __/ /  __/  /
#  /_/  /_____/ \____//_/ /_/\__,_/      ____/|__/  \__,_/ /_/    \___//_/
#
#  Copyright 2020 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#
#  Project name: iBond Wafer
#  File name: light_dataframe.py
#  Created by liwei on 2020/12/8.


import os
from typing import Optional, Union, Dict, List, Tuple
from pathlib import Path
from copy import deepcopy

import pandas as pd
import numpy as np

from .ibond_dataframe import IBondDataFrame
from ..session.ibond_conf import IBondConf
from ..util.xlog import logger
from ..util.hooks import Hook


class Ilocater:
    def __init__(self, df):
        self.df = df

    def __getitem__(self, x: Union[int, list, slice]) -> IBondDataFrame:
        return self.df.derive(self.df._pdf.iloc[x])


class LightDataFrame(IBondDataFrame):
    DATASET_DIR = "dataset"
    DATAIO_DIR = "dataio"

    def __init__(
        self,
        pdf: pd.DataFrame,
        data_desc: Dict[str, List] = None,
        feat_desc: Dict = None,
        conf: IBondConf = None,
    ):
        super().__init__(pdf, data_desc, feat_desc, conf)
        self.iloc = Ilocater(self)

        if pdf is not None:
            self._pdf = pdf
            # NOTE filter data_desc
            columns = set(self._pdf.columns)
            for column_type, column_names in self._data_desc.items():
                len_names = len(column_names)
                intersected = set(column_names).intersection(columns)
                # NOTE keep order
                self._data_desc[column_type] = [c for c in column_names if c in intersected]
                if len(self._data_desc[column_type]) < len_names:
                    logger.warn(f"data desc {column_type} contains unknown name.")
            # NOTE filter feat_desc
            for name in list(self._feat_desc.keys()):
                if name not in columns:
                    del self._feat_desc[name]

        else:
            self._pdf = pd.DataFrame()

    def __repr__(self):
        msg = ""
        msg += "LightDataFrame\n"
        msg += "data_desc:\n"
        msg += f"{self._data_desc}\n"
        msg += "data:\n"
        msg += f"{self._pdf}\n"
        return msg

    def __getitem__(self, x: Union[List[str], str]) -> IBondDataFrame:
        """
        Get columns by their name and generate a new IBondDataFrame from it.

        Args:

        Returns:
            IBondDataFrame
        """
        if isinstance(x, str):
            updated_data_desc = {
                k: [x] if x in v else [] for k, v in self._data_desc.items()
            }
            updated_feat_desc = {
                k: v for k, v in self._feat_desc.items() if k == x
            }
            # This will keep self._pdf[x] returns a DataFrame instead of a Series.
            return self.derive(self._pdf[[x]], updated_data_desc, updated_feat_desc)

        elif isinstance(x, slice):
            return self.derive(self._pdf[x])

        elif isinstance(x, list):
            updated_data_desc = deepcopy(self.data_desc)
            col_set = set(x)
            for k, column_names in updated_data_desc.items():
                intersected = col_set.intersection(set(column_names))
                updated_data_desc[k] = [c for c in column_names if c in intersected]
            updated_feat_desc = {
                k: v for k, v in self._feat_desc.items() if k in x
            }
            return self.derive(self._pdf[x], updated_data_desc, updated_feat_desc)

        else:
            raise TypeError(f"Type {type(x)} is not supported yet.")

    def derive(
        self, pdf: pd.DataFrame, data_desc: Dict[str, List] = None,
        feat_desc: Dict = None
    ) -> "LightDataFrame":
        """
        Derive a new lightdataframe
        """
        if data_desc is None:
            data_desc = self._data_desc
        if feat_desc is None:
            feat_desc = self._feat_desc            
        return LightDataFrame(pdf, data_desc, feat_desc, self.conf)

    def __setitem__(self, x: Union[str, list], value):
        """
        Set a column with new value
        """
        if isinstance(x, str):
            self._pdf[x] = value
        elif isinstance(x, list):
            assert len(x) == value.shape[1]
            df = pd.DataFrame(data=value, columns=x, index=self._pdf.index)
            self._pdf = pd.concat([self._pdf, df], axis=1)


    def __iter__(self):
        return self._pdf.itertuples()

    def __get_mapped_column(
        self, mapped_name: str, first_only: bool = False
    ) -> IBondDataFrame:
        column_name = self._data_desc.get(mapped_name, [])

        if first_only and len(column_name) > 1:
            column_name = column_name[0]

        column = self.__getitem__(column_name)

        return column

    def update_columns(self, columns):
        self._pdf.columns = columns

    def get_id(self, *args, **kwargs) -> IBondDataFrame:
        return self.__get_mapped_column("id_desc", *args, **kwargs)

    def get_y(self, *args, **kwargs) -> IBondDataFrame:
        return self.__get_mapped_column("y_desc", *args, **kwargs)

    def get_time(self, *args, **kwargs) -> IBondDataFrame:
        return self.__get_mapped_column("time", *args, **kwargs)

    def get_pred(self, *args, **kwargs) -> IBondDataFrame:
        return self.__get_mapped_column("pred", *args, **kwargs)

    def get_local_pred(self, *args, **kwargs) -> IBondDataFrame:
        return self.__get_mapped_column("local_pred", *args, **kwargs)

    def get_other(self, *args, **kwargs) -> IBondDataFrame:
        return self.__get_mapped_column("other", *args, **kwargs)

    def has_y(self) -> bool:
        return self._has("y_desc")

    def has_id(self) -> bool:
        return self._has("id_desc")

    def _has(self, column_type: str) -> bool:
        """
        Check if dataframe has column type column_type.
        """
        if column_type not in self._data_desc:
            return False

        return len(self._data_desc[column_type]) > 0

    @property
    def data_desc(self) -> Dict[str, List[str]]:
        return self._data_desc

    @property
    def feat_desc(self) -> Dict:
        return self._feat_desc

    @property
    def prediction(self):
        key = self._data_desc["pred"][0]
        return self._pdf[key].values[0]

    @property
    def shape(self) -> Tuple[int]:
        return self._pdf.shape

    @property
    def columns(self):
        return self._pdf.columns

    def select(self, selection: list) -> "LightDataFrame":
        return self[selection]

    def _save_to_file(self, fpath: str):
        dir_path = Path(fpath).parent
        dir_path.mkdir(parents=True, exist_ok=True)
        self._pdf.to_parquet(fpath)

    def save_as_table(self, table_name: str):
        if isinstance(self._pdf.columns, pd.RangeIndex):
            self._pdf.columns = self._pdf.columns.astype(str)
        dest_path = os.path.join([self.conf.warehouse, self.DATASET_DIR, table_name])
        self._save_to_file(dest_path)

    def save_as_tmp_table(self, table_name: str):
        if isinstance(self._pdf.columns, pd.RangeIndex):
            self._pdf.columns = self._pdf.columns.astype(str)
        dest_path = os.path.join(
            [self.conf.warehouse,
            self.DATAIO_DIR,
            self.conf.bulletin.bond_dag_task_uuid,
            self.conf.bulletin.name,
            table_name]
        )
        self._save_to_file(dest_path)

    def save_as_file(self, path: str, file_type: str = 'parquet'):
        """
        Save dataframe as file.

        Args:
            path: str, abolute path to save, e.g. '/tmp/whatever'.
            file_type: str, file type to save, should be in ['parquet', 'csv',
                'pickle'], default to 'paquent'.
        """
        # check file type
        legal_file_types = ['parquet', 'csv', 'pickle']
        if file_type not in legal_file_types:
            logger.error(f'Unsupported file type in save_as_file: {file_type}')

        # try make parent dir
        try:
            dir_path = Path(path).parent
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f'Failed to make parent dir, {e}.')

        # save to path
        save_path = '.'.join([path, file_type])
        if file_type == 'parquet':
            self._pdf.to_parquet(save_path)
        elif file_type == 'csv':
            self._pdf.to_csv(save_path, index=False)
        elif file_type == 'pickle':
            self._pdf.to_pickle(save_path)

    def to_pandas(self) -> pd.DataFrame:
        return self._pdf

    def toSeries(self, index: Optional[str] = None) -> pd.Series:
        if index is None:
            assert self._pdf.shape[1] == 1
            return self._pdf.iloc[:, 0]
        else:
            return self._pdf[index]

    def to_numpy(self) -> np.ndarray:
        return self._pdf.values

    def drop(self, columns):
        if isinstance(columns, str):
            exclude_cols = set([columns])
        else:
            exclude_cols = set(columns)

        updated_data_desc = deepcopy(self.data_desc)
        for k, column_names in updated_data_desc.items():
            diff_cols = set(column_names).difference(exclude_cols)
            updated_data_desc[k] = [c for c in column_names if c in diff_cols]

        updated_feat_desc = {
            k: v for k, v in self._feat_desc.items() if k not in columns
        }
        return self.derive(self._pdf.drop(columns=columns), updated_data_desc, updated_feat_desc)

    def count(self):
        return self._pdf.shape[0]

    def join(
        self, other: IBondDataFrame, key=None, 
        how='inner', lsuffix='', rsuffix='', sort=False) -> IBondDataFrame:
        """
        :param other: ibond dataframe.
        :param key: join using the key columns to join, should be in both two dataframes.
        :param how: {'left', 'right', 'outer', 'inner'}, default 'inner'.
        :param lsuffix: Suffix to use from left frame’s overlapping columns.
        :param rsuffix: Suffix to use from right frame’s overlapping columns.
        :param sort: Order result DataFrame lexicographically by the join key, default False.

        :return: combined ibond dataframe.
        """
        other_pdf = getattr(other, "_pdf")
        assert other_pdf is not None, "the type of other must be LightDataFrame.."

        if key is not None:
            df = self._pdf.join(
                other_pdf.set_index(key), 
                on=key, how=how, lsuffix=lsuffix, rsuffix=rsuffix, sort=sort
            ) 
        else:
            df = self._pdf.join(
                other_pdf, 
                on=key, how=how, lsuffix=lsuffix, rsuffix=rsuffix, sort=sort
            )

        columns = df.columns.tolist()
        updated_data_desc = deepcopy(self.data_desc)
        updated_feat_desc = deepcopy(self.feat_desc)
        updated_other_data_desc = deepcopy(other.data_desc)
        updated_other_feat_desc = deepcopy(other.feat_desc)

        # update each data_desc and feat_desc
        if lsuffix != '':
            for k, v in self.data_desc.items():
                updated_data_desc[k] = [v1 + lsuffix if v1 + lsuffix in columns else v1 for v1 in v]
            for k in list(updated_feat_desc.keys()):
                if k not in columns and k+lsuffix in columns:
                    updated_feat_desc.update({k+lsuffix: updated_feat_desc[k]})
                    del updated_feat_desc[k]

        if rsuffix != '':
            for k, v in other.data_desc.items():
                updated_other_data_desc[k] = [v1 + rsuffix if v1 + rsuffix in columns else v1 for v1 in v]
            for k in list(updated_other_feat_desc.keys()):
                if k not in columns and k + rsuffix in columns:
                    updated_other_feat_desc.update({k + rsuffix: updated_other_feat_desc[k]})
                    del updated_other_feat_desc[k]

        # combine data_desc
        data_desc = {
            k: updated_data_desc[k]+updated_other_data_desc[k] for _, k in enumerate(updated_data_desc)
        }
        for _, k in enumerate(updated_data_desc):
            updated_data_desc[k] += updated_other_data_desc[k]
        for k, v in data_desc.items():
            uniq_v = list(set(v))
            uniq_v.sort(key=v.index)
            data_desc[k] = uniq_v
        # combine feat_desc
        updated_feat_desc.update(updated_other_feat_desc)

        return self.derive(df, data_desc, updated_feat_desc)


    def as_type(self, col: Union[list, str], new_type):
        """
        Change type of columns.
        """
        self._pdf[col] = self._pdf[col].astype(new_type)

    def isin(self, col: str, vals: np.ndarray) -> "LightDataFrame":
        """
        :param other:
        :param data_desc: default use other's data_desc
        :return:
        """
        df = self._pdf[self._pdf[col].isin(vals)]
        return self.derive(df, self.data_desc, self.feat_desc)

    def scale(self, fea_cols:list, scale_weight: np.ndarray) -> "LightDataFrame":
        """
        Scale columnwise, size of fea_cols and scale_weight should match.
        """
        assert len(fea_cols) == len(scale_weight)
        pdf = self._pdf.copy()
        pdf[fea_cols] = pdf[fea_cols] / scale_weight
        return self.derive(pdf, self.data_desc, self.feat_desc)

    def replace(self, missing_values: list, fill_num=np.nan) -> "LightDataFrame":
        """
        Replace missing values.
        """
        self._pdf.replace(missing_values, fill_num, inplace=True)

    def sort_key(self, key:Union[List[str], str]) -> "LightDataFrame":
        """
        Sort dataframe by key.
        """
        pdf = self._pdf.sort_values(by=key).reset_index(drop=True)
        return self.derive(pdf, self.data_desc, self.feat_desc)

    def drop_duplicates(self, subset=Union[List[str], str]) -> "LightDataFrame":
        """
        Drop duplicates by subset.
        """
        pdf = self._pdf.drop_duplicates(subset=subset)
        return self.derive(pdf, self.data_desc, self.feat_desc)


    def sample(self, frac: Union[float, dict], seed: Optional[int] = None, \
                label_name=None, srs=None) -> "LightDataFrame":
        """
        Sample by fraction, fraction can be float or dict.
        If float, random sampling by frac; if dict, stratified sampling.

        :param frac: fraction can be float or dict.
        :param seed: random seed.
        :param label_name: label name.
        :param srs: sampling function.
        """
        if isinstance(frac, float):
            return self.derive( 
                        self._pdf.sample(frac=frac, random_state=seed),
                    )
        elif isinstance(frac, dict):
            if srs is not None:
                label = label_name if label_name is not None else self.y_desc[0]
                return self.derive(
                    self._pdf.groupby(label, group_keys=False).apply(srs, frac)
                )

    def to_local_iterator(self):
        return self._pdf.itertuples()

    def show(self, n):
        self._pdf.head(n)

    def shuffle(self, seed: Optional[int] = None):
        self._pdf = self._pdf.sample(frac=1, random_state=seed)

    def batches(self, batch_size: int, drop_last: bool = False, max_num: int = None):
        assert (
            batch_size >= 1
        ), f"batch_size must be larger than 1. batch_size={batch_size}"

        idxs = range(0, self.shape[0], batch_size)
        iteration_num = len(idxs)
        self._context["iteration_num"] = iteration_num
        [hook.pre_hook(self._context) for hook in self._hooks]

        for iteration_id, idx in enumerate(idxs):
            if (max_num is not None) and (iteration_id >= max_num):
                break

            border = idx + batch_size
            if drop_last and border > self.shape[0]:
                break

            logger.info(f"Starting batch {iteration_id+1}/{iteration_num}")
            yield LightDataFrame(
                self._pdf.iloc[idx:border, :], self._data_desc, self._feat_desc, self.conf
            )

            self._context["current_iteration"] = iteration_id + 1
            [hook.iter_hook(self._context) for hook in self._hooks]

        [hook.post_hook(self._context) for hook in self._hooks]

    def dummy_batches(self, num_batches):
        self._context["iteration_num"] = num_batches
        [hook.pre_hook(self._context) for hook in self._hooks]

        for iteration_id in range(num_batches):
            yield None

            self._context["current_iteration"] = iteration_id + 1
            [hook.iter_hook(self._context) for hook in self._hooks]

        [hook.post_hook(self._context) for hook in self._hooks]

    def register_hooks(self, hooks: List[Hook]):
        self._hooks = hooks

    def set_context(self, name, data):
        self._context[name] = data

    def is_sparse(self):
        if self._pdf.dtypes.apply(pd.api.types.is_sparse).any():
            return True
        else:
            return False