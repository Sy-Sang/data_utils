#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Sy,Sang"
__version__ = ""
__license__ = "GPLv3"
__maintainer__ = "Sy, Sang"
__email__ = "martin9le@163.com"
__status__ = "Development"
__credits__ = []
__date__ = ""
__copyright__ = ""

# 系统模块
import copy
import pickle
import json
from typing import Union, Self

# 项目模块
from easy_datetime.timestamp import TimeStamp, TimeLine
from easy_datetime.temporal_utils import timer
from data_utils.serial_utils.data_series import DataSeries
from data_utils.serial_utils.series_trans_utils import ColumnTransRec, DataTransformator, MinMax

# 外部模块
import numpy


# 代码块

# @timer
# def init_time_data(data: Union[list, tuple, numpy.ndarray], *args, **kwargs):
#     return numpy.unique([TimeStamp(i).timestamp() for i in data]).tolist()


class TimeSeries(DataSeries):
    """
    时间序列
    """

    def __init__(self, *args, **kwargs):
        largs = list(args)
        dkwargs = dict(kwargs)
        time_stamp = TimeLine()

        if len(largs) > 0:
            time_stamp = TimeLine(numpy.unique(largs[0]))
            largs[0] = time_stamp.timestamp()
        elif len(dkwargs) > 0:
            k = list(dkwargs)[0]
            time_stamp = TimeLine(numpy.unique(dkwargs[k]))
            dkwargs[k] = time_stamp.timestamp()
        else:
            pass

        super().__init__(*largs, **dkwargs)
        self.time_stamp = time_stamp

    def __len__(self):
        return super().__len__()

    def uniq(self) -> Self:
        return super().uniq()

    def __str__(self) -> str:
        return super().__str__()

    def __repr__(self):
        return self.__str__()

    def to_list_dic(self) -> dict:
        """
        输出字典, 值为列表格式
        :return:
        """
        list_dic = {}
        for i, k in enumerate(self.key_list):
            if i == 0:
                list_dic[k] = self.time_stamp.timestamp()
            else:
                list_dic[k] = numpy.array(self.data[k]).tolist()
        return list_dic

    def to_data_list(self) -> list:
        return super().to_data_list()

    def get_array(self) -> numpy.ndarray:
        return super().get_array()

    def get_slice(self, index: int) -> numpy.ndarray:
        return super().get_slice(index)

    def __getitem__(self, item) -> Union[numpy.ndarray, Self]:
        return super().__getitem__(item)

    def _comb_by_key_and_value(self, keys: list, matrix: numpy.ndarray) -> Self:
        return super()._comb_by_key_and_value(keys, matrix)

    # @classmethod
    # def comb_by_key_and_value(cls, keys: list, matrix: numpy.ndarray) -> Self:
    #     return super().comb_by_key_and_value(keys, matrix)

    def get_temporal_expression_array(self, *args) -> numpy.ndarray:
        """
        获取带有时间标记的numpy.array
        :param args:
        :param kwargs:
        :return:
        """
        array = self.get_array()
        for arg in args:
            te_array = numpy.array([
                i.pdf_map[arg] for i in self.time_stamp.data
            ])
            array = numpy.column_stack((array, te_array))
        return array

    def add_temporal_expression_column(self, *args) -> Self:
        """

        :param args:
        :return:
        """
        array = self.get_temporal_expression_array(*args)
        new_keys = self.key_list + list(args)
        return self._comb_by_key_and_value(new_keys, array)

    def get_time_slice(self, temporal_expression: str, value: int = 1, operator: str = "==") -> Self:
        """
        时间切片
        :param temporal_expression:
        :param value:
        :param operator:
        :return:
        """
        index = self.time_stamp.get_time_slice(temporal_expression, value, operator)
        array = self.get_array()[index]
        return self._comb_by_key_and_value(self.key_list, array)

    def __abs__(self):
        return super().__abs__()

    def __add__(self, other):
        y = self.get_array() + other
        y[:, 0] = copy.deepcopy(self.x)
        return self._comb_by_key_and_value(self.key_list, y)

    def __sub__(self, other):
        y = self.get_array() - other
        y[:, 0] = copy.deepcopy(self.x)
        return self._comb_by_key_and_value(self.key_list, y)

    def __mul__(self, other):
        y = self.get_array() * other
        y[:, 0] = copy.deepcopy(self.x)
        return self._comb_by_key_and_value(self.key_list, y)

    def __truediv__(self, other):
        y = self.get_array() / other
        y[:, 0] = copy.deepcopy(self.x)
        return self._comb_by_key_and_value(self.key_list, y)

    def __pow__(self, other):
        y = self.get_array() ** other
        y[:, 0] = copy.deepcopy(self.x)
        return self._comb_by_key_and_value(self.key_list, y)

    def __floordiv__(self, other):
        y = self.get_array() // other
        y[:, 0] = copy.deepcopy(self.x)
        return self._comb_by_key_and_value(self.key_list, y)

    def __mod__(self, other):
        y = self.get_array() % other
        y[:, 0] = copy.deepcopy(self.x)
        return self._comb_by_key_and_value(self.key_list, y)

    def __iadd__(self, other):
        new_series = self + other
        self.data.update(new_series.data)
        return self

    def __isub__(self, other):
        new_series = self - other
        self.data.update(new_series.data)
        return self

    def __imul__(self, other):
        new_series = self * other
        self.data.update(new_series.data)
        return self

    def __itruediv__(self, other):
        new_series = self / other
        self.data.update(new_series.data)
        return self

    def __ipow__(self, other):
        new_series = self ** other
        self.data.update(new_series.data)
        return self

    def __ifloordiv__(self, other):
        new_series = self // other
        self.data.update(new_series.data)
        return self

    def __imod__(self, other):
        new_series = self % other
        self.data.update(new_series.data)
        return self

    def _gt(self, other: (TimeStamp, str, int, float), column_index=0):
        if column_index == 0:
            if isinstance(other, TimeStamp):
                ov = other.timestamp()
            else:
                ov = TimeStamp(other).timestamp()
        else:
            ov = other
        x = self.time_stamp.timestamp_array()
        index = numpy.where(x > ov)
        y = self.get_array()[index]
        return self._comb_by_key_and_value(self.key_list, y)

    def _lt(self, other: (TimeStamp, str, int, float), column_index=0):
        if column_index == 0:
            if isinstance(other, TimeStamp):
                ov = other.timestamp()
            else:
                ov = TimeStamp(other).timestamp()
        else:
            ov = other
        x = self.time_stamp.timestamp_array()
        index = numpy.where(x < ov)
        y = self.get_array()[index]
        return self._comb_by_key_and_value(self.key_list, y)

    def _ge(self, other: (TimeStamp, str, int, float), column_index=0):
        if column_index == 0:
            if isinstance(other, TimeStamp):
                ov = other.timestamp()
            else:
                ov = TimeStamp(other).timestamp()
        else:
            ov = other
        x = self.time_stamp.timestamp_array()
        index = numpy.where(x >= ov)
        y = self.get_array()[index]
        return self._comb_by_key_and_value(self.key_list, y)

    def _le(self, other: (TimeStamp, str, int, float), column_index=0):
        if column_index == 0:
            if isinstance(other, TimeStamp):
                ov = other.timestamp()
            else:
                ov = TimeStamp(other).timestamp()
        else:
            ov = other
        x = self.time_stamp.timestamp_array()
        index = numpy.where(x <= ov)
        y = self.get_array()[index]
        return self._comb_by_key_and_value(self.key_list, y)

    def _eq(self, other: (TimeStamp, str, int, float), column_index=0):
        if column_index == 0:
            if isinstance(other, TimeStamp):
                ov = other.timestamp()
            else:
                ov = TimeStamp(other).timestamp()
        else:
            ov = other
        x = self.time_stamp.timestamp_array()
        index = numpy.where(x == ov)
        y = self.get_array()[index]
        return y

    def where(self, symbol, num, column_index=0):
        """
        检索
        :param symbol:
        :param num:
        :param column_index:
        :return:
        """
        operations = {
            "==": self._eq,
            ">": self._gt,
            "<": self._lt,
            ">=": self._ge,
            "<=": self._le
        }

        return operations[symbol](num, column_index)

    def swap_axis(self, key: str):
        """
        变换排序轴
        :param key:
        :return:
        """
        raise Exception("TimeSeries has no method : swap_axis")

    def swap_axis_by_index(self, index: int):
        """

        :param index:
        :return:
        """
        raise Exception("TimeSeries has no method : swap_axis")

    def bin(self, bin_list: list[float]) -> list[numpy.ndarray]:
        return super().bin(bin_list)

    def bin_method(self, bin_list: list[float], method: callable = numpy.mean, *args, **kwargs) -> Self:
        """
        使用bin方法聚合分组数据
        """
        bin_index_list = self.bin(bin_list)
        new_array = numpy.array([])
        for i, k in enumerate(self.key_list):

            if i == 0:
                new_temp_array = numpy.array(bin_list[:-1])
                new_array = numpy.concatenate((new_array, new_temp_array))
            else:
                new_temp_array = numpy.array([
                    method(self.data[k][bin_index], *args, **kwargs) if len(bin_index) > 0 else numpy.nan
                    for bin_index in bin_index_list
                ])
                new_array = numpy.column_stack((new_array, new_temp_array))

        return self._comb_by_key_and_value(self.key_list, new_array)

    def aggregate(
            self, step: list, method: callable = numpy.mean,
            align: bool = False, align_domain: Union[list, tuple] = None,
            *args, **kwargs
    ) -> Self:
        if align is False:
            datetime_range = TimeStamp.timestamp_range(
                self.time_stamp.data[0],
                self.time_stamp.data[-1],
                step[0],
                step[1],
                True
            )
        else:
            datetime_range = TimeStamp.timestamp_range(
                TimeStamp(align_domain[0]),
                TimeStamp(align_domain[1]),
                step[0],
                step[1],
                True
            )
        bin_list = [i.timestamp() for i in datetime_range]

        return self.bin_method(bin_list, method, *args, **kwargs)

    def data_trans(self, method: DataTransformator = MinMax, *args, **kwargs) -> Self:
        self.transform_record.his.append([])
        array = self.get_array()
        new_array = numpy.array([])
        for i, k in enumerate(self.key_list):

            if i == 0:
                new_array = numpy.concatenate((new_array, array[:, 0]))
            else:
                y, inf, para = method.f(array[:, i], *args, **kwargs)
                rec = ColumnTransRec(method=inf, dim=k, param=para)
                self.transform_record.his[-1].append(rec)
                new_array = numpy.column_stack((new_array, y))

        ts = self._comb_by_key_and_value(self.key_list, new_array)
        ts.transform_record = copy.deepcopy(self.transform_record)
        return ts

    def data_re_trans(self, rollback_num: int = 1) -> Self:
        return super().data_re_trans(rollback_num)


if __name__ == "__main__":
    drange = TimeStamp.timestamp_range("2021-1-1", "2024-1-1", "hour", 1)
    data = numpy.arange(0, len(drange) * 3).reshape(3, -1)
    ts = TimeSeries(drange, *data)

    tss = ts.aggregate(["month", 1], align=True, align_domain=['2021-1-1', '2025-1-1']).data_trans().data_re_trans()

    print(tss.time_stamp)
