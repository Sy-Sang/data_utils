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
from collections import namedtuple

# 项目模块
from easy_datetime.timestamp import TimeStamp, TimeLine
from easy_utils.number_utils.number_utils import EasyFloat
from data_utils.serial_utils.series_trans_utils import SeriesTransRec, ColumnTransRec, DataTransformator, MinMax

# 外部模块
import numpy


# 代码块

def element_from_tuple(*args) -> tuple:
    """从元祖获取初始化元素"""
    dic = {}
    for i in range(len(args)):
        dic[f"column_{i}"] = numpy.array(args[i])
    return element_from_dict(**dic)


def element_from_dict(**kwargs) -> tuple:
    """从字典获取初始化元素"""
    x = numpy.array([])
    data = {}
    key_list = list(kwargs)
    width = len(key_list)
    if width > 0:
        original_x = numpy.array(
            kwargs[key_list[0]]
        )
        seq_index = numpy.argsort(original_x)
        for i, k in enumerate(key_list):
            data[k] = numpy.array(kwargs[k])[seq_index]
            if i == 0:
                x = copy.deepcopy(data[k])
            else:
                pass
    else:
        pass
    return data, key_list, width, x


def element_from_none() -> tuple:
    """空元素"""
    return {}, [], 0, numpy.array([])


class DataSeries(object):
    """
    序列
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) > 0:
            self.data, self.key_list, self.width, self.x = element_from_tuple(*args)
        elif len(kwargs) > 0:
            self.data, self.key_list, self.width, self.x = element_from_dict(**kwargs)
        else:
            self.data, self.key_list, self.width, self.x = element_from_none()
        self.transform_record = SeriesTransRec([])

    def __len__(self):
        return len(self.x)

    def uniq(self) -> Self:
        """对x轴去重"""
        x = self.data[self.key_list[0]]
        new, index = numpy.unique(x, return_index=True)
        array = self.get_array()[index]
        return self._comb_by_key_and_value(self.key_list, array)

    def __str__(self) -> str:
        return str(self.to_list_dic())

    def to_list_dic(self) -> dict:
        """
        输出字典, 值为列表格式
        :return:
        """
        list_dic = {}
        for k in self.key_list:
            list_dic[k] = numpy.array(self.data[k]).tolist()
        return list_dic

    def to_data_list(self) -> list:
        """

        :return:
        """
        l = []
        for k in self.key_list:
            l.append(self.data[k].tolist())
        return l

    def get_array(self) -> numpy.ndarray:
        """
        获取numpy.array
        :return:
        """
        array = numpy.array([])
        for i, k in enumerate(self.key_list):
            if i == 0:
                array = copy.deepcopy(self.data[k])
            else:
                array = numpy.column_stack((array, self.data[k]))
        return array

    def get_slice(self, index: int) -> numpy.ndarray:
        """
        获取切片
        :param index:
        :return:
        """
        return self.get_array()[index]

    def __getitem__(self, item) -> Union[numpy.ndarray, Self]:
        """

        :param item:
        :return:
        """
        if item in self.key_list:
            return copy.deepcopy(self.data[item])
        elif isinstance(item, int):
            return self.get_slice(item)
        else:
            init_dic = {}
            for i, dim in enumerate(self.key_list):
                init_dic[dim] = self.data[dim][item]

        new_series = type(self)(**init_dic)
        return new_series

    def _comb_by_key_and_value(self, keys: list, matrix: numpy.ndarray) -> Self:
        """
        组合键值和value
        :param keys:
        :param matrix:
        :return:
        """
        init_dic = {}
        for i, k in enumerate(keys):
            init_dic[k] = matrix[:, i]
        return type(self)(**init_dic)

    @classmethod
    def comb_by_key_and_value(cls, keys: list, matrix: numpy.ndarray) -> Self:
        """
        组合键值和value
        :param keys:
        :param matrix:
        :return:
        """
        init_dic = {}
        for i, k in enumerate(keys):
            init_dic[k] = matrix[:, i]
        return cls(**init_dic)

    def __abs__(self) -> Self:
        """
        绝对值
        :return:
        """
        y = abs(self.get_array())
        return self._comb_by_key_and_value(self.key_list, y)

    def __add__(self, other) -> Self:
        y = self.get_array() + other
        return self._comb_by_key_and_value(self.key_list, y)

    def __sub__(self, other) -> Self:
        y = self.get_array() - other
        return self._comb_by_key_and_value(self.key_list, y)

    def __mul__(self, other) -> Self:
        y = self.get_array() * other
        return self._comb_by_key_and_value(self.key_list, y)

    def __truediv__(self, other) -> Self:
        y = self.get_array() / other
        return self._comb_by_key_and_value(self.key_list, y)

    def __pow__(self, other) -> Self:
        y = self.get_array() ** other
        return self._comb_by_key_and_value(self.key_list, y)

    def __floordiv__(self, other) -> Self:
        y = self.get_array() // other
        return self._comb_by_key_and_value(self.key_list, y)

    def __mod__(self, other) -> Self:
        y = self.get_array() % other
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

    def _gt(self, other: float, column_index=0):
        x = self.data[self.key_list[column_index]]
        index = numpy.where(x > other)
        y = self.get_array()[index]
        return self._comb_by_key_and_value(self.key_list, y)

    def _lt(self, other: float, column_index=0):
        x = self.data[self.key_list[column_index]]
        index = numpy.where(x < other)
        y = self.get_array()[index]
        return self._comb_by_key_and_value(self.key_list, y)

    def _ge(self, other: float, column_index=0):
        x = self.data[self.key_list[column_index]]
        index = numpy.where(x >= other)
        y = self.get_array()[index]
        return self._comb_by_key_and_value(self.key_list, y)

    def _le(self, other: float, column_index=0):
        x = self.data[self.key_list[column_index]]
        index = numpy.where(x <= other)
        y = self.get_array()[index]
        return self._comb_by_key_and_value(self.key_list, y)

    def _eq(self, other: float, column_index=0):
        x = self.data[self.key_list[column_index]]
        index = numpy.where(x == other)
        y = self.get_array()[index]
        return y

    def where(self, symbol, num, column_index=0) -> Union[Self, numpy.ndarray]:
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

    def swap_axis(self, key: str) -> Self:
        """
        变换排序轴
        :param key:
        :return:
        """
        new_keys = [key] + [k for k in self.key_list if k != key]
        n = self.key_list.index(key)
        array = self.get_array()
        new_array = numpy.concatenate((array[:, n:], array[:, :n]), axis=1)
        new_series = self._comb_by_key_and_value(new_keys, new_array)
        new_series.transform_record = copy.deepcopy(self.transform_record)
        return new_series

    def swap_axis_by_index(self, index: int) -> Self:
        """

        :param index:
        :return:
        """
        key = self.key_list[index]
        return self.swap_axis(key)

    def bin(self, bin_list: list[float]) -> list[numpy.ndarray]:
        """
        bin方法对数据分组
        """
        bin_index_list = []
        first_index = 0
        for i, b in enumerate(bin_list):
            if i > 0:
                bin_index_list.append([])
                if self.x[-1] < bin_list[i - 1]:
                    pass
                else:
                    for tj, x in enumerate(self.x[first_index:]):
                        j = tj + first_index
                        if bin_list[i - 1] <= x < bin_list[i]:
                            bin_index_list[i - 1].append(j)
                        elif x >= bin_list[i]:
                            first_index = j
                            break
            else:
                pass
        return [numpy.array(i) for i in bin_index_list]

    def bin_method(self, bin_list: list[float], method: callable = numpy.mean, *args, **kwargs) -> Self:
        """
        使用bin方法聚合分组数据
        """
        bin_index_list = self.bin(bin_list)
        new_array = numpy.array([])
        for i, k in enumerate(self.key_list):
            new_temp_array = numpy.array([
                method(self.data[k][bin_index], *args, **kwargs) if len(bin_index) > 0 else numpy.nan
                for bin_index in bin_index_list
            ])
            if i == 0:
                new_array = numpy.concatenate((new_array, new_temp_array))
            else:
                new_array = numpy.column_stack((new_array, new_temp_array))

        return self._comb_by_key_and_value(self.key_list, new_array)

    def aggregate(
            self, step: float, method: callable = numpy.mean,
            align: bool = False, align_domain: Union[list, tuple] = (),
            *args, **kwargs
    ) -> Self:
        """
        聚合数据
        """
        if align is False:
            bin_list = EasyFloat.frange(self.x[0], self.x[-1], step, True)
        else:
            bin_list = EasyFloat.frange(align_domain[0], align_domain[1], step, True)

        return self.bin_method(bin_list, method, *args, **kwargs)

    def data_trans(self, method: DataTransformator = MinMax, *args, **kwargs) -> Self:
        """
        数据变形
        """
        self.transform_record.his.append([])
        array = self.get_array()
        new_array = numpy.array([])
        for i, k in enumerate(self.key_list):
            y, inf, para = method.f(array[:, i], *args, **kwargs)
            if i == 0:
                new_array = numpy.concatenate((new_array, y))
            else:
                new_array = numpy.column_stack((new_array, y))
            rec = ColumnTransRec(method=inf, dim=k, param=para)
            self.transform_record.his[-1].append(rec)

        ts = self._comb_by_key_and_value(self.key_list, new_array)
        ts.transform_record = copy.deepcopy(self.transform_record)
        return ts

    def data_re_trans(self, rollback_num: int = 1):
        """
        数据逆变形
        """
        s = copy.deepcopy(self)
        for _ in range(min(len(self.transform_record.his), rollback_num)):
            rec_list: list[ColumnTransRec] = self.transform_record.his[-1]
            for i, r in enumerate(rec_list):
                k = r.dim
                method = r.method
                param = r.param
                s.data[k] = method(s.data[k], *param)
            del s.transform_record.his[-1]
        return s

    def column_entangled(self, f: callable, *args: str, **kwargs) -> Self:
        """
        组间相干
        """
        entangled_column_name = f"{'_'.join(item for item in args)}"
        entangled_data = []
        for i in range(len(self)):
            entangled_data.append(
                f(*[self.data[a][i] for a in args], **kwargs)
            )

        array = self.get_array()
        array = numpy.column_stack((array, entangled_data))
        s = self._comb_by_key_and_value(self.key_list + [entangled_column_name], array)
        s.transform_record = copy.deepcopy(self.transform_record)
        return s


class NamedSeries(DataSeries):
    def __init__(self, data: Union[list, tuple, numpy.ndarray] = None, **kwargs):
        if data is None:
            super().__init__(**kwargs)
        else:
            super().__init__(
                x=list(range(len(data))),
                y=data
            )
        nt = namedtuple("nt", ["x", "y"])
        self.tuple = nt(self.data["x"], self.data["y"])


if __name__ == "__main__":
    data = numpy.random.randint(1, 1000, 90000)
    s = DataSeries(list(range(len(data))), data)
    print(s.x)
    ss = s.aggregate(100, align=True, align_domain=[0, 20000])
    # print(ss)
    print(ss.data_trans())
    # print(ss.data_trans().data_re_trans())
