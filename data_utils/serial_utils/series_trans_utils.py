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
import decimal
import pickle
import json
from typing import Union, Self
from collections import namedtuple

# 项目模块

# 外部模块
import numpy
from sklearn.cluster import KMeans

SeriesTransRec = namedtuple("SeriesTransRec", "his")
ColumnTransRec = namedtuple('ColumnTransRec', ["dim", "method", "param"])


# 代码块

class DataTransformator:
    """
    数据变形类
    """

    @classmethod
    def f(cls, xlist: Union[list, tuple, numpy.ndarray], *args, **kwargs) -> tuple[numpy.ndarray, callable, list]:
        """
        变形函数
        """
        pass

    @classmethod
    def inf(cls, ylist: Union[list, tuple, numpy.ndarray], *args, **kwargs) -> numpy.ndarray:
        """
        反函数
        """
        pass

    @classmethod
    def identity(cls, ylist: Union[list, tuple, numpy.ndarray], *args, **kwargs) -> numpy.ndarray:
        """
        y = x
        """
        return numpy.array(ylist)


class MinMax(DataTransformator):
    @classmethod
    def f(cls, xlist: Union[list, tuple, numpy.ndarray],
          a: Union[float, int, decimal.Decimal, numpy.floating] = 0,
          b: Union[float, int, decimal.Decimal, numpy.floating] = 1,
          eps: Union[float, int, decimal.Decimal, numpy.floating] = 0,
          *args, **kwargs):
        xarray = numpy.array(xlist)
        if max(xarray) == min(xarray):
            yarray = numpy.zeros(len(xlist))
            return yarray, cls.identity, [xarray]
        else:
            yarray = (xarray - min(xarray)) * (b - a - 2 * eps) / (max(xarray) - min(xarray)) + a + eps
            return yarray, cls.inf, [min(xarray), max(xarray), a, b, eps]

    @classmethod
    def inf(cls, ylist: Union[list, tuple, numpy.ndarray], *args, **kwargs) -> numpy.ndarray:
        yarray = numpy.array(ylist)
        xmin = args[0]
        xmax = args[1]
        a = args[2]
        b = args[3]
        eps = args[4]
        return (yarray - a - eps) * (xmax - xmin) / (a + b + 2 * eps) + xmin


class ZScore(DataTransformator):

    @classmethod
    def f(cls, x: Union[list, tuple, numpy.ndarray],
          miu: float = 0,
          sigma: float = 1,
          *args, **kwargs):
        if numpy.std(x, ddof=1) == 0:
            y = numpy.array([miu] * len(x))
            return y, cls.identity, [x]
        else:
            z = (numpy.array(x) - numpy.mean(x)) / numpy.std(x, ddof=1)
            y = z * sigma + miu
            return y, cls.identity, [numpy.mean(x), numpy.std(x, ddof=1), miu, sigma]

    @classmethod
    def inf(cls, y: Union[list, tuple, numpy.ndarray], *args, **kwargs) -> numpy.ndarray:
        x_miu = args[0]
        x_std = args[1]
        miu = args[2]
        sigma = args[3]
        z = (numpy.array(y) - miu) / sigma
        x = z * x_std + x_miu
        return x


class RobustScaler(DataTransformator):
    @classmethod
    def f(cls, x: Union[list, tuple, numpy.ndarray], *args, **kwargs):
        median = numpy.median(x)
        q1 = numpy.percentile(x, 25)
        q3 = numpy.percentile(x, 75)
        iqr = q3 - q1
        if iqr == 0:
            y = numpy.zeros(len(x))
            return y, cls.identity, [x]
        else:
            y = (numpy.array(x) - median) / iqr
            return y, cls.inf, [median, q1, q3, iqr]

    @classmethod
    def inf(cls, y: Union[list, tuple, numpy.ndarray], *args, **kwargs) -> numpy.ndarray:
        median, q1, q3, iqr = args
        return numpy.array(y) * iqr + median


class KmeansCluster(DataTransformator):
    """
    kmeans聚类
    """

    @classmethod
    def f(cls, x: Union[list, tuple, numpy.ndarray], k: int = 3, *args, **kwargs):
        x = numpy.array(x).astype(float)
        sort_index = numpy.argsort(x)
        sorted_x = x[sort_index]
        k = KMeans(n_clusters=k, random_state=0)
        k.fit(sorted_x.reshape(len(x), -1))
        label = 0
        label_list = []
        label_mean = {}
        mean_index = 0
        for i in range(len(k.labels_)):
            if i == 0:
                pass
            elif k.labels_[i] == k.labels_[i - 1]:
                pass
            else:
                mean = numpy.mean(sorted_x[mean_index:i - 1])
                label_mean[int(label)] = mean
                mean_index = i
                label += 1
            label_list.append(label)
        label_mean[int(label)] = numpy.mean(sorted_x[mean_index:])
        resort_index = numpy.argsort(sort_index)
        label_array = numpy.array(label_list)[resort_index]
        return label_array.astype(int), cls.inf, [label_mean]

    @classmethod
    def inf(cls, ylist: Union[list, tuple, numpy.ndarray], *args, **kwargs) -> numpy.ndarray:
        xlist = []
        dic: dict = args[0]
        for i in ylist:
            xlist.append(dic[int(i)])
        return numpy.array(xlist).astype(float)


if __name__ == "__main__":
    data = numpy.random.uniform(0, 10, 100)
    k, f, p = KmeansCluster.f(data, k=3)
    print(data.tolist())
    print(k)
    print(p)
    print(KmeansCluster.inf(k, *p).tolist())
