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
from easy_utils.obj_utils.enumerable_utils import flatten

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
    def n(cls, xlist: Union[list, tuple, numpy.ndarray], *args, **kwargs) -> numpy.ndarray:
        """
        变形函数
        """
        pass

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
    def n(cls, xlist: Union[list, tuple, numpy.ndarray],
          a: Union[float, int, decimal.Decimal, numpy.floating] = 0,
          b: Union[float, int, decimal.Decimal, numpy.floating] = 1,
          eps: Union[float, int, decimal.Decimal, numpy.floating] = 0,
          *args, **kwargs):
        xarray = numpy.array(xlist)
        if max(xarray) == min(xarray):
            yarray = numpy.zeros(len(xlist))
            return yarray
        else:
            yarray = (xarray - min(xarray)) * (b - a - 2 * eps) / (max(xarray) - min(xarray)) + a + eps
            return yarray

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
    def n(cls, x: Union[list, tuple, numpy.ndarray],
          miu: float = 0,
          sigma: float = 1,
          *args, **kwargs):
        if numpy.std(x, ddof=1) == 0:
            y = numpy.array([miu] * len(x))
            return y
        else:
            z = (numpy.array(x) - numpy.mean(x)) / numpy.std(x, ddof=1)
            y = z * sigma + miu
            return y

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
    def n(cls, x: Union[list, tuple, numpy.ndarray], q0=25, *args, **kwargs):
        median = numpy.median(x)
        q1 = numpy.percentile(x, q0)
        q3 = numpy.percentile(x, 100 - q0)
        iqr = q3 - q1
        if iqr == 0:
            y = numpy.zeros(len(x))
            return y
        else:
            y = (numpy.array(x) - median) / iqr
            return y

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
    def n(cls, x: Union[list, tuple, numpy.ndarray], k: int = 3, *args, **kwargs):
        x = numpy.array(x).astype(float)
        k = KMeans(n_clusters=k)
        k.fit(x.reshape(len(x), -1))
        label_index = numpy.argsort(flatten(k.cluster_centers_))
        sorted_label = numpy.array([label_index.tolist().index(i) for i in k.labels_]).astype(int)
        return sorted_label

    @classmethod
    def f(cls, x: Union[list, tuple, numpy.ndarray], k: int = 3, *args, **kwargs):
        x = numpy.array(x).astype(float)
        k = KMeans(n_clusters=k)
        k.fit(x.reshape(len(x), -1))
        label_index = numpy.argsort(flatten(k.cluster_centers_))
        sorted_label = numpy.array([label_index.tolist().index(i) for i in k.labels_]).astype(int)
        return sorted_label, cls.identity, [x]


if __name__ == "__main__":
    data = numpy.random.uniform(0, 10, 100)
    k, f, p = KmeansCluster.f(data, k=3)
    print(data.tolist())
    print(k)
    print(KmeansCluster.identity(p))
