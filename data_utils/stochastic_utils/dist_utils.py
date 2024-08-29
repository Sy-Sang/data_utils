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
from abc import ABC, abstractmethod
from collections import namedtuple

# 项目模块
from easy_utils.number_utils import calculus_utils, number_utils
from easy_utils.obj_utils.sequence_utils import flatten
from data_utils.serial_utils.series_trans_utils import MinMax

# 外部模块
import numpy

# 代码块


curve = namedtuple('curve', ['x', 'y'])


class ABCDistribution(ABC):
    """
    概率分布
    """

    parameterlength = 0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def __str__(self) -> str:
        return str({
            "name": str(type(self)),
            "args": str(self.args),
            "kwargs": str(self.kwargs)
        })

    def _2d_curve(self, f: callable, *args, first, end, step, num, **kwargs) -> Union[float, curve]:
        """
        形成2维曲线
        """
        flatten_args = flatten(args)
        if len(flatten_args) == 1:
            return f(*flatten_args)
        else:
            if len(flatten_args) > 1:
                xarray = numpy.array(flatten_args)
            else:
                if f.__name__ == '_ppf':
                    if num is not None:
                        xarray = number_utils.EasyFloat.finterval(first, end, num, True)
                    else:
                        xarray = number_utils.EasyFloat.frange(first, end, step, True)
                else:
                    if num is not None:
                        ppf_xarray = number_utils.EasyFloat.finterval(first, end, num, True)
                    else:
                        ppf_xarray = number_utils.EasyFloat.frange(first, end, step, True)
                    xarray = numpy.array([self._ppf(x) for x in ppf_xarray])
            c = curve(xarray, numpy.array([f(x) for x in xarray]))
            # c = curve(xarray, f(xarray))
            return c

    @abstractmethod
    def _ppf(self, *args, **kwargs) -> float:
        """
        分位数函数
        """
        pass

    @abstractmethod
    def _pdf(self, *args, **kwargs) -> float:
        """
        pdf
        """
        pass

    @abstractmethod
    def _cdf(self, *args, **kwargs) -> float:
        """
        cdf
        """
        pass

    def ppf(self, *args, first=0.01, end=0.99, step=0.01, num: float = None, **kwargs) -> Union[float, curve]:
        """ppf曲线"""
        return self._2d_curve(self._ppf, *args, first=first, end=end, step=step, num=num, **kwargs)

    def pdf(self, *args, first=0.01, end=0.99, step=0.01, num: float = None, **kwargs) -> Union[float, curve]:
        """pdf曲线"""
        return self._2d_curve(self._pdf, *args, first=first, end=end, step=step, num=num, **kwargs)

    def cdf(self, *args, first=0.01, end=0.99, step=0.01, num: float = None, **kwargs) -> Union[float, curve]:
        """cdf曲线"""
        return self._2d_curve(self._cdf, *args, first=first, end=end, step=step, num=num, **kwargs)

    def rvf(self, num=1) -> numpy.ndarray:
        """
        生成随机数
        :param num:
        :param D:
        :param K:
        :return:
        """
        n = max(100, num) * 2
        x_array = numpy.random.uniform(0 + numpy.finfo(float).eps, 1, size=n)
        random_array = self.ppf(x_array)
        return random_array.y[:num]

    @abstractmethod
    def mean(self) -> float:
        """均值"""
        pass

    @abstractmethod
    def std(self) -> float:
        """标准差"""
        pass


def correlated_rvf(dist_list: list, num: int = 100) -> numpy.ndarray:
    """
    相关的随机数
    """
    rn = max(num, 100)
    array = numpy.array([])
    nt = namedtuple("nt", ["dist", "corr"])
    nt.__annotations__ = {"dist": ABCDistribution, "corr": float}
    for i, a in enumerate(dist_list):
        d = nt(a[0], a[1])
        if i == 0:
            r = numpy.sort(d.dist.rvf(rn))
            array = numpy.concatenate((array, r))
        else:
            l = int(rn * abs(d.corr))
            if l != 0:
                rx = numpy.random.uniform(
                    0 + numpy.finfo(float).eps,
                    1 - numpy.finfo(float).eps,
                    l
                )
                if d.corr >= 0:
                    rx = numpy.sort(rx)
                elif d.corr < 0:
                    rx = numpy.sort(rx)[::-1]
                r0 = d.dist.ppf(*rx).y
            else:
                r0 = numpy.array([])
            r1 = d.dist.rvf(rn - l)
            r = numpy.concatenate((r0, r1))
            array = numpy.column_stack((array, r))

    numpy.random.shuffle(array)
    return array[:num].T


def correlated_random_number(base_dist: ABCDistribution, num: int = 100, *args):
    """
    生成满足指定概率分布并具备相关性的随机数
    """

    def noize_f(x: float) -> float:
        return 0.274882 / (0.0892443 + abs(x)) ** 2.325384114212880

    eps = numpy.finfo(float).eps

    n = max(num, 100)
    nt = namedtuple("nt", ["dist", "corr"])

    base_quantile = numpy.random.uniform(0 + eps, 1 - eps, n)
    array = base_dist.ppf(base_quantile).y

    for i, a in enumerate(args):
        d = nt(a[0], a[1])
        d.dist: ABCDistribution
        d.corr: float
        if d.corr != 0:
            noize_size = noize_f(d.corr)
            noize = numpy.random.uniform(0, noize_size, n)
            noised = base_quantile + noize if d.corr > 0 else noize - base_quantile
            noize_range = numpy.random.uniform(0 + eps, 1 - eps, n)
            normalized_noised, _1, _2 = MinMax.f(noised, a=numpy.min(noize_range), b=numpy.max(noize_range))
            array = numpy.column_stack((array, d.dist.ppf(normalized_noised).y))
        else:
            array = numpy.column_stack((array, d.dist.rvf(n)))

    return array[:num].T


if __name__ == "__main__":
    pass
