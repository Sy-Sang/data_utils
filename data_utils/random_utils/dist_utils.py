#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Sy,Sang"
__version__ = ""
__license__ = "GUN"
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

from pip._internal.distributions import AbstractDistribution
from sympy.diffgeom import rn

# 项目模块
from easy_utils.number_utils import calculus_utils, number_utils
from easy_utils.obj_utils.sequence_utils import flatten

# 外部模块
import numpy


# 代码块

class ABCDistribution(ABC):
    """
    概率分布
    """

    def _2d_curve(self, f: callable, *args, first, end, step, **kwargs) -> Union[float, numpy.ndarray]:
        """
        形成2维曲线
        """
        flatten_args = flatten(args)
        if len(flatten_args) == 1:
            return f(*flatten_args)
        elif len(flatten_args) > 1:
            return numpy.array([[fa, f(fa)] for fa in flatten_args])
        else:
            if f.__name__ == '_ppf':
                xarray = number_utils.EasyFloat.frange(first, end, step, True)
                return numpy.array([[x, f(x)] for x in xarray])
            else:
                xarray = number_utils.EasyFloat.frange(first, end, step, True)
                yarray = numpy.array([self._ppf(x) for x in xarray])
                return numpy.array([[y, f(y)] for y in yarray])

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

    def ppf(self, *args, first=0.01, end=0.99, step=0.01, **kwargs):
        """ppf曲线"""
        return self._2d_curve(self._ppf, *args, first=first, end=end, step=step, **kwargs)

    def pdf(self, *args, first=0.01, end=0.99, step=0.01, **kwargs):
        """pdf曲线"""
        return self._2d_curve(self._pdf, *args, first=first, end=end, step=step, **kwargs)

    def cdf(self, *args, first=0.01, end=0.99, step=0.01, **kwargs):
        """cdf曲线"""
        return self._2d_curve(self._cdf, *args, first=first, end=end, step=step, **kwargs)

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
        return random_array[:num][:, 1]

    def n_mean(self, method: callable = calculus_utils.simpsons_integrate, first: float = 0 + numpy.finfo(float).eps,
               end: float = 1 - numpy.finfo(float).eps,
               num: int = 100) -> float:
        """
        数值方法计算概率分布期望
        """

        def f(x: float) -> float:
            return x * self._pdf(x)

        return method(f, self._ppf(first), self._ppf(end), num)

    def n_std(self, method: callable = calculus_utils.simpsons_integrate, first: float = 0 + numpy.finfo(float).eps,
              end: float = 1 - numpy.finfo(float).eps,
              num: int = 100) -> float:
        """
        数值方法计算概率分布标准差
        """
        mean = self.n_mean(method, first, end, num)

        def f(x: float) -> float:
            return ((x - mean) ** 2) * self._pdf(x)

        return method(f, self._ppf(first), self._ppf(end), num) ** 0.5

    def n_skewness(self, method: callable = calculus_utils.simpsons_integrate,
                   first: float = 0 + numpy.finfo(float).eps,
                   end: float = 1 - numpy.finfo(float).eps,
                   num: int = 100) -> float:
        """
        数值方法计算偏度
        """
        mean = self.n_mean(method, first, end, num)
        std = self.n_std(method, first, end, num)

        def f(x: float) -> float:
            return self._pdf(x) * ((x - mean) / std) ** 3

        return method(f, self._ppf(first), self._ppf(end), num)

    def n_kurtosis(self, method: callable = calculus_utils.simpsons_integrate,
                   first: float = 0 + numpy.finfo(float).eps,
                   end: float = 1 - numpy.finfo(float).eps,
                   num: int = 100):
        """
        数值方法计算峰度
        """
        mean = self.n_mean(method, first, end, num)
        std = self.n_std(method, first, end, num)

        def f(x: float) -> float:
            return self._pdf(x) * ((x - mean) / std) ** 4

        return method(f, self._ppf(first), self._ppf(end), num)

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
    nt.__annotations__ = {"dist": AbstractDistribution, "corr": float}
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
                r0 = d.dist.ppf(*rx)[:, 1]
            else:
                r0 = numpy.array([])
            r1 = d.dist.rvf(rn - l)
            r = numpy.concatenate((r0, r1))
            array = numpy.column_stack((array, r))

    numpy.random.shuffle(array)
    return array[:num].T


def correlated_random_number(*args, num: int = 100):
    """
    生成满足指定概率分布并具备相关性的随机数
    """

    def corrf(x: int, xlist: numpy.ndarray, ylist: numpy.ndarray, target: float) -> numpy.ndarray:
        nx = numpy.concatenate((xlist[:x], xlist[x:]))
        return abs(numpy.corrcoef(nx, ylist) - target)

    def newtown_corrf(xlist: numpy.ndarray, ylist: numpy.ndarray, target: float):
        return calculus_utils.newton_method(
            corrf, 50, 0.1, 100, 1,
            xlist=xlist, ylist=ylist, target=target
        )

    n = max(num, 100)
    array = numpy.arange(n)
    nt = namedtuple("nt", ["dist", "corr"])
    nt.__annotations__ = {"dist": AbstractDistribution, "corr": float}
    for i, a in enumerate(args):
        d = nt(a[0], a[1])
        if d.corr > 0:
            rx = number_utils.EasyFloat.np_finterval(-1 * d.corr, d.corr, n) + d.dist.rvf(n)
        elif d.corr < 0:
            rx = number_utils.EasyFloat.np_finterval(-1 * d.corr, d.corr, n) + d.dist.rvf(n)
        else:
            rx = d.dist.rvf(n)
        array = numpy.column_stack((array, rx))
    numpy.random.shuffle(array)
    return array[:, 1:][:num].T


if __name__ == "__main__":
    pass
