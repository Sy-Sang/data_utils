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
from math import e, pi, sqrt, erfc, log

# 项目模块
from easy_utils.number_utils import calculus_utils, number_utils
from easy_utils.obj_utils.sequence_utils import flatten

# 外部模块
import numpy
from scipy.special import betaincinv, beta, iv, gamma, erfinv, erfcinv, betainc


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

    def n_mean(self, first: float = 0 + numpy.finfo(float).eps, end: float = 1 - numpy.finfo(float).eps,
               num: int = 100) -> float:
        """
        数值方法计算概率分布期望
        """

        def f(x: float) -> float:
            return x * self._pdf(x)

        return calculus_utils.simpsons_integrate(f, self._ppf(first), self._ppf(end), num)

    def n_std(self, first: float = 0 + numpy.finfo(float).eps, end: float = 1 - numpy.finfo(float).eps,
              num: int = 100) -> float:
        """
        数值方法计算概率分布标准差
        """
        mean = self.n_mean(first, end, num)

        def f(x: float) -> float:
            return ((x - mean) ** 2) * self._pdf(x)

        return calculus_utils.simpsons_integrate(f, self._ppf(first), self._ppf(end), num) ** 0.5

    def n_skewness(self, first: float = 0 + numpy.finfo(float).eps, end: float = 1 - numpy.finfo(float).eps,
                   num: int = 100) -> float:
        """
        数值方法计算偏度
        """
        mean = self.n_mean(first, end, num)
        std = self.n_std(first, end, num)

        def f(x: float) -> float:
            return self._pdf(x) * ((x - mean) / std) ** 3

        return calculus_utils.simpsons_integrate(f, self._ppf(first), self._ppf(end), num)

    def n_kurtosis(self, first: float = 0 + numpy.finfo(float).eps, end: float = 1 - numpy.finfo(float).eps,
                   num: int = 100):
        """
        数值方法计算峰度
        """
        mean = self.n_mean(first, end, num)
        std = self.n_std(first, end, num)

        def f(x: float) -> float:
            return self._pdf(x) * ((x - mean) / std) ** 4

        return calculus_utils.simpsons_integrate(f, self._ppf(first), self._ppf(end), num)


class NormalDistribution(ABCDistribution):
    def __init__(self, mu: float, sigma: float):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def _ppf(self, x):
        if 0 < x < 1:
            return self.mu - sqrt(2) * self.sigma * erfcinv(2 * x)
        else:
            return numpy.nan

    def _pdf(self, x):
        return e ** (-((-self.mu + x) ** 2 / (2 * self.sigma ** 2))) / (sqrt(2 * pi) * self.sigma)

    def _cdf(self, x):
        return (1 / 2) * erfc((self.mu - x) / (sqrt(2) * self.sigma))


if __name__ == "__main__":
    nd = NormalDistribution(0, 1)
    print(nd.cdf(first=0.1, end=0.9, step=0.1))
    print(nd.n_std())
    print(nd.n_skewness())
    print(nd.n_kurtosis())
