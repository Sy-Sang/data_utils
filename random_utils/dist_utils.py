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
#
# def curve_wrapper(f:callable):
#     def wrapper(instance, *args, **kwargs):
#
#

class distribution(ABC):
    """
    概率分布
    """

    def _2d_curve(self, f: callable, *args, first=0.01, end=0.99, step=0.01) -> Union[float, numpy.ndarray]:
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
                xarray = self.ppf(first=first, end=end, step=step)[:, 1]
                return numpy.array([[x, f(x)] for x in xarray])

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

    @abstractmethod
    def ppf(self, *args, **kwargs) -> Union[float, numpy.ndarray]:
        """
        分位数函数
        """
        pass

    @abstractmethod
    def pdf(self, *args, **kwargs) -> Union[float, numpy.ndarray]:
        """
        pdf
        """
        pass

    @abstractmethod
    def cdf(self, *args, **kwargs) -> Union[float, numpy.ndarray]:
        """
        cdf
        """
        pass

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


class NormalDistribution(distribution):
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

    def ppf(self, *args, **kwargs):
        return self._2d_curve(self._ppf, *args, **kwargs)

    def pdf(self, *args, **kwargs):
        return self._2d_curve(self._pdf, *args, **kwargs)

    def cdf(self, *args, **kwargs):
        return self._2d_curve(self._cdf, *args, **kwargs)


if __name__ == "__main__":
    nd = NormalDistribution(0, 1)
    print(nd.cdf(first=0.1, end=0.9, step=0.1))
