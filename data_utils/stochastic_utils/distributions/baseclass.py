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
from easy_utils.obj_utils.enumerable_utils import flatten
from data_utils.serial_utils.series_trans_utils import MinMax

# 外部模块
import numpy

# 代码块

eps = numpy.finfo(float).eps

DistCurve = namedtuple('curve', ['x', 'y'])
Domain = namedtuple("Domain", ["low", "high"])
DistParamDomain = namedtuple("DistParamDomain", ["low", "high", "default"])


class ABCDistribution(ABC):
    """
    概率分布
    """

    parameter_range: dict[str, DistParamDomain] = {}

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

    def __repr__(self):
        return self.__str__()

    def clone(self) -> Self:
        """深度拷贝克隆"""
        return copy.deepcopy(self)

    def _2d_curve(self, f: callable, *args, first, end, step, num, **kwargs) -> Union[float, DistCurve]:
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
                    xarray = numpy.array([self._ppf(x) for x in ppf_xarray]).astype(float)
            c = DistCurve(xarray, numpy.array([f(x) for x in xarray]).astype(float))
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

    def ppf(self, *args, first=0.01, end=0.99, step=0.01, num: float = None, **kwargs) -> Union[float, DistCurve]:
        """ppf曲线"""
        return self._2d_curve(self._ppf, *args, first=first, end=end, step=step, num=num, **kwargs)

    def pdf(self, *args, first=0.01, end=0.99, step=0.01, num: float = None, **kwargs) -> Union[float, DistCurve]:
        """pdf曲线"""
        return self._2d_curve(self._pdf, *args, first=first, end=end, step=step, num=num, **kwargs)

    def cdf(self, *args, first=0.01, end=0.99, step=0.01, num: float = None, **kwargs) -> Union[float, DistCurve]:
        """cdf曲线"""
        return self._2d_curve(self._cdf, *args, first=first, end=end, step=step, num=num, **kwargs)

    def pdf_domain(self) -> Domain:
        """
        pdf函数定义域
        """
        low = self._ppf(0 + eps)
        high = self._ppf(1 - eps)
        d = Domain(low, high)
        return d

    def cdf_domain(self) -> Domain:
        """cdf函数定义域"""
        return self.pdf()

    def rvf(self, num: int = 1) -> numpy.ndarray:
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
        rv = random_array.y[:num]
        return rv

    def random_sample(self) -> float:
        return numpy.asarray(self.rvf(1), dtype=float)[0]

    def limited_rvf(self, domain_list: Union[list, tuple] = (0, 1), num: int = 100) -> Union[numpy.ndarray, float]:
        """
        给定范围的随机数
        """
        rvf_domain = self.pdf_domain()
        domain = Domain(domain_list[0], domain_list[1])

        low = rvf_domain.low if numpy.isinf(domain.low) else numpy.max([rvf_domain.low, domain.low])
        high = rvf_domain.high if numpy.isinf(domain.high) else numpy.min([rvf_domain.high, domain.high])

        n = max(100, num) * 2
        x_array = numpy.random.uniform(self._cdf(low), self._cdf(high), size=n)
        random_array = self.ppf(x_array)
        rv = random_array.y[:num]
        if num == 1:
            return rv[0]
        else:
            return rv

    @abstractmethod
    def mean(self) -> float:
        """均值"""
        pass

    @abstractmethod
    def std(self) -> float:
        """标准差"""
        pass

    # def nmean(self) -> float:
    #     def f(x):
    #         x * self._pdf(x)
    #
    #     domain = self.pdf_domain()
    #     return calculus_utils.simpsons_integrate(f, domain.low, domain.high, 100)
    #
    # def nstd(self) -> float:
    #     def f(x):
    #         (x - self.nmean()) ^ 2 * self._pdf(x)
    #
    #     domain = self.pdf_domain()
    #     return calculus_utils.simpsons_integrate(f, domain.low, domain.high, 100) ** 0.5


class UniformDistribution(ABCDistribution):
    """均匀分布"""

    parameter_range = {
        "low": Domain(-numpy.inf, numpy.inf),
        "high": Domain(-numpy.inf, numpy.inf),
    }

    def __init__(self, low, high):
        super().__init__(low=low, high=high)
        self.low = min(low, high)
        self.high = max(low, high)

    def _pdf(self, x):
        return 1 / (self.high - self.low)

    def _cdf(self, x):
        if x < self.low:
            return 0
        elif x > self.high:
            return 1
        else:
            return (x - self.low) / (self.high - self.low)

    def _ppf(self, x):
        if 0 < x < 1:
            return x / (self.high - self.low)
        else:
            return numpy.nan

    def mean(self):
        return (self.high - self.low) / 2

    def std(self):
        return 0

    def rvf(self, num=1):
        if numpy.isinf(self.low) and self.high < numpy.inf:
            rarray = numpy.random.uniform(self.high - 1, self.high, num)
        elif self.low > -numpy.inf and numpy.isinf(self.high):
            rarray = numpy.random.uniform(self.low, self.low + 1, num)
        elif numpy.isinf(self.low) and numpy.isinf(self.high):
            rarray = numpy.random.uniform(0, 1, num)
        else:
            return super().rvf(num)

        if num == 1:
            return rarray[0]
        else:
            return rarray


# def correlated_rvf(dist_list: list, num: int = 100) -> numpy.ndarray:
#     """
#     相关的随机数
#     """
#     rn = max(num, 100)
#     array = numpy.array([])
#     nt = namedtuple("nt", ["dist", "corr"])
#     nt.__annotations__ = {"dist": ABCDistribution, "corr": float}
#     for i, a in enumerate(dist_list):
#         d = nt(a[0], a[1])
#         if i == 0:
#             r = numpy.sort(d.dist.rvf(rn))
#             array = numpy.concatenate((array, r))
#         else:
#             l = int(rn * abs(d.corr))
#             if l != 0:
#                 rx = numpy.random.uniform(
#                     0 + numpy.finfo(float).eps,
#                     1 - numpy.finfo(float).eps,
#                     l
#                 )
#                 if d.corr >= 0:
#                     rx = numpy.sort(rx)
#                 elif d.corr < 0:
#                     rx = numpy.sort(rx)[::-1]
#                 r0 = d.dist.ppf(*rx).y
#             else:
#                 r0 = numpy.array([])
#             r1 = d.dist.rvf(rn - l)
#             r = numpy.concatenate((r0, r1))
#             array = numpy.column_stack((array, r))
#
#     numpy.random.shuffle(array)
#     return array[:num].T
#
#
# def correlated_random_number(base_dist: ABCDistribution, num: int = 100, *args):
#     """
#     生成满足指定概率分布并具备相关性的随机数
#     """
#
#     def noize_f(x: float) -> float:
#         return 0.274882 / (0.0892443 + abs(x)) ** 2.325384114212880
#
#     eps = numpy.finfo(float).eps
#
#     n = max(num, 100)
#     nt = namedtuple("nt", ["dist", "corr"])
#
#     base_quantile = numpy.random.uniform(0 + eps, 1 - eps, n)
#     array = base_dist.ppf(base_quantile).y
#
#     for i, a in enumerate(args):
#         d = nt(a[0], a[1])
#         d.dist: ABCDistribution
#         d.corr: float
#         if d.corr != 0:
#             noize_size = noize_f(d.corr)
#             noize = numpy.random.uniform(0, noize_size, n)
#             noised = base_quantile + noize if d.corr > 0 else noize - base_quantile
#             noize_range = numpy.random.uniform(0 + eps, 1 - eps, n)
#             normalized_noised, _1, _2 = MinMax.f(noised, a=numpy.min(noize_range), b=numpy.max(noize_range))
#             array = numpy.column_stack((array, d.dist.ppf(normalized_noised).y))
#         else:
#             array = numpy.column_stack((array, d.dist.rvf(n)))
#
#     return array[:num].T


if __name__ == "__main__":
    # u = UniformDistribution(-numpy.inf, numpy.inf)
    eps = numpy.finfo(float).eps
    u = UniformDistribution(0 + eps, numpy.inf)
    print(u.low)
    print(u.high)
    print(u.rvf())
