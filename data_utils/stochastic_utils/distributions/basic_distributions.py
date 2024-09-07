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
import math

# 项目模块
from data_utils.stochastic_utils.distributions.baseclass import ABCDistribution, DistParamDomain
from easy_utils.number_utils.calculus_utils import newton_method

# 外部模块
import numpy
from scipy.special import betaincinv, beta, iv, gamma, erfinv, erfcinv, betainc


# 代码块


class NormalDistribution(ABCDistribution):
    """
    正态分布
    """

    parameter_range = {
        "mu": DistParamDomain(-numpy.inf, numpy.inf, 0),
        "std": DistParamDomain(0 + numpy.finfo(float).eps, numpy.inf, 1),
    }

    def __init__(self, mu: Union[float, numpy.floating] = 0, sigma: Union[float, numpy.floating] = 1):
        super().__init__(mu=mu, sigma=sigma)
        self.mu = mu
        self.sigma = sigma

    def _ppf(self, x):
        if 0 < x < 1:
            return self.mu - numpy.sqrt(2) * self.sigma * erfcinv(2 * x)
        else:
            return numpy.nan

    def _pdf(self, x):
        return numpy.e ** (-((-self.mu + x) ** 2 / (2 * self.sigma ** 2))) / (numpy.sqrt(2 * numpy.pi) * self.sigma)

    def _cdf(self, x):
        return (1 / 2) * math.erfc((self.mu - x) / (numpy.sqrt(2) * self.sigma))

    def mean(self):
        return self.mu

    def std(self):
        return self.sigma


class LogNormalDistribution(ABCDistribution):
    """
    对数正态分布
    """

    parameter_range = {
        "mu": DistParamDomain(-numpy.inf, numpy.inf, 0),
        "std": DistParamDomain(0 + numpy.finfo(float).eps, numpy.inf, 1)
    }

    def __init__(self, mu: Union[float, numpy.floating] = 0, sigma: Union[float, numpy.floating] = 1):
        """

        :param mu:
        :param sigma:
        """
        super().__init__(mu=mu, sigma=sigma)
        self.mu = mu
        self.sigma = sigma

    def _ppf(self, x):
        if 0 < x < 1:
            return numpy.exp(self.mu + self.sigma * numpy.sqrt(2) * erfinv(2 * x - 1))
        else:
            return numpy.nan

    def _pdf(self, x):
        if x > 0:
            c = 1 / (x * self.sigma * numpy.sqrt(2 * numpy.pi))
            e = -((numpy.log(x) - self.mu) ** 2) / (2 * self.sigma ** 2)
            return c * numpy.exp(e)
        else:
            return 0

    def _cdf(self, x):
        if x > 0:
            return 0.5 * (1 + math.erf((numpy.log(x) - self.mu) / (self.sigma * numpy.sqrt(2))))
        else:
            return 0

    def mean(self):
        return numpy.exp(self.mu + (self.sigma ** 2) / 2)

    def std(self):
        return numpy.sqrt(numpy.exp(2 * self.mu + self.sigma ** 2) * (-1 + numpy.exp(self.sigma ** 2)))


class WeibullDistribution(ABCDistribution):
    """威布尔分布"""

    parameter_range = {
        "alpha": DistParamDomain(0 + numpy.finfo(float).eps, numpy.inf, 2),
        "beta": DistParamDomain(0 + numpy.finfo(float).eps, numpy.inf, 5),
        "miu": DistParamDomain(-numpy.inf, numpy.inf, 0)
    }

    def __init__(self, alpha: Union[float, numpy.floating], beta: Union[float, numpy.floating],
                 miu: Union[float, numpy.floating] = 0):
        """
        威布尔分布构造函数
        :param alpha: 形状参数α
        :param beta: 尺度参数β
        :param miu: 位置参数μ, 默认为0
        """
        super().__init__(alpha=alpha, beta=beta, miu=miu)
        self.alpha = alpha
        self.beta = beta
        self.miu = miu

    def _ppf(self, x):
        if 0 < x < 1:
            return self.miu + self.beta * (-numpy.log(1 - x)) ** (1 / self.alpha)
        else:
            return numpy.nan

    def _pdf(self, x):
        if x > self.miu:
            return (self.alpha * numpy.e ** -((-self.miu + x) / self.beta) ** self.alpha * (
                    (-self.miu + x) / self.beta) ** (-1 + self.alpha)) / self.beta
        else:
            return 0

    def _cdf(self, x):
        if x > self.miu:
            return 1 - numpy.e ** -((-self.miu + x) / self.beta) ** self.alpha
        else:
            return 0

    def mean(self):
        return self.miu + self.beta * gamma(1 + 1 / self.alpha)

    def std(self):
        return self.beta * numpy.sqrt(-gamma(1 + 1 / self.alpha) ** 2 + gamma(1 + 2 / self.alpha))


class StudentTDistribution(ABCDistribution):
    """
    student T 分布
    """

    parameter_range = {
        "u": DistParamDomain(-numpy.inf, numpy.inf, 0),
        "s": DistParamDomain(0 + numpy.finfo(float).eps, numpy.inf, 1),
        "v": DistParamDomain(0 + numpy.finfo(float).eps, numpy.inf, 1)
    }

    def __init__(self, u: Union[float, numpy.floating] = 0, s: Union[float, numpy.floating] = 1,
                 v: Union[float, numpy.floating] = 1):
        """
        表示由定位参数 u、尺度参数 s 和自由度 v 决定的学生 t 分布.
        :param u:
        :param s:
        :param v:
        """
        super().__init__(u=u, s=s, v=v)
        self.u = u
        self.s = s
        self.v = v

    def _pdf(self, x):
        numerator = (self.v / (self.v + (-self.u + x) ** 2 / self.s ** 2)) ** ((1 + self.v) / 2)
        denominator = self.s * numpy.sqrt(self.v) * beta(self.v / 2, 1 / 2)
        result = numerator / denominator
        return result

    def _cdf(self, x):
        if x <= self.u:
            numerator = self.s ** 2 * self.v
            denominator = self.s ** 2 * self.v + (-self.u + x) ** 2
            result = 0.5 * betainc(self.v / 2, 1 / 2, numerator / denominator)
            return result
        else:
            numerator = (-self.u + x) ** 2
            denominator = self.s ** 2 * self.v + (-self.u + x) ** 2
            result = 0.5 * (1 + betainc(1 / 2, self.v / 2, numerator / denominator))
            return result

    def _ppf(self, x):
        def f(y):
            return self.cdf(y) - x

        if 0 < x < 1:
            guess = 0
            q, _ = newton_method(f, guess)
            return q
        else:
            return numpy.nan

    def mean(self) -> float:
        pass

    def std(self) -> float:
        pass


if __name__ == "__main__":
    lnd = WeibullDistribution(1, 1)
    print(lnd.limited_rvf([0, 100], 20))

    # print(correlated_rvf([
    #     [NormalDistribution(0, 1), 1],
    #     [NormalDistribution(0, 1), 0.5],
    #     [LogNormalDistribution(0, 1), -0.5],
    #     [NormalDistribution(0, 1), 0],
    #     [WeibullDistribution(2, 5), 0.3]
    # ], 100).tolist())

    # print(correlated_random_number(
    #     WeibullDistribution(2, 5),
    #     100,
    #     [WeibullDistribution(2, 5), -0.5],
    #     [StudentTDistribution(v=3), 0.5],
    #     [NormalDistribution(0, 1), 0.8],
    # ).tolist())
