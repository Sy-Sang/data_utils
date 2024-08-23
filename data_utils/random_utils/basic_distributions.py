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
from collections import namedtuple
import math

# 项目模块
from dist_utils import ABCDistribution, correlated_rvf, correlated_random_number

# 外部模块
import numpy
from scipy.special import betaincinv, beta, iv, gamma, erfinv, erfcinv, betainc


# 代码块

class NormalDistribution(ABCDistribution):
    def __init__(self, mu: float = 0, sigma: float = 1):
        super().__init__()
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
    def __init__(self, mu: float = 0, sigma: float = 1):
        """

        :param mu:
        :param sigma:
        """
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def _ppf(self, x):
        if 0 < x < 1:
            return numpy.exp(self.mu + self.sigma * numpy.sqrt(2) * erfinv(2 * x - 1))
        else:
            return numpy.nan

    def _pdf(self, x):
        c = 1 / (x * self.sigma * numpy.sqrt(2 * numpy.pi))
        e = -((numpy.log(x) - self.mu) ** 2) / (2 * self.sigma ** 2)
        return c * numpy.exp(e)

    def _cdf(self, x):
        return 0.5 * (1 + math.erf((numpy.log(x) - self.mu) / (self.sigma * numpy.sqrt(2))))

    def mean(self):
        return numpy.exp(self.mu + (self.sigma ** 2) / 2)

    def std(self):
        return numpy.sqrt(numpy.exp(2 * self.mu + self.sigma ** 2) * (-1 + numpy.exp(self.sigma ** 2)))


if __name__ == "__main__":
    # from easy_utils.number_utils import calculus_utils, number_utils
    #
    # nd = LogNormalDistribution(0, 1)
    # m = calculus_utils.simpsons_integrate
    # n = 100
    # print(nd.pdf(first=0 + numpy.finfo(float).eps, end=1 - numpy.finfo(float).eps, step=0.01).tolist())
    # print(nd.n_mean(m, num=n))
    # print(nd.n_std(m, num=n))
    # print(nd.n_skewness(m, num=n))
    # print(nd.n_kurtosis(m, num=n))
    # print(nd.mean())
    # print(nd.std())

    print(correlated_random_number(
        [NormalDistribution(0, 1), 0.9],
        [NormalDistribution(0, 1), 0.5],
        [LogNormalDistribution(0, 1), -0.5],
        [NormalDistribution(0, 1), 0],
        num=50
    ).tolist())
