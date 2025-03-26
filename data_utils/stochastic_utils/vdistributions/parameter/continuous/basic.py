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
from data_utils.stochastic_utils.vdistributions.abstract import eps
from data_utils.stochastic_utils.vdistributions.parameter.abstract import ParameterDistribution, DistributionParams

# 外部模块
import numpy
from scipy.special import betaincinv, beta, iv, gamma, erfinv, erfcinv, betainc, erfc, erf


# 代码块

class NormalDistribution(ParameterDistribution):
    """正态分布"""

    def __init__(self, mu=0, sigma=1):
        super().__init__(mu, sigma, **{"mu": mu, "sigma": sigma})
        self.mu = mu
        self.sigma = sigma

    def get_param_constraints(self) -> list[DistributionParams]:
        return [
            DistributionParams("mu", -numpy.inf, numpy.inf),
            DistributionParams("sigma", 0 + eps, numpy.inf)
        ]

    def ppf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        r = numpy.where(
            (x > 0) & (x < 1),
            self.mu - numpy.sqrt(2) * self.sigma * erfcinv(2 * x),
            numpy.nan
        )
        return r

    def pdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        r = numpy.e ** (-((-self.mu + x) ** 2 / (2 * self.sigma ** 2))) / (numpy.sqrt(2 * numpy.pi) * self.sigma)
        return r

    def cdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        return 0.5 * erfc((self.mu - x) / (numpy.sqrt(2) * self.sigma))


class LogNormalDistribution(NormalDistribution):

    def ppf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        r = numpy.where(
            (x > 0) & (x < 1),
            numpy.exp(self.mu + self.sigma * numpy.sqrt(2) * erfinv(2 * x - 1)),
            numpy.nan
        )
        return r

    def pdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        c = 1 / (x * self.sigma * numpy.sqrt(2 * numpy.pi))
        e = -((numpy.log(x) - self.mu) ** 2) / (2 * self.sigma ** 2)
        return c * numpy.exp(e)

    def cdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        r = numpy.where(
            x > 0,
            0.5 * (1 + erf((numpy.log(x) - self.mu) / (self.sigma * numpy.sqrt(2)))),
            numpy.nan
        )
        return r


class ExponentialDistribution(ParameterDistribution):
    """指数分布"""

    def __init__(self, lam):
        super().__init__(lam, **{"lam": lam})
        self.lam = lam

    def get_param_constraints(self) -> list[DistributionParams]:
        return [DistributionParams("lam", -numpy.inf, numpy.inf)]

    def ppf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        r = numpy.where(
            (x > 0) & (x < 1),
            -numpy.log(1 - x) / self.lam,
            numpy.nan
        )
        return r

    def pdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        r = numpy.where(
            x >= 0,
            numpy.e ** (-self.lam * x) * self.lam,
            0
        )
        return r

    def cdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        r = numpy.where(
            x >= 0,
            1 - numpy.e ** (-self.lam * x),
            0
        )
        return r


if __name__ == "__main__":
    n = LogNormalDistribution()
    print(n.cdf([1, 2, 3, 4, 5]))
    print(n.rvf(100))
