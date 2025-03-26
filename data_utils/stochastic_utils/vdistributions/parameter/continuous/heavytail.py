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
from scipy.optimize import brentq
from scipy.special import betaincinv, beta, iv, gamma, erfinv, erfcinv, betainc


# 代码块

class StudentTDistribution(ParameterDistribution):
    """学生t分布"""

    def __init__(self, u=0, s=1, v=1):
        super().__init__(u, s, v, **{"u": 0, "s": s, "v": v})
        self.u = u
        self.s = s
        self.v = v

    def get_param_constraints(self) -> list[DistributionParams]:
        return [
            DistributionParams("u", -numpy.inf, numpy.inf),
            DistributionParams("s", 0 + eps, numpy.inf),
            DistributionParams("v", 0 + eps, numpy.inf),
        ]

    def pdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        numerator = (self.v / (self.v + (-self.u + x) ** 2 / self.s ** 2)) ** ((1 + self.v) / 2)
        denominator = self.s * numpy.sqrt(self.v) * beta(self.v / 2, 0.5)
        r = numerator / denominator
        return r

    def cdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)

        s2v = self.s ** 2 * self.v
        delta = x - self.u
        delta2 = delta ** 2

        below = 0.5 * betainc(self.v / 2, 1 / 2, s2v / (s2v + delta2))
        above = 0.5 * (1 + betainc(1 / 2, self.v / 2, delta2 / (s2v + delta2)))

        result = numpy.where(x <= self.u, below, above)
        return result

    def ppf(self, q, *args, **kwargs):
        q = numpy.atleast_1d(q)
        results = numpy.empty_like(q, dtype=float)

        a = self.u - 10 * self.s
        b = self.u + 10 * self.s

        for i, qi in enumerate(q):
            def objective(x): return float(self.cdf(x) - qi)

            results[i] = brentq(objective, a, b)

        return results if results.shape[0] > 1 else results[0]


if __name__ == "__main__":
    s = StudentTDistribution()
    print(s.ppf(numpy.arange(0.1, 1, 0.01)))
