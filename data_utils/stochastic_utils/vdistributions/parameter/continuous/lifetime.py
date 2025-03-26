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

class WeibullDistribution(ParameterDistribution):
    """威布尔分布"""

    def __init__(self, alpha, beta, miu=0):
        super().__init__(alpha, beta, miu, **{"alpha": alpha, "beta": beta, "miu": miu})
        self.alpha = alpha
        self.beta = beta
        self.miu = miu

    def get_param_constraints(self) -> list[DistributionParams]:
        return [
            DistributionParams("alpha", 0 + eps, numpy.inf),
            DistributionParams("beta", 0 + eps, numpy.inf),
            DistributionParams("miu", -numpy.inf, numpy.inf),
        ]

    def ppf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        r = numpy.where(
            (x > 0) & (x < 1),
            self.miu + self.beta * (-numpy.log(1 - x)) ** (1 / self.alpha),
            numpy.nan
        )
        return r

    def pdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        r = numpy.where(
            x > self.miu,
            (self.alpha * numpy.e ** -((-self.miu + x) / self.beta) ** self.alpha * ((-self.miu + x) / self.beta) ** (
                    -1 + self.alpha)) / self.beta,
            0
        )
        return r

    def cdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        r = numpy.where(
            x > self.miu,
            1 - numpy.e ** -((-self.miu + x) / self.beta) ** self.alpha,
            0
        )
        return r


if __name__ == "__main__":
    w = WeibullDistribution(2, 5)
    print(w.rvf(100))
