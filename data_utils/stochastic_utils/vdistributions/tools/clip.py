#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""对分布进行剪切"""

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
from data_utils.stochastic_utils.vdistributions.abstract import AbstractDistribution, eps
from data_utils.stochastic_utils.vdistributions.nonparametric.continuous.kernel2 import KernelMixDistribution
from data_utils.stochastic_utils.vdistributions.nonparametric.continuous.histogram import HistogramDistribution

# 外部模块
import numpy
from scipy.integrate import quad

# 代码块

DomainNamedtuple = namedtuple("DomainNamedtuple", ["min", "max"])


class ClippedHistogramDistribution(HistogramDistribution):

    def __init__(self, dist: AbstractDistribution, x_min, x_max, his_num=1000, bin_num: int = None):
        self.dist = dist.clone()
        self.min = x_min
        self.max = x_max
        data = numpy.clip(self.dist.rvf(his_num), self.min, self.max)
        super().__init__(data, bin_num)

    def __repr__(self):
        return str({"distribution": self.dist, "min": self.min, "max": self.max})

    def __str__(self):
        return str({"distribution": self.dist, "min": self.min, "max": self.max})

    def domain(self):
        return DomainNamedtuple(self.min, self.max)


class ClampedDistribution(AbstractDistribution):
    def __init__(self, dist: AbstractDistribution, x_min, x_max):
        super().__init__()
        self.distribution = dist.clone()
        self.min = x_min
        self.max = x_max

    def __repr__(self):
        return str({"distribution": self.distribution, "min": self.min, "max": self.max})

    def __str__(self):
        return str({"distribution": self.distribution, "min": self.min, "max": self.max})

    def domain(self):
        return DomainNamedtuple(self.min, self.max)

    def cdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        F = self.distribution.cdf
        return numpy.where(
            x < self.min,
            0,
            numpy.where(x > self.max, 1, F(x))
        )

    def pdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        left_mass = self.distribution.cdf(self.min)
        right_mass = 1 - self.distribution.cdf(self.max)
        pdf_val = self.distribution.pdf(x)
        result = numpy.empty_like(pdf_val)
        result[(x < self.min) | (x > self.max)] = 0
        result[(x > self.min) & (x < self.max)] = pdf_val[(x > self.min) & (x < self.max)]
        result[x == self.min] = left_mass
        result[x == self.max] = right_mass
        return result

    def ppf(self, q, *args, **kwargs):
        q = numpy.asarray(q)
        q_min = self.distribution.cdf(self.min)
        q_max = self.distribution.cdf(self.max)
        q_clamped = numpy.clip(q, q_min, q_max)
        return self.distribution.ppf(q_clamped)


if __name__ == "__main__":
    from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution
    from data_utils.stochastic_utils.vdistributions.tools.divergence import kl_divergence_continuous, crps, \
        quantile_RMSE
    from matplotlib import pyplot

    chd = ClampedDistribution(NormalDistribution(0, 1), 0, 1)
    # ppf, pdf, cdf = chd.curves()
    # pyplot.plot(pdf[:,0], pdf[:, 1])
    # pyplot.show()
    # pyplot.hist(chd.rvf(1000).tolist())
    # pyplot.show()
    print(quantile_RMSE(chd, NormalDistribution(0, 1)))
    print(quantile_RMSE(NormalDistribution(0, 10), NormalDistribution(0, 1)))
    # print(crps(NormalDistribution(0, 1), 1))
