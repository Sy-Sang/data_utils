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
from typing import Union, Self, Type
from collections import namedtuple

# 项目模块
from easy_utils.number_utils.calculus_utils import newton_method
from data_utils.stochastic_utils.vdistributions.abstract import AbstractDistribution, eps
from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution
from data_utils.stochastic_utils.vdistributions.nonparametric.continuous.histogram import HistogramDistribution, \
    freedman_diaconis

# 外部模块
from scipy.stats import t, iqr
from scipy.interpolate import interp1d
import numpy


# 代码块

def find_closest_divisor(n: int, k: int):
    """最临近整除因子"""
    divisors = set()
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return min(divisors, key=lambda x: abs(x - k))


def silverman_bandwidth(data) -> float:
    """
    Silverman 规则
    """
    data = numpy.asarray(data)
    n = data.size
    std_dev = numpy.std(data, ddof=1)
    ir = iqr(data)

    h = 0.9 * numpy.min([std_dev, ir / 1.34]) * n ** (-0.2)

    return h


class KernelMixDistribution(AbstractDistribution):
    """混合核分布"""

    def __init__(self, data, kernel_num: int = None):
        super().__init__()
        data = numpy.sort(numpy.asarray(data))
        self.len = data.size
        kernel_len = freedman_diaconis(data) if kernel_num is None else kernel_num

        if kernel_num is None or kernel_len < self.len:
            kernel_len = find_closest_divisor(self.len, kernel_len)
            matrix = data.reshape(-1, kernel_len)
            m = numpy.mean(matrix, axis=1)
        else:
            m = data
        h = silverman_bandwidth(m)
        self.kernels = [
            NormalDistribution(mi, h) for i, mi in enumerate(m)
        ]
        self.domain_min = self.kernels[0].domain().min
        self.domain_max = self.kernels[-1].domain().max

        x_grid = numpy.linspace(self.domain_min, self.domain_max, 1000)
        cdf_vals = self.cdf(x_grid)
        # for i in range(1, len(cdf_vals)):
        #     if cdf_vals[i] <= cdf_vals[i - 1]:
        #         cdf_vals[i] = cdf_vals[i - 1] + eps
        self.ppf_inter = interp1d(cdf_vals, x_grid, bounds_error=False, fill_value=(x_grid[0], x_grid[-1]))

    def __str__(self):
        return str({self.__class__.__name__: self.kernels})

    def __repr__(self):
        return str({self.__class__.__name__: self.kernels})

    def pdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        m = numpy.stack([k.pdf(x) for k in self.kernels], axis=0)
        r = numpy.sum(m, axis=0) / len(self.kernels)
        return r

    def cdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        m = numpy.stack([k.cdf(x) for k in self.kernels], axis=0)
        r = numpy.sum(m, axis=0) / len(self.kernels)
        return r

    def ppf(self, x, *args, **kwargs):
        x = numpy.atleast_1d(x)
        result = self.ppf_inter(x)
        return numpy.clip(result, self.domain_min, self.domain_max)


if __name__ == "__main__":
    from data_utils.stochastic_utils.vdistributions.parameter.continuous.lifetime import WeibullDistribution
    from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import LogNormalDistribution
    from matplotlib import pyplot

    data = WeibullDistribution(2, 5).rvf(5000)
    print(KernelMixDistribution(data))
    # print(LogKernelMixDist(data))

    curve_index = 1

    pyplot.scatter(WeibullDistribution(2, 5).curves(100)[curve_index][:, 0],
                   WeibullDistribution(2, 5).curves(100)[curve_index][:, 1])
    pyplot.scatter(KernelMixDistribution(data).curves(100)[curve_index][:, 0],
                   KernelMixDistribution(data).curves(100)[curve_index][:, 1])
    # pyplot.scatter(LogKernelMixDist(data).curves(1000)[curve_index][:, 0],
    #                LogKernelMixDist(data).curves(1000)[curve_index][:, 1])
    pyplot.show()
