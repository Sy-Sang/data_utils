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

# 项目模块
from dist_utils import ABCDistribution, NormalDistribution
from easy_utils.number_utils import calculus_utils, number_utils

# 外部模块
import numpy
from scipy.stats import t, iqr


# 代码块
def silverman_bandwidth(data) -> float:
    """
    Silverman 规则
    """
    n = len(data)
    std_dev = numpy.std(data, ddof=1)
    ir = iqr(data)

    h = 0.9 * min([std_dev, ir / 1.34]) * n ** (-0.2)

    return h


class GaussianKernel(NormalDistribution):
    """
    高斯核
    """

    def __init__(self, mu, sigma):
        super().__init__(mu, sigma)

    def _pdf(self, x: float) -> float:
        return numpy.exp(-((-self.mu + x) ** 2 / (2 * self.sigma ** 2))) / numpy.sqrt(2 * numpy.pi)


class KernelMixDist(ABCDistribution):
    def __init__(self, data: Union[list, tuple, numpy.ndarray], h=None):
        super().__init__()
        self.data = sorted(data)
        self.h = silverman_bandwidth(self.data) if h is None else h
        self.kernels = [
            GaussianKernel(i, self.h) for i in self.data
        ]

    def _pdf(self, x: float) -> float:
        return sum([k._pdf(x) for k in self.kernels]) * (1 / (len(self.data) * self.h))

    def _cdf(self, x: float) -> float:
        return sum([k._cdf(x) for k in self.kernels]) * (1 / len(self.data))

    def _ppf(self, x: float) -> float:
        if 0 < x < 1:
            def f(y):
                return self.cdf(y) - x

            x0 = numpy.mean(self.data)
            p, error = calculus_utils.newton_method(f, x0)
            return p
        else:
            return numpy.nan


if __name__ == "__main__":
    nd = NormalDistribution(0, 1)
    r = nd.rvf(1000)
    kd = KernelMixDist(r)
    print(kd.n_mean())
    print(kd.n_std())
    print(kd.n_skewness())
    print(kd.n_kurtosis())
