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
from easy_utils.number_utils.calculus_utils import newton_method
from data_utils.stochastic_utils.vdistributions.abstract import AbstractDistribution, eps
from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution
from data_utils.stochastic_utils.vdistributions.nonparametric.continuous.histogram import freedman_diaconis

# 外部模块
from scipy.stats import t, iqr
import numpy


# 代码块

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


class GaussianKernel(NormalDistribution):
    """
    高斯核
    """

    def pdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        return numpy.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2)) / (self.sigma * numpy.sqrt(2 * numpy.pi))
        # return numpy.exp(-((x - self.mu) ** 2 / (2 * self.sigma ** 2))) / numpy.sqrt(2 * numpy.pi)


class KernelMixDist(AbstractDistribution):
    """混合核分布"""

    def __init__(self, data, h: float = None, kernel_num: int = None):
        data = numpy.asarray(data)
        self.ppf_guess = numpy.mean(data)
        self.len = data.size
        # self.h = silverman_bandwidth(data) if h is None else h
        kernel_num = freedman_diaconis(data) if kernel_num is None else kernel_num

        aggregation_num = ((self.len - 1) // kernel_num) * kernel_num

        matrix = data[:aggregation_num].reshape(-1, kernel_num)

        m = numpy.concatenate((
            numpy.mean(matrix, axis=0), numpy.atleast_1d(numpy.mean(data[aggregation_num:]))
        ))

        h = silverman_bandwidth(data) if h is None else h

        self.kernels = [
            GaussianKernel(mi, h) for i, mi in enumerate(m)
        ]

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
        x = numpy.atleast_1d(numpy.asarray(x))
        results = numpy.empty_like(x, dtype=float)

        for i, xi in enumerate(x):
            def _cdf(q):
                return self.cdf(q) - xi

            results[i], _ = newton_method(_cdf, self.ppf_guess)

        return results if results.shape[0] > 1 else results[0]


if __name__ == "__main__":
    from matplotlib import pyplot

    data = NormalDistribution(0, 1).rvf(100)
    print(KernelMixDist(data))

    curve_index = 2

    pyplot.scatter(NormalDistribution(0, 1).curves(1000)[curve_index][:, 0],
                   NormalDistribution(0, 1).curves(1000)[curve_index][:, 1])
    pyplot.scatter(KernelMixDist(data, h=0.9).curves(1000)[curve_index][:, 0],
                   KernelMixDist(data, h=0.9).curves(1000)[curve_index][:, 1])
    pyplot.show()
