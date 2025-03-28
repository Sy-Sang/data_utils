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
from data_utils.stochastic_utils.vdistributions.nonparametric.continuous.histogram import HistogramDistribution, \
    freedman_diaconis

# 外部模块
from scipy.stats import t, iqr
from scipy.interpolate import interp1d
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

    def __init__(self, data, kernel_num: int = None):
        data = numpy.sort(numpy.asarray(data))
        self.len = data.size
        kernel_num = freedman_diaconis(data) if kernel_num is None else kernel_num

        adjusted_len = ((self.len // kernel_num) + 1) * kernel_num
        interp = interp1d(numpy.arange(0, self.len, 1), data, fill_value="extrapolate")
        extrapolate = interp(numpy.arange(self.len, adjusted_len, 1))

        data = numpy.concatenate((data, extrapolate))

        matrix = data.reshape(-1, kernel_num)

        m = numpy.mean(matrix, axis=1)

        h = silverman_bandwidth(m)

        self.kernels = [
            GaussianKernel(mi, h) for i, mi in enumerate(m)
        ]
        self.domain_min = self.kernels[0].domain().min
        self.domain_max = self.kernels[-1].domain().max

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

    def ppf_guess(self, x):
        """牛顿法猜测值"""
        x = numpy.asarray(x)
        m = numpy.stack([k.ppf(x) for k in self.kernels], axis=0)
        r = numpy.sum(m, axis=0) / len(self.kernels)
        return r

    def ppf(self, x, *args, **kwargs):
        x = numpy.atleast_1d(numpy.asarray(x))
        guess = self.ppf_guess(x)
        results = numpy.empty_like(x, dtype=float)

        for i, xi in enumerate(x):
            def _cdf(q):
                return self.cdf(q) - xi

            results[i], _ = newton_method(_cdf, guess[i])

        return numpy.clip(results if results.shape[0] > 1 else results[0], self.domain_min, self.domain_max)


class LogKernelMixDist(KernelMixDist):
    def __init__(self, data, kernel_num: int = None):
        data = numpy.asarray(data)
        self.diff = 1 - numpy.min(data)
        super().__init__(numpy.log(data + self.diff), kernel_num)

    def ppf(self, x, *args, **kwargs):
        """存在问题2025-3-28 12:01:59"""
        return numpy.e ** super().ppf(x) - self.diff

    def pdf(self, x, *args, **kwargs):
        x = numpy.asarray(x) + self.diff
        return super().pdf(numpy.log(x)) / x

    def cdf(self, x, *args, **kwargs):
        x = numpy.asarray(x) + self.diff
        return super().cdf(numpy.log(x))


if __name__ == "__main__":
    from data_utils.stochastic_utils.vdistributions.parameter.continuous.lifetime import WeibullDistribution
    from matplotlib import pyplot

    data = WeibullDistribution(2, 5).rvf(1000)
    print(KernelMixDist(data))
    # print(LogKernelMixDist(data))

    curve_index = 1

    pyplot.scatter(WeibullDistribution(2, 5).curves(100)[curve_index][:, 0],
                   WeibullDistribution(2, 5).curves(100)[curve_index][:, 1])
    pyplot.scatter(LogKernelMixDist(data).curves(100)[curve_index][:, 0],
                   LogKernelMixDist(data).curves(100)[curve_index][:, 1])
    # pyplot.scatter(LogKernelMixDist(data).curves(1000)[curve_index][:, 0],
    #                LogKernelMixDist(data).curves(1000)[curve_index][:, 1])
    pyplot.show()
