#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""指定参数的核分布"""

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
from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution
from data_utils.stochastic_utils.vdistributions.parameter.abstract import ParameterDistribution

# 外部模块
import numpy
from scipy.stats import t, iqr
from scipy.interpolate import interp1d
import numpy


# 代码块

class GaussianKernelMixDistribution(AbstractDistribution):

    def __init__(self, *args):
        super().__init__()
        self.args = args
        self.kernels = [
            NormalDistribution(i[0], i[1]) for i in args
        ]

        self.domain_min = self.kernels[0].domain().min
        self.domain_max = self.kernels[-1].domain().max

        x_grid = numpy.linspace(self.domain_min, self.domain_max, 1000)
        cdf_vals = self.cdf(x_grid)
        self.ppf_inter = interp1d(cdf_vals, x_grid, bounds_error=False, fill_value=(x_grid[0], x_grid[-1]))

    def __str__(self):
        return str({self.__class__.__name__: self.kernels})

    def __repr__(self):
        return str({self.__class__.__name__: self.kernels})

    def kernel_data(self, sort_index=0):
        d = []
        for k in self.kernels:
            d.append([k.mu, k.sigma])
        d = numpy.asarray(d)
        if sort_index is None:
            return d
        else:
            return d[numpy.argsort(d[:, sort_index])]

    def pdf(self, x, *args, **kwargs):
        x = numpy.atleast_1d(x)
        m = numpy.stack([k.pdf(x) for k in self.kernels], axis=0)
        r = numpy.sum(m, axis=0) / len(self.kernels)
        return r if r.shape != (1,) else r[0]

    def cdf(self, x, *args, **kwargs):
        x = numpy.atleast_1d(x)
        m = numpy.stack([k.cdf(x) for k in self.kernels], axis=0)
        r = numpy.sum(m, axis=0) / len(self.kernels)
        return r if r.shape != (1,) else r[0]

    def ppf(self, x, *args, **kwargs):
        x = numpy.atleast_1d(x)
        result = self.ppf_inter(x)
        r = numpy.clip(result, self.domain_min, self.domain_max)
        return r if r.shape != (1,) else r[0]


class GaussianKernelWeightedMixDistribution(AbstractDistribution):

    def __init__(self, *args):
        super().__init__()
        self.args = args
        self.kernels = []
        self.weights = []

        for i in args:
            mu, sigma, weight = i
            self.kernels.append(NormalDistribution(mu, sigma))
            self.weights.append(weight)

        self.weights = numpy.asarray(self.weights, dtype=float)
        weight_sum = numpy.sum(self.weights)
        if weight_sum == 0:
            raise ValueError("Sum of weights must be > 0")
        self.weights /= weight_sum  # 归一化

        self.domain_min = self.kernels[0].domain().min
        self.domain_max = self.kernels[-1].domain().max

        x_grid = numpy.linspace(self.domain_min, self.domain_max, 1000)
        cdf_vals = self.cdf(x_grid)
        self.ppf_inter = interp1d(cdf_vals, x_grid, bounds_error=False, fill_value=(x_grid[0], x_grid[-1]))

    def __str__(self):
        return str({self.__class__.__name__: self.kernels})

    def __repr__(self):
        return str({self.__class__.__name__: self.kernels})

    def kernel_data(self, sort_index=0):
        d = []
        for k, w in zip(self.kernels, self.weights):
            d.append([k.mu, k.sigma, w])
        d = numpy.asarray(d)
        return d[numpy.argsort(d[:, sort_index])]

    def pdf(self, x, *args, **kwargs):
        x = numpy.atleast_1d(x)
        m = numpy.stack([k.pdf(x) for k in self.kernels], axis=0)
        weighted_sum = numpy.sum(self.weights[:, None] * m, axis=0)
        return weighted_sum if weighted_sum.shape != (1,) else weighted_sum[0]

    def cdf(self, x, *args, **kwargs):
        x = numpy.atleast_1d(x)
        m = numpy.stack([k.cdf(x) for k in self.kernels], axis=0)
        weighted_sum = numpy.sum(self.weights[:, None] * m, axis=0)
        return weighted_sum if weighted_sum.shape != (1,) else weighted_sum[0]

    def ppf(self, x, *args, **kwargs):
        x = numpy.atleast_1d(x)
        result = self.ppf_inter(x)
        cliped_result = numpy.clip(result, self.domain_min, self.domain_max)
        return cliped_result if cliped_result.shape != (1,) else cliped_result[0]

def divergenced_gaussian_kernel_mix_distribution(dist: GaussianKernelMixDistribution, kl_divergence_value:float):


if __name__ == "__main__":
    gkmd = GaussianKernelWeightedMixDistribution((0, 0.1, 1), (0, 0.1, 1))
    # print(gkmd.rvf(100))
    print(gkmd.ppf(0.1))
