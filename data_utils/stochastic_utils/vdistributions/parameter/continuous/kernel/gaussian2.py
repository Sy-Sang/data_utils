#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""高斯核分布2"""

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
from typing import Union, Self, Tuple, Optional, List
from collections import namedtuple

# 项目模块
from data_utils.stochastic_utils.vdistributions.abstract import AbstractDistribution, eps
from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution
from data_utils.stochastic_utils.vdistributions.parameter.abstract import ParameterDistribution
from data_utils.stochastic_utils.vdistributions.tools.divergence import kl_divergence_continuous, \
    js_divergence_continuous, tv_divergence

# 外部模块
import numpy
from scipy.stats import t, iqr
from scipy.interpolate import interp1d
from scipy.optimize import minimize, differential_evolution


# 代码块

class WeightedGaussianKernelMixDistribution(AbstractDistribution):

    def __init__(self, *args):
        super().__init__()
        self.args = args
        self.kernels = []
        self.weights = []

        for arg in args:
            # mu, sigma, weight = i
            mu, sigma, *rest = arg
            weight = rest[0] if rest else 1

            self.kernels.append(NormalDistribution(mu, numpy.clip(sigma, eps, numpy.inf)))
            self.weights.append(numpy.clip(weight, 0, None))

        self.weights = numpy.clip(numpy.asarray(self.weights, dtype=float), 0, None)
        weight_sum = numpy.sum(self.weights)
        # if weight_sum == 0:
        #     raise ValueError(f"Sum of weights must be > 0, {self.weights}")
        self.weights /= weight_sum  # 归一化

        mins = [k.domain().min for k in self.kernels]
        maxs = [k.domain().max for k in self.kernels]
        self.domain_min, self.domain_max = min(mins), max(maxs)

        x_grid = numpy.linspace(self.domain_min, self.domain_max, 1000)
        cdf_vals = self.cdf(x_grid)
        self.ppf_inter = interp1d(cdf_vals, x_grid, bounds_error=False, fill_value=(x_grid[0], x_grid[-1]))

    def __str__(self):
        return str({self.__class__.__name__: self.kernels, "weight": self.weights})

    def __repr__(self):
        return str({self.__class__.__name__: self.kernels, "weight": self.weights})

    def kernel_weight_data(self, sort_index: int = None):
        d = []
        for k, w in zip(self.kernels, self.weights):
            d.append([k.mu, k.sigma, w])
        d = numpy.asarray(d)
        if sort_index is None:
            return d
        else:
            return d[numpy.argsort(d[:, sort_index])]

    def kernel_data(self, sort_index: int = None):
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

    def add_kernel(self, *new_kernels):
        this_kernels = numpy.column_stack((self.kernel_data(sort_index=None), numpy.ones(len(self.weights))))
        all_kernels = numpy.concatenate((this_kernels, numpy.asarray(new_kernels)))
        return type(self)(*all_kernels)

    def mean(self):
        return numpy.sum(self.kernel_data(None)[:, 0] * self.weights)

    def variance(self):
        kd = self.kernel_data(None)
        return numpy.sum(
            self.weights * (kd[:, 1] ** 2 + (kd[:, 0] - self.mean()) ** 2)
        )

    def tv_divergence_modify(self, target_tv_divergence: float, dist: Self = None):
        def target_function(x):
            x = numpy.asarray(x).reshape(-1, 3)
            new_dist = type(self)(*x)
            new_js = tv_divergence(self, new_dist)
            return (new_js - numpy.clip(target_tv_divergence, 0, 1)) ** 2

        if dist is None:
            dist = self.clone()
        else:
            pass

        dist_kernels = numpy.column_stack((dist.kernel_data(), dist.weights))
        bound_list = []
        for _ in range(len(dist_kernels)):
            bound_list.append((None, None))
            bound_list.append((1e-3, None))
            bound_list.append((eps, None))

        optimize_kernels = minimize(
            target_function,
            dist_kernels.reshape(-1),
            bounds=bound_list,
            method="L-BFGS-B",
            options={'maxiter': 100, 'disp': False}
        )
        optimize_kernels_data = numpy.asarray(optimize_kernels.x).reshape(-1, 3)
        res = type(self)(*optimize_kernels_data)
        return res, tv_divergence(self, res)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # gkmd = GaussianKernelWeightedMixDistribution((0, 0.1, 1), (0, 0.1, 1))
    # # print(gkmd.rvf(100))
    # print(gkmd.ppf(0.1))
    dist = WeightedGaussianKernelMixDistribution([0, 1, 0.1], [10, 1, 0.5])
    new_dist, tv = dist.tv_divergence_modify(0.1, WeightedGaussianKernelMixDistribution([0, 1]))
    print(tv)
    # print(
    #     js_divergence_continuous(dist, new_dist)
    # )
    # print(new_dist)
    _, pdf, _ = dist.curves(1000)
    plt.plot(pdf[:, 0], pdf[:, 1])
    _, pdf, _ = new_dist.curves(1000)
    plt.plot(pdf[:, 0], pdf[:, 1])
    plt.show()
    print(dist.mean())
    print(new_dist.mean())
