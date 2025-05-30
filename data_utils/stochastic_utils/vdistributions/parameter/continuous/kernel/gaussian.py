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
from data_utils.stochastic_utils.vdistributions.tools.divergence import kl_divergence_continuous, \
    js_divergence_continuous

# 外部模块
import numpy
from scipy.stats import t, iqr
from scipy.interpolate import interp1d
from scipy.optimize import minimize


# 代码块

class KernelMixDistribution(AbstractDistribution):
    """核混合分布"""

    def kernel_data(self, *args, **kwargs) -> numpy.ndarray:
        pass


class GaussianKernelMixDistribution(KernelMixDistribution):

    def __init__(self, *args):
        super().__init__()
        self.args = args
        self.kernels = [
            NormalDistribution(i[0], numpy.clip(i[1], eps, numpy.inf)) for i in args
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


class WeightedGaussianKernelMixDistribution(KernelMixDistribution):

    def __init__(self, *args):
        super().__init__()
        self.args = args
        self.kernels = []
        self.weights = []

        for i in args:
            mu, sigma, weight = i
            self.kernels.append(NormalDistribution(mu, numpy.clip(sigma, eps, numpy.inf)))
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
        this_kernels = self.kernel_weight_data(sort_index=None)
        all_kernels = numpy.concatenate((this_kernels, numpy.asarray(new_kernels)))
        return type(self)(*all_kernels)


def divergenced_gaussian_kernel_mix_distribution(
        dist: KernelMixDistribution,
        kl_divergence_value: float,
        kernel_num: int = None
) -> GaussianKernelMixDistribution:
    """kl散度约束的混合核分布"""

    def target_function(x):
        x = numpy.asarray(x).reshape(-1, 2)
        new_dist = GaussianKernelMixDistribution(*x)
        kl = js_divergence_continuous(dist, new_dist)
        return (kl - numpy.clip(kl_divergence_value, 0, 0.6931)) ** 2

    kernels = dist.kernel_data()
    # kernels = numpy.asarray([[0, 1]] * len(dist.kernel_data()))
    kernel_num = int(numpy.clip(kernel_num, 1, numpy.inf)) if kernel_num is not None else len(kernels)

    additional_kernel_num = kernel_num - len(kernels)

    if additional_kernel_num == 0:
        additional_kernel = numpy.asarray([])
        method = "trust-constr"
    elif additional_kernel_num > 0:
        additional_kernel = numpy.asarray([0, 1] * additional_kernel_num)
        method = "L-BFGS-B"
    else:
        additional_kernel = numpy.asarray([])
        kernels = kernels[:int(numpy.clip(kernel_num, 1, numpy.inf))]
        method = "L-BFGS-B"

    bound_list = []
    for _ in range(kernel_num):
        bound_list.append((None, None))
        bound_list.append((1e-3, None))
        # bound_list.append((0, 1))

    optimize_kernels = minimize(
        target_function,
        numpy.concatenate((kernels.reshape(-1), additional_kernel)),
        bounds=bound_list,
        method=method,
        options={'maxiter': 100, 'disp': False}
    )
    optimize_kernels_data = numpy.asarray(optimize_kernels.x).reshape(-1, 2)
    return GaussianKernelMixDistribution(*optimize_kernels_data)


def divergenced_weight_kernel_mix_distribution(
        dist: KernelMixDistribution,
        kl_divergence_value: float,
        kernel_num: int = None
) -> WeightedGaussianKernelMixDistribution:
    """kl散度约束的混合核分布"""

    def target_function(x):
        x = numpy.asarray(x).reshape(-1, 3)
        new_dist = WeightedGaussianKernelMixDistribution(*x)
        kl = js_divergence_continuous(dist, new_dist)
        return (kl - numpy.clip(kl_divergence_value, 0, 0.6931)) ** 2

    kernels = dist.kernel_data()
    kernels = numpy.column_stack((kernels, numpy.random.uniform(0, 1, len(kernels))))

    # kernels = numpy.asarray([[0, 1, 1]] * len(dist.kernel_data()))
    kernel_num = int(numpy.clip(kernel_num, 1, numpy.inf)) if kernel_num is not None else len(kernels)

    additional_kernel_num = kernel_num - len(kernels)

    if additional_kernel_num == 0:
        additional_kernel = numpy.asarray([])
        method = "trust-constr"
    elif additional_kernel_num > 0:
        additional_kernel = numpy.asarray([numpy.random.uniform(-1, 1), 1, 1] * additional_kernel_num)
        method = "L-BFGS-B"
    else:
        additional_kernel = numpy.asarray([])
        kernels = kernels[:int(numpy.clip(kernel_num, 1, numpy.inf))]
        method = "L-BFGS-B"

    bound_list = []
    for _ in range(kernel_num):
        bound_list.append((None, None))
        bound_list.append((1e-3, None))
        bound_list.append((0, 1))

    optimize_kernels = minimize(
        target_function,
        numpy.concatenate((kernels.reshape(-1), additional_kernel)),
        bounds=bound_list,
        method=method,
        options={'maxiter': 100, 'disp': False}
    )
    optimize_kernels_data = numpy.asarray(optimize_kernels.x).reshape(-1, 3)
    return WeightedGaussianKernelMixDistribution(*optimize_kernels_data)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # gkmd = GaussianKernelWeightedMixDistribution((0, 0.1, 1), (0, 0.1, 1))
    # # print(gkmd.rvf(100))
    # print(gkmd.ppf(0.1))
    dist = WeightedGaussianKernelMixDistribution([0, 1, 1])
    new_dist = divergenced_weight_kernel_mix_distribution(dist, 0.3, None)
    print(
        js_divergence_continuous(dist, new_dist)
    )
    print(new_dist)
    _, pdf, _ = dist.curves(1000)
    plt.plot(pdf[:, 0], pdf[:, 1])
    _, pdf, _ = new_dist.curves(1000)
    plt.plot(pdf[:, 0], pdf[:, 1])
    plt.show()
