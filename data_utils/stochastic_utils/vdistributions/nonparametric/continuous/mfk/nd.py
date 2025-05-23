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
from data_utils.stochastic_utils.vdistributions.abstract import AbstractDistribution, eps
from data_utils.stochastic_utils.vdistributions.nonparametric.continuous.kernel2 import KernelMixDistribution
from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution

# 外部模块
import numpy
from scipy.optimize import differential_evolution, minimize
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d


# 代码块

def data_moment(data):
    """数据的矩"""
    data = numpy.asarray(data)
    return numpy.asarray([numpy.mean(data), numpy.var(data), skew(data), kurtosis(data)])


def moment_loss(data, mu_target, var_target, skew_target, kurt_target, method="basic"):
    """矩损失函数"""
    moment = data_moment(data)
    target = numpy.asarray([mu_target, var_target, skew_target, kurt_target])
    if method == "basic":
        loss = numpy.sum((moment - target) ** 2)
        return loss
    elif method == "minmax":
        mm = numpy.concatenate((moment, target))
        mm = ((mm - numpy.min(mm)) / (numpy.max(mm) + 1e-8 - numpy.min(mm))).reshape(2, 4)
        return numpy.sum(
            (mm[0] - mm[1]) ** 2
        )
    elif method == "softmax":
        mm = numpy.concatenate((moment, target))
        mm = (numpy.exp(mm) / numpy.sum(numpy.exp(mm))).reshape(2, 4)
        return numpy.sum(
            (mm[0] - mm[1]) ** 2
        )
    else:
        raise Exception(f"method {method} was not found")


class KernelDistribution(AbstractDistribution):
    """核分布"""

    def __init__(self, *args):
        super().__init__()
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


def auto_bounds(mu_target, var_target, kurt_target):
    """自动化搜索边界"""
    norm = NormalDistribution(mu_target, var_target ** 0.5)
    mu_domain_min, mu_domain_max = norm.ppf([eps, 1 - eps])
    var_domain_min = var_target / 100
    var_domain_max = var_target * (1 + numpy.clip((kurt_target - 3), 0, 5))
    return mu_domain_min, mu_domain_max, var_domain_min, var_domain_max


def moment_fitted_kde(mu_target, var_target, skew_target, kurt_target, kernel_num=5, method="basic"):
    """矩拟合KDE分布"""

    def f(x):
        args = numpy.asarray(x).reshape(kernel_num, 2)
        kd = KernelDistribution(*args)
        data = kd.ppf(numpy.linspace(0.01, 0.99, 100))
        return moment_loss(data, mu_target, var_target, skew_target, kurt_target, method)

    mu_domain_min, mu_domain_max, var_domain_min, var_domain_max = auto_bounds(mu_target, var_target, kurt_target)

    result = minimize(
        f,
        numpy.asarray([[mu_target, var_target ** 0.5]] * kernel_num).reshape(kernel_num * 2),
        bounds=[(mu_domain_min, mu_domain_max), (var_domain_min, var_domain_max)] * kernel_num,
        # maxiter=20,
        # popsize=5, polish=False
    )
    return KernelDistribution(*numpy.asarray(result.x).reshape(kernel_num, 2))


if __name__ == "__main__":
    dist = moment_fitted_kde(45, 1 ** 2, -2, 1, kernel_num=4, method="minmax")
    print(
        data_moment(dist.rvf(1000))
    )
    print(dist.rvf(100).tolist())
