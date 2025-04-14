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
from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import SkewNormalDistribution
from data_utils.stochastic_utils.vdistributions.nonparametric.continuous.mfk.nd import data_moment, moment_loss

# 外部模块
import numpy
from scipy.optimize import differential_evolution, minimize
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d


# 代码块

class SkewNormalKernel(SkewNormalDistribution):

    def mean(self):
        return (numpy.sqrt(2 / numpy.pi) * self.alpha * self.sigma) / numpy.sqrt(self.alpha ** 2 + 1) + self.mu

    def variance(self):
        return (1 - (2 * self.alpha ** 2) / (numpy.pi * (self.alpha ** 2 + 1))) * self.sigma ** 2

    def skewness(self):
        numerator = numpy.sqrt(2) * (4 - numpy.pi) * self.alpha ** 3
        denominator = ((numpy.pi - 2) * self.alpha ** 2 + numpy.pi) ** 1.5
        return numerator / denominator

    def kurtosis(self):
        numerator = 8 * (numpy.pi - 3) * self.alpha ** 4
        denominator = ((numpy.pi - 2) * self.alpha ** 2 + numpy.pi) ** 2
        return numerator / denominator + 3

    def moment(self):
        return numpy.asarray([self.mean(), self.variance(), self.skewness(), self.kurtosis()])


class SkewKernelDistribution(AbstractDistribution):
    """核分布"""

    def __init__(self, x):
        super().__init__()
        x = numpy.asarray(x).reshape(-1, 3)
        self.kernel_len = len(x)
        self.kernels = [
            SkewNormalKernel(i[0], i[1], i[2]) for i in x
        ]

        domains = numpy.asarray([i.domain() for i in self.kernels])
        self.domain_min = numpy.min(domains[:, 0])
        self.domain_max = numpy.max(domains[:, 1])

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

    def mean(self):
        w = 1 / self.kernel_len
        return numpy.sum([i.mean() * w for i in self.kernels])
        # return numpy.mean(self.ppf(numpy.linspace(eps, 1-eps, 1000)))

    def variance(self):
        w = 1 / self.kernel_len
        mu_hat = self.mean()
        return numpy.sum([(i.variance() + (i.mean() - mu_hat) ** 2) * w for i in self.kernels])
        # return numpy.std(self.ppf(numpy.linspace(eps, 1-eps, 1000)), ddof=1) ** 2

    def skewness(self):
        w = 1 / self.kernel_len
        mu_hat = self.mean()
        var_hat = self.variance()
        std_hat = var_hat ** 0.5
        mu_3 = numpy.sum([
            (
                    i.skewness() * i.variance() ** 1.5 + 3 * (i.mean() - mu_hat) * i.variance() + (
                    i.mean() - mu_hat) ** 3
            ) * w for i in self.kernels])
        return mu_3 / std_hat ** 3
        # return skew(self.ppf(numpy.linspace(eps, 1-eps, 1000)))

    def moment(self):
        w = 1 / self.kernel_len
        mu_hat = self.mean()
        var_hat = self.variance()
        # std_hat = var_hat ** 0.5
        ske_hat = self.skewness()
        mu_4 = numpy.sum([
            (i.kurtosis() * i.variance() ** 2 + 6 * (i.mean() - mu_hat) ** 2 * i.variance() + (
                    i.mean() - mu_hat) ** 4) * w
            for i in self.kernels])
        return numpy.asarray([mu_hat, var_hat, ske_hat, mu_4 / var_hat ** 2])
        # return kurtosis(self.ppf(numpy.linspace(eps, 1-eps, 1000)))

    # def moment(self):
    #     return numpy.asarray([
    #         self.mean(),
    #         self.variance(),
    #         self.skewness(),
    #         self.kurtosis()
    #     ])


class SkewKDFitter:
    """使用偏度KernelDistribution拟合"""

    def __init__(self, mu_target, var_target, skew_target, kurt_target):
        self.target = numpy.asarray([
            mu_target, var_target, skew_target, kurt_target
        ])

    def loss(self, x):
        d = SkewKernelDistribution(x)
        m = d.moment()
        return numpy.sum((m - self.target) ** 2)

    def fit(self, kernel_num=6):
        result = minimize(
            self.loss,
            # numpy.asarray([[0, 1, 0]] * kernel_num).reshape(kernel_num * 3),
            # numpy.arange(kernel_num * 3),
            numpy.random.uniform(1e-8, 1, kernel_num * 3),
            bounds=[(-numpy.inf, numpy.inf), (self.target[1] / kernel_num, numpy.inf),
                    (-numpy.inf, numpy.inf)] * kernel_num,
            # method='BFGS'
        )
        return SkewKernelDistribution(result.x)


if __name__ == "__main__":
    # snd = SkewNormalKernel(45, 5, 500)
    # print(snd.mean())

    finder = SkewKDFitter(45, 2 ** 2, 2, 1)
    print(finder.target)
    # d = SkewKernelDistribution([0, 1, 0])
    d = finder.fit(4)
    print(d.moment())
    print(d)
    print(d.kernels[0].mean())
    #
    # d = SkewKernelDistribution([45, 1, 0] * 4 )
    # # print(d.domain_min)
    # # print(d.domain_max)
    # print(d.moment())
    # print(d.rvf(200).tolist())

    print(numpy.sum((d.moment() - finder.target) ** 2))
    print(d.rvf(200).tolist())
