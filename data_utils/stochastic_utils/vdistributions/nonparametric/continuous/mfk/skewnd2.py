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

from data_utils.stochastic_utils.vdistributions.nonparametric.continuous.mfk.skewnd import SkewKDFitter, \
    SkewNormalKernel

# 外部模块
import numpy
from scipy.optimize import differential_evolution, minimize
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d


# 代码块

# class SkewNormalKernel(SkewNormalDistribution):
#
#     def mean(self):
#         return (numpy.sqrt(2 / numpy.pi) * self.alpha * self.sigma) / numpy.sqrt(self.alpha ** 2 + 1) + self.mu
#
#     def variance(self):
#         return (1 - (2 * self.alpha ** 2) / (numpy.pi * (self.alpha ** 2 + 1))) * self.sigma ** 2
#
#     def skewness(self):
#         numerator = numpy.sqrt(2) * (4 - numpy.pi) * self.alpha ** 3
#         denominator = ((numpy.pi - 2) * self.alpha ** 2 + numpy.pi) ** 1.5
#         return numerator / denominator
#
#     def kurtosis(self):
#         numerator = 8 * (numpy.pi - 3) * self.alpha ** 4
#         denominator = ((numpy.pi - 2) * self.alpha ** 2 + numpy.pi) ** 2
#         return numerator / denominator + 3
#
#     def moment(self):
#         return numpy.asarray([self.mean(), self.variance(), self.skewness(), self.kurtosis()])


class SkewWeightKernelDistribution(AbstractDistribution):
    """核分布"""

    def __init__(self, x):
        super().__init__()
        x = numpy.asarray(x).reshape(-1, 4)
        self.kernel_len = len(x)
        self.kernels = [
            SkewNormalKernel(i[0], i[1], i[2]) for i in x
        ]
        self.w = [i[3] for i in x]
        self.ws = numpy.sum(self.w)

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
        m = numpy.stack([k.pdf(x) * self.w[i] for i, k in enumerate(self.kernels)], axis=0)
        r = numpy.sum(m, axis=0) / self.ws
        return r

    def cdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        m = numpy.stack([k.cdf(x) * self.w[i] for i, k in enumerate(self.kernels)], axis=0)
        r = numpy.sum(m, axis=0) / self.ws
        return r

    def ppf(self, x, *args, **kwargs):
        x = numpy.atleast_1d(x)
        result = self.ppf_inter(x)
        return numpy.clip(result, self.domain_min, self.domain_max)

    def mean(self):
        # w = 1 / self.kernel_len
        return numpy.sum([k.mean() * self.w[i] for i, k in enumerate(self.kernels)]) / self.ws

    def variance(self):
        # w = 1 / self.kernel_len
        mu_hat = self.mean()
        return numpy.sum(
            [(k.variance() + (k.mean() - mu_hat) ** 2) * self.w[i] for i, k in enumerate(self.kernels)]) / self.ws

    def skewness(self):
        # w = 1 / self.kernel_len
        mu_hat = self.mean()
        var_hat = self.variance()
        std_hat = var_hat ** 0.5
        mu_3 = numpy.sum([
            (
                    k.skewness() * k.variance() ** 1.5 + 3 * (k.mean() - mu_hat) * k.variance() + (
                    k.mean() - mu_hat) ** 3
            ) * self.w[i] for i, k in enumerate(self.kernels)]) / self.ws
        return mu_3 / std_hat ** 3

    def moment(self) -> numpy.ndarray:
        # w = 1 / self.kernel_len
        mu_hat = self.mean()
        var_hat = self.variance()
        # std_hat = var_hat ** 0.5
        ske_hat = self.skewness()
        mu_4 = numpy.sum([
            (k.kurtosis() * k.variance() ** 2 + 6 * (k.mean() - mu_hat) ** 2 * k.variance() + (
                    k.mean() - mu_hat) ** 4) * self.w[i]
            for i, k in enumerate(self.kernels)]) / self.ws
        return numpy.asarray([mu_hat, var_hat, ske_hat, mu_4 / var_hat ** 2])


class SkewWeightKDFitter:
    """使用偏度KernelDistribution拟合"""

    def __init__(self, x0, mu_target, var_target, skew_target, kurt_target):
        self.x0 = numpy.asarray(x0)
        self.target = numpy.asarray([
            mu_target, var_target, skew_target, kurt_target
        ])

    def adj(self, xw):
        x = self.x0.reshape(-1, 3)
        xw = numpy.asarray(xw).reshape(len(x), -1)
        y = numpy.column_stack((x, xw))
        return y.reshape(len(x) * 4)

    def adj_loss(self, xw):
        y = self.adj(xw)
        # return y.reshape(len(x) * 4)
        d = SkewWeightKernelDistribution(y)
        m = d.moment()
        return numpy.sum((m - self.target) ** 2)

    def fit(self, kernel_num):
        result = differential_evolution(
            self.adj_loss,
            bounds=[(0.1, 1)] * kernel_num
        )
        y = self.adj(result.x)
        return SkewWeightKernelDistribution(y), numpy.asarray(y).reshape(-1, 4)


class SkewWeightKDFitter2:

    def __init__(self, x0, mu_target, var_target, skew_target, kurt_target, kernel_num):
        self.x0 = numpy.asarray(x0)
        self.target = numpy.asarray([
            mu_target, var_target, skew_target, kurt_target
        ])
        self.kernel_num = kernel_num

    def zoom(self, xz):
        return self.x0 * xz

    def zoom_and_adj(self, xz, xw):
        x = self.zoom(xz).reshape(-1, 3)
        xw = numpy.asarray(xw).reshape(len(x), -1)
        y = numpy.column_stack((x, xw))
        return y.reshape(len(x) * 4)

    def zoom_and_adj_loss(self, xzw):
        xz = xzw[:self.kernel_num * 3]
        xw = xzw[self.kernel_num * 3:]
        y = self.zoom_and_adj(xz, xw)
        # return y.reshape(len(x) * 4)
        d = SkewWeightKernelDistribution(y)
        m = d.moment()
        return numpy.sum((m - self.target) ** 2)

    def dof_zoom_and_adj_loss(self, xzw, dof_index):
        xz = xzw[:self.kernel_num * 3]
        xw = xzw[self.kernel_num * 3:]
        y = self.zoom_and_adj(xz, xw)
        d = SkewWeightKernelDistribution(y)
        m = d.moment()[dof_index]
        target = self.target[dof_index]
        return numpy.sum((m - target) ** 2)

    def fit(self, de_maxiter=50):
        result = differential_evolution(
            self.zoom_and_adj_loss,
            bounds=[(0.1, 1.9)] * (self.kernel_num * 3) + [(0.1, 1)] * self.kernel_num,
            maxiter=de_maxiter
        )
        y = self.zoom_and_adj(result.x[:self.kernel_num * 3], result.x[self.kernel_num * 3:])
        return SkewWeightKernelDistribution(y), numpy.asarray(y).reshape(-1, 4)

    def dof_fit(self, dof_index):
        result = minimize(
            self.dof_zoom_and_adj_loss,
            numpy.random.uniform(0.1, 1, self.kernel_num * 4),
            bounds=[(0.1, numpy.inf)] * (self.kernel_num * 3) + [(0.1, 1)] * self.kernel_num,
            # maxiter=de_maxiter,
            args=(dof_index,)
        )
        y = self.zoom_and_adj(result.x[:self.kernel_num * 3], result.x[self.kernel_num * 3:])
        return SkewWeightKernelDistribution(y), numpy.asarray(y)


def snd_fitter(mu_target, var_target, skew_target, kurt_target, kernel_num, de_maxiter=20):
    """snd拟合"""
    finder = SkewKDFitter(mu_target, var_target, skew_target, kurt_target)
    d, p = finder.fit(kernel_num)
    finder2 = SkewWeightKDFitter2(p, mu_target, var_target, skew_target, kurt_target, kernel_num)
    d2, p2 = finder2.fit(de_maxiter)
    return d, d2, p2


def dof_snd_fitter(mu_target, var_target, skew_target, kurt_target, kernel_num, de_maxiter=20):
    """带有自由度的snd拟合"""
    default_moment = numpy.asarray([0, 1, 0, 3])
    input_target = numpy.asarray([mu_target, var_target, skew_target, kurt_target])
    target = numpy.where(input_target != None, input_target, default_moment)
    dof_index = numpy.where(input_target != None)

    finder = SkewKDFitter(*target)
    d, p = finder.fit(kernel_num)

    finder2 = SkewWeightKDFitter2(p, mu_target, var_target, skew_target, kurt_target, kernel_num)
    d2, p2 = finder2.dof_fit(dof_index)
    return d, d2, p2


if __name__ == "__main__":
    # d, d2, p2 = snd_fitter(40, 2 ** 2, -3, 1, 2)
    # print(d.moment())
    # print(d2.moment())
    # print(p2)
    # print(d2.rvf(200).tolist())

    # finder = SkewKDFitter(40, 2 ** 2, -3, 5)
    # # print(finder.adj([0, 1, 2, 3, 4, 5], [-1, -2]))
    # d, p = finder.fit(2)
    # print(d.moment())
    # finder2 = SkewWeightKDFitter2(p, 40, 2 ** 2, -3, 5, 2)
    # d2, p2 = finder2.fit(20)
    # print(finder2.target)
    # print(d2.moment())
    # print(p2)

    d, d2, p2 = dof_snd_fitter(45, 0.1, -0.5, None, 4)
    print(d.moment())
    print(d2.moment())
    print(d2.rvf(100).tolist())
