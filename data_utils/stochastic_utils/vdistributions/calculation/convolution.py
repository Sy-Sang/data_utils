#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""概率分布的pdf卷积运算"""

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
from data_utils.stochastic_utils.vdistributions.abstract import AbstractDistribution, eps as float_eps

# 外部模块
import numpy
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.integrate import quad, cumulative_trapezoid as cumtrapz


# 代码块

class ConvolutedDistribution(AbstractDistribution):
    def __init__(self, dist_0: AbstractDistribution, dist_1: AbstractDistribution):
        self.dist_0 = dist_0.clone()
        self.dist_1 = dist_1.clone()
        self._cached_grid = None
        self._cached_cdf_interp = None
        self._cached_ppf_interp = None

    def __str__(self):
        return f"{self.dist_0} * {self.dist_1}"

    def __repr__(self):
        return self.__str__()

    def pdf_convolute_at_x(self, x_val):
        def integrand(g):
            return self.dist_0.pdf(x_val - g) * self.dist_1.pdf(g)

        a, b = self.dist_1.domain()
        result, _ = quad(integrand, a, b)
        return result

    def pdf(self, x):
        x = numpy.asarray(x)
        return numpy.array([self.pdf_convolute_at_x(xi) for xi in x])

    def _build_cdf_interp(self, grid_points=500):
        # 构造一个在联合 domain 上的插值表
        a0, b0 = self.dist_0.domain()
        a1, b1 = self.dist_1.domain()
        xmin, xmax = a0 + a1, b0 + b1
        x_grid = numpy.linspace(xmin, xmax, grid_points)
        pdf_values = self.pdf(x_grid)
        cdf_values = cumtrapz(pdf_values, x_grid, initial=0)
        cdf_values /= cdf_values[-1]

        self._cached_grid = x_grid
        self._cached_cdf_interp = interp1d(x_grid, cdf_values, bounds_error=False, fill_value=(0, 1))
        self._cached_ppf_interp = interp1d(cdf_values, x_grid, bounds_error=False, fill_value=(xmin, xmax))

    def cdf(self, x: Union[float, numpy.ndarray]):
        if self._cached_cdf_interp is None:
            self._build_cdf_interp()
        return self._cached_cdf_interp(numpy.asarray(x))

    def ppf(self, q: Union[float, numpy.ndarray]):
        if self._cached_ppf_interp is None:
            self._build_cdf_interp()
        return self._cached_ppf_interp(numpy.asarray(q))


if __name__ == "__main__":
    from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution
    from matplotlib import pyplot

    n0 = NormalDistribution(1, 0.5)
    n1 = NormalDistribution(-1, 1)
    d = ConvolutedDistribution(n0, n1)
    ppf, pdf, cdf = n0.curves()
    pyplot.plot(pdf[:, 0], pdf[:, 1])

    ppf, pdf, cdf = n1.curves()
    pyplot.plot(pdf[:, 0], pdf[:, 1])

    ppf, pdf, cdf = d.curves()
    pyplot.plot(pdf[:, 0], pdf[:, 1])
    pyplot.show()

    print(d.rvf(10))
