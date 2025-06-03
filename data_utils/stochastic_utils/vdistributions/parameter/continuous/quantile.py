#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""根据分位数函数定义的分布"""

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
from data_utils.stochastic_utils.vdistributions.abstract import eps as float_eps
from data_utils.stochastic_utils.vdistributions.parameter.abstract import ParameterDistribution, DistributionParams

# 外部模块
import numpy
from scipy.interpolate import interp1d


# 代码块

class CDFDefinedDistribution(ParameterDistribution):
    """由CDF函数定义的分布"""

    def __init__(self, x, y):
        super().__init__(x, y, x=x, y=y)
        self.x = numpy.atleast_1d(x)
        self.y = numpy.atleast_1d(y)
        # self.y = numpy.sort(numpy.clip(numpy.atleast_1d(y), 0, 1))
        self.y = numpy.sort((self.y - numpy.min(self.y)) / (numpy.max(self.y) - numpy.min(self.y)))

        self.cdf_interp = interp1d(self.x, self.y, bounds_error=False, fill_value=0.0)
        self.ppf_interp = interp1d(self.y, self.x, bounds_error=False, fill_value=0.0)

        x_grid = numpy.linspace(self.x[0], self.x[-1], 1000)
        pdf_values = numpy.gradient(self.cdf_interp(x_grid), x_grid)
        self.pdf_interp = interp1d(x_grid, pdf_values, bounds_error=False, fill_value=0.0)

    def cdf(self, x):
        x = numpy.asarray(x)
        return self.cdf_interp(x)

    def pdf(self, x):
        x = numpy.asarray(x)
        return self.pdf_interp(x)

    def ppf(self, x):
        x = numpy.asarray(x)
        return self.ppf_interp(x)

    def get_param_constraints(self, args):
        """获取参数范围"""
        xl = []
        yl = []
        for i in range(len(self.x)):
            xl.append(DistributionParams(
                f"x_{i}", -numpy.inf, numpy.inf
            ))
            if i == 0:
                yl.append(DistributionParams(
                    f"y_{i}", 0, 1
                ))
            else:
                yl.append(DistributionParams(
                    f"y_{i}", yl[-1].min + float_eps, 1
                ))
        return xl + yl


if __name__ == "__main__":
    from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution, \
        LogNormalDistribution
    from matplotlib import pyplot

    noice = numpy.sort(NormalDistribution(0, 1).rvf(100))
    # numpy.random.shuffle(noice)

    nd = NormalDistribution(1, 1)
    ppf, pdf, cdf = nd.curves(100)
    pyplot.plot(pdf[:, 0], pdf[:, 1])
    cdd = CDFDefinedDistribution(x=cdf[:, 0], y=cdf[:, 1] + noice)
    ppf, pdf, cdf = cdd.curves(100)
    pyplot.plot(pdf[:, 0], pdf[:, 1])
    pyplot.show()
