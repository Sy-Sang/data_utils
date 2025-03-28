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

# 外部模块
import numpy
from scipy.interpolate import interp1d


# 代码块

def freedman_diaconis(data) -> int:
    """
    Freedman-Diaconis 方法
    """
    data = numpy.asarray(data)
    q25, q75 = numpy.percentile(data, [25, 75])
    n = data.size
    bin_width = 2 * (q75 - q25) / numpy.cbrt(n)
    if bin_width == 0:
        return 1
    else:
        bin_count = int((numpy.max(data) - numpy.min(data)) / bin_width)
        return bin_count if bin_count < n / 2 else int(n ** 0.5)


class HistogramDistribution(AbstractDistribution):
    """直方图分布"""

    def __init__(self, data, bin_count: int = None):
        super().__init__()
        data = numpy.asarray(data)
        self.bin_count = freedman_diaconis(data) if bin_count is None else bin_count
        self.his_y, self.his_x = numpy.histogram(data, bins=self.bin_count, density=False)
        probs = self.his_y / len(data)
        accumulation_his = numpy.concatenate([[0], numpy.cumsum(probs)])
        self.his_curve = interp1d(self.his_x, accumulation_his, bounds_error=False, fill_value=(0, 1))
        self.his_icurve = interp1d(accumulation_his, self.his_x, bounds_error=False, fill_value=(numpy.nan, numpy.nan))

    def __str__(self):
        return str({self.__class__.__name__: self.bin_count})

    def __repr__(self):
        return str({self.__class__.__name__: self.bin_count})

    def ppf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        r = numpy.where(
            (x > 0) & (x < 1),
            self.his_icurve(x),
            numpy.nan
        )
        return r

    def cdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        r = numpy.where(
            x < self.his_x[0],
            0,
            numpy.where(
                x > self.his_x[-1],
                1,
                self.his_curve(x)
            )
        )
        return r

    def pdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        dy = self.cdf(x + eps) - self.cdf(x - eps)
        dx = 2 * eps
        r = numpy.where(
            dy > 0,
            dy / dx,
            eps
        )
        return r


class LogHistogramDistribution(AbstractDistribution):
    """对数直方图分布"""

    def __init__(self, data, bin_count: int = None):
        super().__init__()
        data = numpy.asarray(data)
        self.diff = 1 - numpy.min(data)
        log_data = numpy.log(data + self.diff)
        self.his_dist = HistogramDistribution(log_data, bin_count)

    def __str__(self):
        return str({self.__class__.__name__: self.his_dist.bin_count})

    def __repr__(self):
        return str({self.__class__.__name__: self.his_dist.bin_count})

    def ppf(self, x, *args, **kwargs):
        return numpy.e ** self.his_dist.ppf(x) - self.diff

    def pdf(self, x, *args, **kwargs):
        x = numpy.asarray(x) + self.diff
        return self.his_dist.pdf(numpy.log(x)) / x

    def cdf(self, x, *args, **kwargs):
        x = numpy.asarray(x) + self.diff
        return self.his_dist.cdf(numpy.log(x))


if __name__ == "__main__":
    from data_utils.stochastic_utils.vdistributions.parameter.continuous.lifetime import WeibullDistribution
    from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution
    from matplotlib import pyplot

    data = WeibullDistribution(2, 5).rvf(1000)
    pyplot.scatter(WeibullDistribution(2, 5).curves(1000)[2][:, 0],
                   WeibullDistribution(2, 5).curves(1000)[2][:, 1])
    pyplot.scatter(HistogramDistribution(data).curves(1000)[2][:, 0],
                   HistogramDistribution(data).curves(1000)[2][:, 1])
    pyplot.show()
