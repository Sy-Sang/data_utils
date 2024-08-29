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
from data_utils.stochastic_utils.dist_utils import ABCDistribution
from data_utils.stochastic_utils.basic_distributions import NormalDistribution
from data_utils.serial_utils.data_series import DataSeries
from easy_utils.number_utils import calculus_utils

# 外部模块
import numpy
from scipy.stats import t, iqr

# 代码块

ND = NormalDistribution(0, 1)


def silverman_bandwidth(data) -> float:
    """
    Silverman 规则
    """
    n = len(data)
    std_dev = numpy.std(data, ddof=1)
    ir = iqr(data)

    h = 0.9 * min([std_dev, ir / 1.34]) * n ** (-0.2)

    return h


def adaptive_bandwidth(data, k: int = 5) -> list:
    """
    自适应带宽
    """
    h0 = silverman_bandwidth(data)
    blist = []
    max_index = len(data) - 1
    bk = (k if k % 2 == 0 else k + 1) // 2
    for i, d in enumerate(data):
        if i < bk:
            if i >= max_index - bk:
                distance = max(abs(d - data[0]), abs(d - data[-1]))
            else:
                distance = abs(d - data[i + bk])
        else:
            if i >= max_index - bk:
                distance = abs(d - data[i - bk])
            else:
                distance = min(abs(d - data[i + bk]), abs(d - data[i - bk]))
        h1 = h0 * distance
        blist.append(h1)
    return blist


class GaussianKernel(NormalDistribution):
    """
    高斯核
    """

    def __init__(self, mu, sigma):
        super().__init__(mu, sigma)

    def _pdf(self, x: float) -> float:
        return numpy.exp(-((-self.mu + x) ** 2 / (2 * self.sigma ** 2))) / numpy.sqrt(2 * numpy.pi)


class KernelMixDist(ABCDistribution):
    """
    核混合分布
    """

    def __init__(self, data: Union[list, tuple, numpy.ndarray],
                 h: Union[float, int, list, tuple, numpy.ndarray] = None, hk: int = 5,
                 bin: int = None):
        """
        data: 待拟合数据
        h: 初始带宽
        hk: 自动化带宽时的k
        bin: 使用bin方法压缩数据时的长度
        """
        super().__init__()
        sorted_data = sorted(data)
        if bin is None:
            self.data = sorted_data
        else:
            bin_step = len(data) / min(bin, len(data))
            bin_data = DataSeries(list(range(len(sorted_data))), sorted_data).aggregate(bin_step)
            self.data = bin_data.data["column_1"]
        if isinstance(h, (float, int)):
            self.h = [h] * len(self.data)
        elif h is None:
            self.h = adaptive_bandwidth(data, hk)
            # self.h = [silverman_bandwidth(data)] * len(self.data)
        else:
            self.h = h

        self.kernels = [
            GaussianKernel(d, self.h[i]) for i, d in enumerate(self.data)
        ]

    def _pdf(self, x: float) -> float:
        return sum(
            k._pdf((x - self.data[i]) / self.h[i]) / self.h[i] for i, k in enumerate(self.kernels)
        ) / len(self.data)

    def _cdf(self, x: float) -> float:
        # snd = NormalDistribution(0, 1)
        return sum(
            k._cdf((x - self.data[i]) / self.h[i]) for i, k in enumerate(self.kernels)
        ) / len(self.data)

    def _ppf(self, x: float) -> float:
        if 0 < x < 1:
            def f(y):
                return self.cdf(y) - x

            x0 = numpy.mean(self.data)
            p, error = calculus_utils.newton_method(f, x0)
            return p
        else:
            return numpy.nan

    def mean(self) -> float:
        return numpy.mean(self.data)

    def std(self) -> float:
        return numpy.std(self.data)


if __name__ == "__main__":
    from matplotlib import pyplot
    import json

    nd = NormalDistribution(0, 1)
    r = nd.rvf(1000)

    print(silverman_bandwidth(r))
    print(adaptive_bandwidth(r))
    kd = KernelMixDist(r, bin=100)
    print([str(i) for i in kd.kernels])
    # # print(kd.pdf().y.tolist())
    # # print(kd.cdf().y.tolist())
    # # print(kd.n_skewness())
    # # print(kd.n_kurtosis())
    #
    pyplot.hist(r)
    pyplot.show()

    pyplot.plot(kd.pdf().y)
    pyplot.show()

    pyplot.plot(kd.cdf().y)
    pyplot.show()
