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
from data_utils.serial_utils.data_series import NamedSeries
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
                 h: Union[float, int] = None,
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
            bin_data = NamedSeries(sorted_data).aggregate(bin_step)
            self.data = bin_data.tuple.y

        if h is None:
            self.h = silverman_bandwidth(data)
        else:
            self.h = h

        self.kernels = [
            GaussianKernel(d, self.h) for i, d in enumerate(self.data)
        ]

    def _pdf(self, x: float) -> float:
        return sum([k._pdf(x) for i, k in enumerate(self.kernels)]) * (1 / (len(self.data) * self.h))

    def _cdf(self, x: float) -> float:
        return sum([k._cdf(x) for i, k in enumerate(self.kernels)]) * (1 / len(self.data))

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
    kd = KernelMixDist(r, bin=50)
    # print([str(i) for i in kd.kernels])
    # # print(kd.pdf().y.tolist())
    # # print(kd.cdf().y.tolist())
    # # print(kd.n_skewness())
    # # print(kd.n_kurtosis())
    #
    # pyplot.hist(r)
    # pyplot.show()

    pyplot.scatter(x=kd.pdf().x, y=kd.pdf().y)
    pyplot.scatter(x=nd.pdf().x, y=nd.pdf().y)
    pyplot.show()

    pyplot.scatter(x=kd.cdf().x, y=kd.cdf().y)
    pyplot.scatter(x=nd.cdf().x, y=nd.cdf().y)
    pyplot.show()
