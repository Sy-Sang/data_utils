#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""混合核分布估计"""

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

# 系统模块
import copy
import pickle
import json
from typing import Union, Self, Type
from collections import namedtuple

# 项目模块
from easy_utils.number_utils.calculus_utils import newton_method
from data_utils.stochastic_utils.vdistributions.abstract import AbstractDistribution, eps as float_eps
from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution
from data_utils.stochastic_utils.vdistributions.nonparametric.continuous.histogram import HistogramDistribution, \
    freedman_diaconis
from data_utils.stochastic_utils.vdistributions.parameter.continuous.kernel.gaussian2 import \
    WeightedGaussianKernelMixDistribution

# 外部模块
from scipy.stats import t, iqr
from scipy.interpolate import interp1d
import numpy


# 代码块
def find_closest_divisor(n: int, k: int):
    """最临近整除因子"""
    divisors = set()
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return min(divisors, key=lambda x: abs(x - k))


def silverman_bandwidth(data) -> float:
    """
    Silverman 规则
    """
    data = numpy.asarray(data)
    n = data.size
    std_dev = numpy.std(data, ddof=1)
    ir = iqr(data)

    h = 0.9 * numpy.min([std_dev, ir / 1.34]) * n ** (-0.2)
    if h == 0:
        raise Exception(f"h=0, data:{data.tolist()}")

    return h


def gaussia_kernel_mix_estimate(data, kernel_num: int = None):
    """高斯核混合分布估计"""
    x = numpy.sort(numpy.asarray(data))
    kernel_len = freedman_diaconis(x) if kernel_num is None else kernel_num
    if kernel_len < x.size:
        kernel_len = find_closest_divisor(x.size, kernel_len)
        matrix = x.reshape(kernel_len, -1)
        m = numpy.mean(matrix, axis=1)
    else:
        m = x
    h = silverman_bandwidth(m)
    kernels = [(mi, h, 1) for i, mi in enumerate(m)]
    return WeightedGaussianKernelMixDistribution(*kernels)


if __name__ == "__main__":
    from data_utils.stochastic_utils.vdistributions.parameter.continuous.lifetime import WeibullDistribution
    from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import LogNormalDistribution
    from matplotlib import pyplot

    data = WeibullDistribution(2, 5).rvf(500)
    dist = gaussia_kernel_mix_estimate(data, None)

    pyplot.plot(WeibullDistribution(2, 5).curves(1000)[1][:, 0],
                WeibullDistribution(2, 5).curves(1000)[1][:, 1])
    pyplot.plot(dist.curves(1000)[1][:, 0],
                dist.curves(1000)[1][:, 1])

    pyplot.show()
