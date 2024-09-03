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
from typing import Union, Self, Type
from collections import namedtuple

# 项目模块
from data_utils.stochastic_utils.distributions.baseclass import ABCDistribution, UniformDistribution
from data_utils.stochastic_utils.distributions.basic_distributions import NormalDistribution
from data_utils.serial_utils.data_series import NamedSeries
from easy_utils.number_utils import calculus_utils

# 外部模块
import numpy
from scipy.stats import t, iqr
from matplotlib import pyplot

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


def adam_estimated_distribution(
        data: Union[list, tuple, numpy.ndarray],
        dist: Type[ABCDistribution],
        diff: float = 1e-5,
        lr: float = 0.1,
        epoch: int = 200,
        x_len: int = None,
        kernel_len: int = 100
) -> tuple[ABCDistribution, float]:
    """
    参数估计
    """

    kernel_dist = KernelMixDist(data, bin=kernel_len)

    init_param = []
    for item in dist.parameter_range.items():
        u = UniformDistribution(item[1].low, item[1].high)
        r = u.rvf()
        init_param.append(r)

    x = numpy.array(init_param) if x_len is None else numpy.array(init_param[:x_len])

    def moment(x):
        new_dist = dist(*[float(i) for i in x])
        return numpy.array([new_dist.mean(), new_dist.std()])

    def f(x):
        new_dist = dist(*[float(i) for i in x])
        return new_dist.ppf().y

    y = kernel_dist.ppf().y
    target_moment = numpy.array([kernel_dist.mean(), kernel_dist.std()])

    guess = calculus_utils.adam_method(moment, x, target_moment, diff, lr, epoch)
    parameter = calculus_utils.adam_method(f, guess, y, diff, lr, epoch)

    return dist(*[float(i) for i in parameter]), numpy.mean((y - f(parameter)) ** 2)


def MLE_estimated_distribution(
        data: Union[list, tuple, numpy.ndarray],
        dist: Type[ABCDistribution],
        diff: float = 1e-5,
        lr: float = 0.1,
        epoch: int = 200,
        x_len: int = None,
) -> tuple[ABCDistribution, float]:
    def likelihood(xlist: Union[list, tuple, numpy.ndarray]):
        l = 0
        for i, d in enumerate(data):
            like_pdf = dist(*xlist).pdf(d)
            l += numpy.log(like_pdf) if like_pdf > 0 else numpy.log(like_pdf + diff)
        return [-1 * l]

    init_param = []
    for item in dist.parameter_range.items():
        u = UniformDistribution(item[1].low, item[1].high)
        r = u.rvf()
        init_param.append(r)

    x = numpy.array(init_param) if x_len is None else numpy.array(init_param[:x_len])

    parameter = calculus_utils.adam_method(likelihood, x, [0], diff, lr, epoch)
    return dist(*[float(i) for i in parameter]), likelihood(parameter)[0]


if __name__ == "__main__":
    from data_utils.stochastic_utils.distributions.basic_distributions import NormalDistribution, LogNormalDistribution, \
        WeibullDistribution

    wd = WeibullDistribution(2, 5)
    rwd = wd.rvf(1000)
    ed, loss = adam_estimated_distribution(rwd, WeibullDistribution, x_len=3, kernel_len=100)
    mle, mloss = MLE_estimated_distribution(rwd, WeibullDistribution, x_len=3)
    print([ed, mle])
    print([loss, mloss])

    pyplot.plot(ed.ppf().y)
    pyplot.plot(wd.ppf().y)
    pyplot.show()

    # from matplotlib import pyplot
    # import json
    #
    # nd = NormalDistribution(0, 1)
    # r = nd.rvf(1000)
    #
    # print(silverman_bandwidth(r))
    # kd = KernelMixDist(r, bin=50)
    # # print([str(i) for i in kd.kernels])
    # # # print(kd.pdf().y.tolist())
    # # # print(kd.cdf().y.tolist())
    # # # print(kd.n_skewness())
    # # # print(kd.n_kurtosis())
    # #
    # # pyplot.hist(r)
    # # pyplot.show()
    #
    # pyplot.scatter(x=kd.pdf().x, y=kd.pdf().y)
    # pyplot.scatter(x=nd.pdf().x, y=nd.pdf().y)
    # pyplot.show()
    #
    # pyplot.scatter(x=kd.cdf().x, y=kd.cdf().y)
    # pyplot.scatter(x=nd.cdf().x, y=nd.cdf().y)
    # pyplot.show()
