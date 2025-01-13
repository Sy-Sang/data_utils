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
from data_utils.serial_utils.data_series import OneDimSeries
from easy_utils.number_utils import calculus_utils
from easy_datetime.temporal_utils import timer

# 外部模块
import numpy
from scipy.stats import t, iqr
from matplotlib import pyplot
from scipy.interpolate import interp1d

# 代码块

ND = NormalDistribution(0, 1)
EPS = numpy.finfo(float).eps


def silverman_bandwidth(data) -> float:
    """
    Silverman 规则
    """
    n = len(data)
    std_dev = numpy.std(data, ddof=1)
    ir = iqr(data)

    h = 0.9 * min([std_dev, ir / 1.34]) * n ** (-0.2)

    return h


def freedam_diaconis(data) -> int:
    """
    Freedman-Diaconis 方法
    """
    if numpy.max(data) == numpy.min(data):
        return 1
    else:
        q25, q75 = numpy.percentile(data, [25, 75])
        n = len(data)
        bin_width = 2 * (q75 - q25) / numpy.cbrt(n)
        bin_count = int((numpy.max(data) - numpy.min(data)) / bin_width)
        return bin_count if bin_count < n / 2 else int(n ** 0.5)


class GaussianKernel(NormalDistribution):
    """
    高斯核
    """

    def __init__(self, mu, sigma):
        super().__init__(mu, sigma)

    def _pdf(self, x: float) -> float:
        return numpy.exp(-((-self.mu + x) ** 2 / (2 * self.sigma ** 2))) / numpy.sqrt(2 * numpy.pi)


class HistogramDist(ABCDistribution):
    """直方图分布"""

    def __init__(
            self,
            data: Union[list, tuple, numpy.ndarray],
            kernel_len: int = None
    ):
        # print(kernel_len)
        if kernel_len is None:
            kernel_len = freedam_diaconis(data)
        else:
            pass

        super().__init__(kernel_len=kernel_len)
        self.data = data
        his = numpy.histogram(data, bins=kernel_len, density=False)
        self.his_x = his[1]
        self.his_y = his[0]
        accumulation_his = [0]
        for i in range(kernel_len):
            p = self.his_y[i] / len(data)
            accumulation_his.append(p + accumulation_his[-1])

        self.his_curve = interp1d(self.his_x, accumulation_his)
        self.his_icurve = interp1d(accumulation_his, self.his_x)

    def mean(self) -> float:
        return numpy.mean(self.data)

    def std(self) -> float:
        return numpy.std(self.data)

    def _cdf(self, x):
        if x < self.his_x[0]:
            return 0
        elif x > self.his_x[-1]:
            return 1
        else:
            return self.his_curve(x)

    def _pdf(self, x):
        eps = 1e-6
        dy = self._cdf(x + eps) - self._cdf(x - eps)
        dx = (x + eps) - (x - eps)
        return dy / dx if dy > 0 else numpy.finfo(float).eps

    def _ppf(self, x):
        if 0 < x < 1:
            return self.his_icurve(x)
        else:
            return numpy.nan


class KernelMixDist(ABCDistribution):
    """
    核混合分布
    """

    def __init__(self, data: Union[list, tuple, numpy.ndarray],
                 h: Union[float, int] = None,
                 kernel_len: int = None):
        """
        data: 待拟合数据
        h: 初始带宽
        hk: 自动化带宽时的k
        bin: 使用bin方法压缩数据时的长度
        """
        super().__init__(h=h, kernel_len=kernel_len)
        sorted_data = sorted(data)
        if kernel_len is None:
            self.data = sorted_data
        else:
            bin_step = len(data) / min(kernel_len, len(data))
            bin_data = OneDimSeries(sorted_data).aggregate(bin_step)
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


class SmoothHisDist(KernelMixDist):
    """平滑直方图分布"""

    def __init__(
            self,
            data: Union[list, tuple, numpy.ndarray],
            kernel_len: int = None
    ):
        his_dist = HistogramDist(data, kernel_len)
        data = his_dist.ppf().y
        super().__init__(data)


class LogHisDist(ABCDistribution):
    def __init__(self, data: Union[list, tuple, numpy.ndarray], kernel_len: int = None):
        array = numpy.array(data)
        diff = 1 - numpy.min(array)
        array += diff
        log_array = numpy.log(array)
        his = HistogramDist(log_array, kernel_len=kernel_len)
        super().__init__(kernel_len=kernel_len, diff=diff)
        self.data = array
        self.log_dist = copy.deepcopy(his)
        self.diff = diff

    def __str__(self):
        return str({
            "name": str(type(self)),
            "args": str(self.log_dist.args),
            "kwargs": str(self.log_dist.kwargs)
        })

    def _ppf(self, x):
        if 0 < x < 1:
            return numpy.e ** self.log_dist._ppf(x) - self.kwargs["diff"]
        else:
            return numpy.nan

    def _cdf(self, x):
        return self.log_dist._cdf(numpy.log(x + self.kwargs["diff"]))

    def _pdf(self, x):
        return self.log_dist._pdf(numpy.log(x + self.kwargs["diff"])) / (x + self.kwargs["diff"])

    def mean(self):
        return numpy.mean(self.data)

    def std(self):
        return numpy.std(self.data, ddof=1)


class LogSmoothHisDist(LogHisDist):
    def __init__(self, data: Union[list, tuple, numpy.ndarray], kernel_len: int = None):
        array = numpy.array(data)
        diff = 1 - numpy.min(array)
        array += diff
        log_array = numpy.log(array)
        his = SmoothHisDist(log_array, kernel_len=kernel_len)
        super().__init__(data, kernel_len)
        self.data = array
        self.log_dist = copy.deepcopy(his)
        self.diff = diff


@timer
def mle_estimated_distribution(
        data: Union[list, tuple, numpy.ndarray],
        dist: Type[ABCDistribution],
        diff: float = 1e-5,
        lr: float = 0.1,
        epoch: int = 200,
        x_len: int = None,
        kernel_len: int = None,
        *args, **kwargs
) -> tuple[ABCDistribution, float]:
    """
    最大似然函数参数估计
    """

    if kernel_len is None:
        pass
    else:
        data = OneDimSeries(data).aggregate(len(data) / min(len(data), kernel_len)).tuple.y

    def likelihood(xlist: Union[list, tuple, numpy.ndarray]):
        """似然函数"""
        l = 0
        for i, d in enumerate(data):
            like_pdf = dist(*xlist).pdf(d)
            l += numpy.log(like_pdf) if like_pdf > 0 else numpy.log(diff)
        return [-1 * l]

    init_param = []
    for item in dist.parameter_range.items():
        init_param.append(float(item[1].default))

    x = numpy.array(init_param) if x_len is None else numpy.array(init_param[:x_len])

    parameter, _ = calculus_utils.adam_method(likelihood, x, [0], diff, lr, epoch)
    return dist(*[float(i) for i in parameter]), likelihood(parameter)[0]


if __name__ == "__main__":
    from data_utils.stochastic_utils.distributions.basic_distributions import NormalDistribution, LogNormalDistribution, \
        WeibullDistribution, StudentTDistribution

    wd = WeibullDistribution(2, 10)
    # wd = NormalDistribution(100, 100)
    rwd = wd.rvf(1000)

    # s = numpy.array([0 if i == 0 else (1 if i > 0 else -1) for i in rwd])
    # eps_rwd = rwd + numpy.finfo(float).eps
    # pyplot.hist(s * numpy.log(abs(eps_rwd)))
    # pyplot.show()
    # pyplot.hist(rwd)
    # pyplot.show()

    logdist = LogHisDist(rwd)
    pyplot.scatter(x=logdist.ppf().x, y=logdist.ppf().y)
    pyplot.scatter(x=logdist.log_dist.ppf().x, y=logdist.log_dist.ppf().y)
    pyplot.scatter(x=wd.ppf().x, y=wd.ppf().y)
    pyplot.legend(["relog", "log", "wd"])
    pyplot.show()

    pyplot.scatter(x=logdist.cdf().x, y=logdist.cdf().y)
    pyplot.scatter(x=wd.cdf().x, y=wd.cdf().y)
    pyplot.show()

    pyplot.scatter(x=logdist.pdf().x, y=logdist.pdf().y)
    pyplot.scatter(x=wd.pdf().x, y=wd.pdf().y)
    pyplot.show()
