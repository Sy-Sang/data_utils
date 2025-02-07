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
from data_utils.stochastic_utils.distributions.baseclass import ABCDistribution
from data_utils.stochastic_utils.distributions.basic_distributions import NormalDistribution
from data_utils.serial_utils.series_trans_utils import ZScore

# 外部模块
import numpy
from sklearn.preprocessing import PowerTransformer


# 代码块

def to_standard_normal_distribution_old(
        data: Union[list, tuple, numpy.ndarray],
) -> numpy.ndarray:
    """
    将数据变为标准正态分布(会产生关联相关性的GBM路径在末尾重合的奇怪问题)
    """
    array = numpy.array(data).astype(float)
    sort_index = numpy.argsort(array)
    nd = NormalDistribution(numpy.mean(array), numpy.std(array, ddof=1))
    q = numpy.sort(nd.rvf(len(array)))
    nd_list = [0] * len(array)
    for i in range(len(array)):
        nd_list[sort_index[i]] = q[i]

    nd_array = numpy.array(nd_list)
    mu = numpy.mean(nd_array)
    std = numpy.std(nd_array, ddof=1)

    return (nd_array - mu) / std


def to_standard_normal_distribution(
        data: Union[list, tuple, numpy.ndarray],
) -> numpy.ndarray:
    """
    将数据变为标准正态分布
    """
    r = numpy.sort(NormalDistribution(0, 1).rvf(len(data)))
    data_array = numpy.array(data)
    data_index = numpy.argsort(data_array)
    nd_list = [0] * len(data)
    for i, di in enumerate(data_index):
        nd_list[di] = r[i]

    return numpy.array(nd_list).astype(float)


def uniformed(
        data: Union[list, tuple, numpy.ndarray],
) -> namedtuple("typename", ["value", "cdf"]):
    """
    将数据变为取值范围在0-1的均匀分布
    """
    snd = to_standard_normal_distribution(data)
    nd = NormalDistribution(0, 1)

    return nd.cdf(snd).y


def convert_to_dist(data: Union[list, tuple, numpy.ndarray], dist: ABCDistribution) -> numpy.ndarray:
    """
    将数据变为特定分布
    """
    return dist.ppf(uniformed(data)).y


if __name__ == "__main__":
    from data_utils.stochastic_utils.distributions.basic_distributions import WeibullDistribution, LogNormalDistribution
    from matplotlib import pyplot

    w = WeibullDistribution(1, 20)
    rw = w.rvf(100)
    rw2 = w.rvf(100)

    pyplot.plot(
        numpy.sort(to_standard_normal_distribution(rw))
    )
    pyplot.plot(
        numpy.sort(to_standard_normal_distribution(rw2))
    )
    pyplot.show()

    pyplot.plot(
        numpy.sort(to_standard_normal_distribution_old(rw))
    )
    pyplot.plot(
        numpy.sort(to_standard_normal_distribution_old(rw2))
    )
    pyplot.show()

    # print(uniformed(w.rvf(100)).tolist())
    # rn = to_standard_normal_distribution(rw)
    # pyplot.plot(rw)
    # pyplot.plot(rn)
    # pyplot.show()
    # print(rn.tolist())
    # rl = convert_to_dist(rw, LogNormalDistribution(0, 1))
    # pyplot.plot(rw)
    # pyplot.plot(rl)
    # pyplot.show()
    # print(rl.tolist())
