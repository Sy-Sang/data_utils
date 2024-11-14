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
from data_utils.stochastic_utils.distributions.deform import convert_to_dist
from easy_utils.number_utils.float_utils import Eps

# 外部模块
import numpy


# 代码块

def correlated_series(data: Union[list, tuple, numpy.ndarray], dist: ABCDistribution, pearson: float):
    """生成满足相关性的序列"""
    pearson = max(min(pearson, 1), -1)
    base = numpy.array(data).astype(float)
    z_scored_base = convert_to_dist(base, NormalDistribution(0, 1))
    rv = NormalDistribution(0, 1).rvf(len(data))
    x = pearson * z_scored_base + (1 - pearson ** 2) ** 0.5 * rv
    return convert_to_dist(x, dist)


def random_correlated_series(dist: list[ABCDistribution], pearson: list[float], num: int = 100,
                             data: Union[list, tuple, numpy.ndarray] = None) -> numpy.ndarray:
    """生成满足相关性的随机数表"""
    x = NormalDistribution(0, 1).rvf(num) if data is None else numpy.array(data).astype(float)
    r = numpy.array([])
    for i, d in enumerate(dist):
        y = correlated_series(x, d, pearson[i])
        if i == 0:
            r = numpy.concatenate((r, y))
        else:
            r = numpy.column_stack((r, y))
    return r.T


if __name__ == "__main__":
    from data_utils.stochastic_utils.distributions.basic_distributions import NormalDistribution, WeibullDistribution, \
        LogNormalDistribution

    from data_utils.stochastic_utils.random_process.timeSeriesProcess.ar import ARProcess

    d = random_correlated_series([
        WeibullDistribution(2, 5),
        WeibullDistribution(2, 5),
        WeibullDistribution(2, 5)
    ], [-0.5, -0.5, -0.5],
        data=ARProcess(0, [0.9, 0.1], 1).next(num=100)
    )
    print(d.tolist())
