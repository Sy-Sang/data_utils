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
from typing import Union, Self, Type, Tuple
from collections import namedtuple

# 项目模块
from data_utils.stochastic_utils.vdistributions.abstract import AbstractDistribution, eps
from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution

# 外部模块
import numpy


# 代码块

def resample_like_standard_normal(data) -> numpy.ndarray:
    """将数据变为标准正态分布"""
    data = numpy.atleast_1d(numpy.asarray(data))
    samples = numpy.sort(NormalDistribution(0, 1).rvf(data.size))
    data_index = numpy.argsort(numpy.argsort(data))
    return samples[data_index]


def resample_like_distribution(data, distribution: AbstractDistribution) -> numpy.ndarray:
    """将数据变为分布"""
    data = numpy.atleast_1d(numpy.asarray(data))
    samples = numpy.sort(distribution.rvf(data.size))
    data_index = numpy.argsort(numpy.argsort(data))
    return samples[data_index]


def generate_correlated_sample(base_sample, target_distribution: AbstractDistribution, pearson: float) -> numpy.ndarray:
    """相关性的随机样本"""
    pearson = numpy.clip(pearson, -1, 1)
    z_scored_base = resample_like_standard_normal(base_sample)
    new_sample = NormalDistribution(0, 1).rvf(z_scored_base.size)
    new_sample = pearson * z_scored_base + (1 - pearson ** 2) ** 0.5 * new_sample
    return resample_like_distribution(new_sample, target_distribution)


def generate_correlated_sample_matrix(base_sample, *args: Tuple[AbstractDistribution, float]) -> numpy.ndarray:
    """相关性的随机样本"""
    return numpy.stack([
        generate_correlated_sample(base_sample, arg[0], arg[1]) for arg in args
    ], axis=0)


if __name__ == "__main__":
    from matplotlib import pyplot
    from data_utils.stochastic_utils.vdistributions.parameter.continuous.uniform import UniformDistribution
    from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import LogNormalDistribution
    from data_utils.stochastic_utils.vdistributions.nonparametric.discrete.basic import DiscreteDistribution

    # dd = DiscreteDistribution(numpy.arange(10).reshape(-1, 2))
    dd = LogNormalDistribution(0, 1)

    d = NormalDistribution(1, 2).rvf(100)
    n = generate_correlated_sample(d, dd, -0.5)
    # pyplot.plot(d)
    # pyplot.plot(n)
    # pyplot.show()

    pyplot.plot(generate_correlated_sample_matrix(
        d,
        (NormalDistribution(0, 1), 0.5),
        (NormalDistribution(0, 2), -0.5),
        (DiscreteDistribution(numpy.arange(10).reshape(-1, 2)), 1),
    ).T)
    pyplot.show()
    print(
        generate_correlated_sample_matrix(
            d,
            (NormalDistribution(0, 1), 0.5),
            (NormalDistribution(0, 2), -0.5),
            (DiscreteDistribution(numpy.arange(10).reshape(-1, 2)), 1),
        )
    )
