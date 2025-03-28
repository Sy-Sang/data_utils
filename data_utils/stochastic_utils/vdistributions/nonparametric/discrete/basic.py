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


# 代码块

def standardization(x) -> numpy.ndarray:
    """直方图数据标准化"""
    x = numpy.asarray(x).astype(float)
    p = 1 / numpy.sum(x[:, 1])
    x[:, 1] = x[:, 1] * p
    return x[numpy.argsort(x[:, 0])]


class DiscreteDistribution(AbstractDistribution):
    """基本离散分布"""

    def __init__(self, sample_space):
        super().__init__()
        self.sample_space = standardization(sample_space)

    def __str__(self):
        return str({self.__class__.__name__: self.sample_space})

    def __repr__(self):
        return str({self.__class__.__name__: self.sample_space})

    def pdf(self, x, *args, **kwargs):
        vx = numpy.atleast_1d(numpy.asarray(x).astype(float))
        lookup = dict(zip(self.sample_space[:, 0], self.sample_space[:, 1]))
        r = numpy.array([lookup.get(xi, 0.0) for xi in vx])
        return r[0] if numpy.isscalar(x) else r

    def cdf(self, x, *args, **kwargs):
        vx = numpy.atleast_1d(numpy.asarray(x).astype(float))
        grid = self.sample_space[:, 0]
        prob = self.sample_space[:, 1]
        cdf_vals = numpy.array([prob[grid <= xi].sum() for xi in vx])
        return cdf_vals[0] if numpy.isscalar(x) else cdf_vals

    def ppf(self, q, *args, **kwargs):
        vq = numpy.atleast_1d(numpy.asarray(q).astype(float))
        if numpy.any((vq < 0) | (vq > 1)):
            return numpy.nan

        grid = self.sample_space[:, 0]
        cdf_vals = numpy.cumsum(self.sample_space[:, 1])
        indices = numpy.searchsorted(cdf_vals, vq, side='left')
        result = grid[numpy.clip(indices, 0, len(grid) - 1)]
        return result[0] if numpy.isscalar(q) else result


if __name__ == "__main__":
    dd = DiscreteDistribution(numpy.arange(10).reshape(-1, 2))
    print(dd)

    print(dd.cdf(numpy.arange(-10, 20, 1)))
    print(dd.pdf(2.0))
    print(dd.ppf(numpy.arange(0.1, 0.9, 0.1)))
