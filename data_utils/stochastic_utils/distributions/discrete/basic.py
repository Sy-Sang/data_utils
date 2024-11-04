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

# 外部模块
import numpy


# 代码块

def standardization(xlist: list[list[float]]) -> numpy.ndarray:
    """直方图数据标准化"""
    a = numpy.array(xlist).astype(float)
    p = 1 / numpy.sum(a[:, 1])
    a[:, 1] = a[:, 1] * p
    return a[numpy.argsort(a[:, 0])]


class DiscreteDistribution(ABCDistribution):
    """基础离散分布"""

    def __init__(self, hist_list):
        super().__init__()
        self.pdf_array = standardization(hist_list)
        self.pdf_map = {float(a[0]): float(a[1]) for a in self.pdf_array}
        cdf_list = []
        for i, p in enumerate(self.pdf_array):
            if i == 0:
                cdf_list.append(p)
            else:
                cdf_list.append([
                    p[0], p[1] + cdf_list[-1][1]
                ])
        self.cdf_array = numpy.array(cdf_list).astype(float)
        self.cdf_map = {float(a[0]): float(a[1]) for a in self.cdf_array}

    def __str__(self):
        return str(self.pdf_array.tolist())

    def _pdf(self, x, *args, **kwargs) -> float:
        if x in self.pdf_map.keys():
            return self.pdf_map[x]
        else:
            return 0

    def _cdf(self, x, *args, **kwargs) -> float:
        if x in self.cdf_map.keys():
            return self.cdf_map[x]
        else:
            return 0

    def _ppf(self, x, *args, **kwargs) -> float:
        if 0 < x < 1:
            for i, cdf in enumerate(self.cdf_array):
                if i != len(self.cdf_array) - 1:
                    if x < cdf[1]:
                        return cdf[0]
                    else:
                        pass
                else:
                    return cdf[0]
        else:
            return numpy.nan

    def mean(self) -> float:
        return numpy.sum(self.pdf_array[:, 0] * self.pdf_array[:, 1])

    def std(self) -> float:
        return numpy.sum((self.pdf_array[:0] - self.mean()) ** 2 * self.pdf_array[:, 1]) ** 0.5


if __name__ == "__main__":
    pass
