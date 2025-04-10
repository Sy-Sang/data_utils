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
from abc import ABC, abstractmethod

import numpy
# 项目模块

# 外部模块
import numpy as np
from scipy.integrate import quad

# 代码块
eps = np.finfo(float).eps


class AbstractDistribution(ABC):
    """概率分布(抽象类)"""

    def __init__(self, *args, **kwargs):
        pass

    def clone(self) -> Self:
        """自身的深度复制"""
        return copy.deepcopy(self)

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def ppf(self, *args, **kwargs) -> Union[float, numpy.ndarray]:
        """分位数函数"""
        pass

    @abstractmethod
    def pdf(self, *args, **kwargs) -> Union[float, numpy.ndarray]:
        """概率密度函数"""
        pass

    @abstractmethod
    def cdf(self, *args, **kwargs) -> Union[float, numpy.ndarray]:
        """累计概率函数"""
        pass

    def domain(self):
        """定义域"""
        nt = namedtuple("nt", ["min", "max"])
        p = self.ppf([eps, 1 - eps])
        return nt(p[0], p[1])

    def curves(self, num: int = 100, tail_eps: float = eps):
        """ppf, pdf, cdf曲线"""
        x = numpy.linspace(tail_eps, 1 - tail_eps, num)
        ppf_curve = numpy.column_stack((
            x, self.ppf(x)
        ))
        pdf_curve = numpy.column_stack((
            ppf_curve[:, 1],
            self.pdf(ppf_curve[:, 1])
        ))
        cdf_curve = numpy.column_stack((
            ppf_curve[:, 1],
            self.cdf(ppf_curve[:, 1])
        ))
        return ppf_curve, pdf_curve, cdf_curve

    def rvf(self, num: int = 1):
        """随机函数"""
        seed = np.random.uniform(0 + eps, 1, size=num)
        return self.ppf(seed)

    def rvf_scalar(self):
        """生成一个随机数标量"""
        return self.rvf(1)[0]

    def mean_integral(self) -> float:
        """均值"""
        a, b = self.domain()
        return quad(lambda x: x * self.pdf(x), a, b)[0]

    def variance_integral(self) -> float:
        """方差"""
        a, b = self.domain()
        mu = self.mean_integral()
        return quad(lambda x: (x - mu) ** 2 * self.pdf(x), a, b)[0]

    def skewness_integral(self) -> float:
        """偏度"""
        a, b = self.domain()
        mu = self.mean_integral()
        sigma = numpy.sqrt(self.variance_integral())
        return quad(lambda x: ((x - mu) / sigma) ** 3 * self.pdf(x), a, b)[0]

    def kurtosis_integral(self) -> float:
        """峰度"""
        a, b = self.domain()
        mu = self.mean_integral()
        sigma = numpy.sqrt(self.variance_integral())
        return quad(lambda x: ((x - mu) / sigma) ** 4 * self.pdf(x), a, b)[0]

    def moment_integral(self) -> numpy.ndarray:
        """矩"""
        return numpy.asarray(
            [self.mean_integral(), self.variance_integral(), self.skewness_integral(), self.kurtosis_integral()])


if __name__ == "__main__":
    pass
