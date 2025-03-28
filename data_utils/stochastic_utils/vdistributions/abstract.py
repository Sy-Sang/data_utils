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

    def curves(self, num: int = 100):
        """ppf, pdf, cdf曲线"""
        x = numpy.linspace(eps, 1 - eps, num)
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


if __name__ == "__main__":
    pass
