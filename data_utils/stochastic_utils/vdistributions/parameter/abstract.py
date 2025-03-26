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

# 项目模块
from data_utils.stochastic_utils.vdistributions.abstract import AbstractDistribution

# 外部模块
import numpy


# 代码块

class DistributionParams:
    """概率分辨参数"""

    def __init__(self, name: str, min: float, max: float):
        self.name = name
        self.min = min
        self.max = max


class ParameterDistribution(AbstractDistribution):
    """参数分布"""

    def __init__(self, *args, **kwargs):
        self.args = numpy.asarray(args)
        self.kwargs = copy.deepcopy(kwargs)

    def __str__(self):
        return str({self.__class__.__name__: self.kwargs})

    def __repr__(self):
        return str({self.__class__.__name__: self.kwargs})

    @abstractmethod
    def ppf(self, *args, **kwargs):
        """分位数函数"""
        pass

    @abstractmethod
    def pdf(self, *args, **kwargs):
        """概率密度函数"""
        pass

    @abstractmethod
    def cdf(self, *args, **kwargs):
        """累计概率函数"""
        pass

    @abstractmethod
    def get_param_constraints(self) -> list[DistributionParams]:
        """获取参数范围"""
        pass

    def parameter_verification(self, *args) -> numpy.ndarray[bool]:
        """参数验证"""
        param_list = self.get_param_constraints()
        verification_list = []
        for i, arg in enumerate(args):
            if param_list[i].min <= arg <= param_list[i].max:
                verification_list.append(True)
            else:
                verification_list.append(False)
        return numpy.asarray(verification_list)


if __name__ == "__main__":
    pass
