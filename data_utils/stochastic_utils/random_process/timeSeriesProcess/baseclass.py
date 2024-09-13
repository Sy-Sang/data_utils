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


# 外部模块
import numpy


# 代码块

class TimeSeriesProcess(ABC):
    """时间序列"""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.order = None

    def __str__(self):
        return f"model:{type(self)}, args:{self.args}, kwargs:{self.kwargs}, order:{self.order}"

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def next(self, first: Union[list, tuple, numpy.ndarray], num: int = 1, use_eps: bool = True, *args,
             **kwargs) -> numpy.ndarray:
        """下num个随机数"""
        pass

    def stochastic_component(self, data: Union[list, tuple, numpy.ndarray]) -> numpy.ndarray:
        """获取数据随机项"""
        n = len(data)
        data = numpy.array(data).astype(float)
        noise = []
        for t in range(self.order, n):
            first = data[:t]
            walk = self.next(first=first, num=1, use_eps=False)
            noise.append(data[t] - walk[t])
        return noise

    @classmethod
    @abstractmethod
    def fit(cls, data: Union[list, tuple, numpy.ndarray], p: int = 1, *args, **kwargs) -> Self:
        """拟合时间序列过程"""
        pass


if __name__ == "__main__":
    pass
