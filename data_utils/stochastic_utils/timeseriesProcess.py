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
from data_utils.stochastic_utils.distributions.basic_distributions import NormalDistribution

# 外部模块
import numpy


# 代码块


class TimeSeriesProcess(ABC):
    """时间序列过程"""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def random_process(self, *args, **kwargs) -> numpy.ndarray:
        """
        随机过程
        """
        pass


class ARProcess(TimeSeriesProcess):
    """自回归过程"""

    def __init__(self, phi: float = 0.1, sigma: float = 1.0):
        super().__init__(phi=phi, sigma=sigma)
        self.phi = phi
        self.sigma = sigma

    def random_process(self, first: float = 0, num: int = 100, *args, **kwargs) -> numpy.ndarray:
        """
        随机过程
        """
        p = numpy.array([float(first)] * num)
        dist = NormalDistribution(0, self.sigma)
        noize = dist.rvf(num)
        for i in range(1, num):
            p[i] = p[i - 1] * self.phi + noize[i]
        return p


class MAProcess(TimeSeriesProcess):
    def __init__(self, mu: float = 0, theta: float = 1, sigma: float = 1):
        super().__init__(mu=mu, theta=theta, sigma=sigma)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

    def random_process(self, first: float = 0, num: int = 100, *args, **kwargs) -> numpy.ndarray:
        eps = NormalDistribution(0, self.sigma).rvf(num)
        p = numpy.array([float(first)] * num)
        for i in range(1, num):
            p[i] = self.mu + eps[i] + self.theta * eps[i - 1]
        return p


if __name__ == "__main__":
    ar = ARProcess()
    ma = MAProcess()
    # print(ar.forward(0))
    p = ar.random_process().tolist()
    pp = ma.random_process().tolist()
    # print(p)
    # ip = ar.noize(p)
    # print(ip.tolist())
    print(p)
    print(pp)
