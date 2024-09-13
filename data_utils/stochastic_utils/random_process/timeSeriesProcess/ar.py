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
from data_utils.stochastic_utils.distributions.baseclass import ABCDistribution
from data_utils.stochastic_utils.random_process.timeSeriesProcess.baseclass import TimeSeriesProcess
from data_utils.stochastic_utils.distributions.basic_distributions import NormalDistribution
from data_utils.solve_utils.equationNSolve import gradient_descent, bfgs

# 外部模块
import numpy


# 代码块

class ARProcess(TimeSeriesProcess):
    """自回归过程"""

    def __init__(self, mu: float = 0, phi: Union[list, tuple, numpy.ndarray] = [0.1], sigma: float = 1):
        super().__init__(mu=mu, phi=phi, sigma=sigma)
        self.mu = mu
        self.phi = numpy.array(phi).astype(float)
        self.sigma = sigma
        self.order = len(self.phi)

    def next(self, first: Union[list, tuple, numpy.ndarray], num: int = 1, use_eps: bool = True, *args,
             **kwargs):
        first = numpy.array(first).astype(float)
        eps = NormalDistribution(0, self.sigma).rvf(num) if use_eps is True else numpy.zeros(num)
        y = numpy.concatenate((first, eps))
        start = max(self.order, len(first))
        for t in range(start, num):
            y[t] = self.mu + sum(self.phi[i] * y[t - 1 - i] for i in range(self.order)) + eps[t]
        return y

    @classmethod
    def design_matrix(self, data: Union[list, tuple, numpy.ndarray], p: int = 1) -> tuple[numpy.ndarray, numpy.ndarray]:
        """设计矩阵"""
        n = len(data)
        data = numpy.array(data).astype(float)
        dm = []
        for t in range(p, n):
            dm.append([1] + [data[t - i - 1] for i in range(p)])
        dm = numpy.array(dm).astype(float)
        tm = data[p:]
        return dm, tm

    @classmethod
    def fit(cls, data: Union[list, tuple, numpy.ndarray], p: int = 1, *args, **kwargs) -> Self:
        """
        拟合模型
        """
        data = numpy.array(data).astype(float)
        dm, tm = cls.design_matrix(data, p)
        xtx_inv = numpy.linalg.inv(dm.T @ dm)
        xty = dm.T @ tm
        beta = xtx_inv @ xty

        mu = beta[0]
        x = beta[1:]

        new_ar = cls(mu=mu, phi=x, sigma=1)
        sigma = numpy.std(new_ar.stochastic_component(data), ddof=1)
        return cls(mu=mu, phi=x, sigma=sigma)


if __name__ == "__main__":
    ar = ARProcess()
    d = ar.next([0], 100)
    print(ARProcess.fit(d, 1))
