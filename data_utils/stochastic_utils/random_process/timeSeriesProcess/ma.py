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
from data_utils.stochastic_utils.random_process.timeSeriesProcess.baseclass import TimeSeriesProcess
from data_utils.stochastic_utils.distributions.basic_distributions import NormalDistribution

# 外部模块
import numpy


# 代码块

class MAProcess(TimeSeriesProcess):
    def __init__(self, mu: float = 0, theta: Union[list, tuple, numpy.ndarray] = [1], sigma: float = 1):
        super().__init__(mu=mu, theta=theta, sigma=sigma)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.order = len(theta)

    def next(self, first: Union[list, tuple, numpy.ndarray], num: int = 1, use_eps: bool = True, *args,
             **kwargs) -> numpy.ndarray:
        first = numpy.array(first).astype(float)
        eps = NormalDistribution(0, self.sigma).rvf(num) if use_eps is True else numpy.zeros(num)
        y = numpy.concatenate((first, eps))
        for t in range(max(self.order, len(first)), num):
            y[t] = self.mu + eps[t] + sum(self.theta[i] * eps[t - 1 - i] for i in range(self.order))
        return y


if __name__ == "__main__":
    pass
