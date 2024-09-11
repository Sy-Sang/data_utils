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
from data_utils.stochastic_utils.distributions.basic_distributions import NormalDistribution
from data_utils.solve_utils.equationNSolve import gradient_descent

# 外部模块
import numpy


# 代码块


class TimeSeriesProcess(ABC):
    """时间序列过程"""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.order = None

    def __str__(self):
        return f"model:{type(self)}, args:{self.args}, kwargs:{self.kwargs}, order:{self.order}"

    def __repr__(self):
        return self.__str__()

    def noise(self, noise_dist: ABCDistribution = None, sigma: float = 0, num: int = 100):
        """
        生成噪音序列, 通常为正态分布, 可以通过使用<noise_dist=True>指定噪音分布
        """
        if noise_dist is None:
            noise_dist = NormalDistribution(0, sigma)

        return noise_dist.rvf(num)

    @abstractmethod
    def first_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, *args,
                    **kwargs) -> numpy.ndarray:
        pass

    @abstractmethod
    def high_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, *args,
                   **kwargs) -> numpy.ndarray:
        pass

    def random_process(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, *args,
                       **kwargs) -> numpy.ndarray:
        """随机过程"""
        if self.order == 1:
            return self.first_order(first, num, *args, **kwargs)
        elif self.order > 1:
            return self.high_order(first, num, *args, **kwargs)
        else:
            raise Exception(f"order error: {self.order}")


class ARProcess(TimeSeriesProcess):
    """自回归过程"""

    def __init__(self, mu: float = 0, phi: Union[list, tuple, numpy.ndarray] = [0.1], sigma: float = 1.0):
        super().__init__(mu=mu, phi=phi, sigma=sigma)
        self.mu = mu
        self.phi = numpy.array(phi).astype(float)
        self.sigma = sigma
        self.order = len(phi)

    def first_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, *args,
                    **kwargs) -> numpy.ndarray:
        """一阶过程的随机序列"""
        p = numpy.array(list(first) + [0] * (num - len(first))).astype(float)
        eps = self.noise(sigma=self.sigma, num=num, *args, **kwargs)
        for i in range(len(first), num):
            p[i] = self.mu + p[i - 1] * self.phi[0] + eps[i]
        return p

    def high_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, *args,
                   **kwargs) -> numpy.ndarray:
        """高阶过程的随机序列"""
        p = numpy.array(list(first) + [0] * (num - len(first))).astype(float)
        eps = self.noise(sigma=self.sigma, num=num, *args, **kwargs)
        for t in range(max(self.order, len(first)), num):
            p[t] = self.mu + sum(self.phi[i] * p[t - 1 - i] for i in range(self.order)) + eps[t]
        return p

    # @classmethod
    # def fit(cls, data: Union[list, tuple, numpy.ndarray], p: int = 1, epoch: int = 2000):
    #     """
    #     拟合模型
    #     """
    #     data = numpy.array(data).astype(float)
    #     x = numpy.array([0] + [0.1] * p + [1]).astype(float)
    #     first = data[:p]
    #     y = data[p:]
    #
    #     def f(_x):
    #         phi = [i for i in x[:p]]
    #         sigma = x[p]
    #         values = cls(phi=phi, sigma=sigma).random_process(first=first, num=len(data) - p)
    #         return values
    #
    #     xlist = gradient_descent(f, x, y, epoch=epoch, print_loss=True)
    #     return xlist


class MAProcess(TimeSeriesProcess):
    def __init__(self, mu: float = 0, theta: Union[list, tuple, numpy.ndarray] = [1], sigma: float = 1):
        super().__init__(mu=mu, theta=theta, sigma=sigma)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.order = len(theta)

    def first_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, *args, **kwargs):
        eps = self.noise(sigma=self.sigma, num=num, *args, **kwargs)
        p = numpy.array(list(first) + [0] * (num - len(first))).astype(float)
        for i in range(len(first), num):
            p[i] = self.mu + eps[i] + self.theta[0] * eps[i - 1]
        return p

    def high_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, *args,
                   **kwargs) -> numpy.ndarray:
        p = numpy.array(list(first) + [0] * (num - len(first))).astype(float)
        eps = self.noise(sigma=self.sigma, num=num, *args, **kwargs)
        for t in range(max(self.order, len(first)), num):
            p[t] = self.mu + eps[t] + sum(self.theta[i] * p[t - 1 - i] for i in range(self.order))
        return p


class ARMAProcess(TimeSeriesProcess):
    """自回归移动平均过程"""

    def __init__(self, mu: float = 0, phi: Union[list, tuple, numpy.ndarray] = [0.1],
                 theta: Union[list, tuple, numpy.ndarray] = [1],
                 sigma=1):
        super().__init__(mu, phi=phi, theta=theta, sigma=sigma)
        self.mu = mu
        self.phi = phi
        self.theta = theta
        self.sigma = sigma
        self.p = len(phi)
        self.q = len(theta)
        self.order = max(self.p, self.q)

    def first_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, *args,
                    **kwargs) -> numpy.ndarray:
        ar = ARProcess(mu=self.mu, phi=self.phi, sigma=self.sigma).random_process(first, num, *args, **kwargs)
        ma = MAProcess(mu=self.mu, theta=self.theta, sigma=self.sigma).random_process(first, num, *args, **kwargs)
        return ar + ma

    def high_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, *args,
                   **kwargs) -> numpy.ndarray:
        return self.first_order(first, num, *args, **kwargs)


class ARIMAProcess(ARMAProcess):
    """ARIMA差分自回归移动平均过程"""

    def __init__(self, mu, phi: Union[list, tuple, numpy.ndarray] = [0.1],
                 theta: Union[list, tuple, numpy.ndarray] = [1],
                 sigma=1, d=1):
        super().__init__(mu, phi, theta, sigma)
        self.d = d

    def first_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, *args,
                    **kwargs) -> numpy.ndarray:
        arma = super().first_order(first, num, *args, **kwargs)
        for _ in range(self.d):
            arma = numpy.cumsum(arma)
        return arma

    def high_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, *args,
                   **kwargs) -> numpy.ndarray:
        return self.first_order(first, num, *args, **kwargs)


if __name__ == "__main__":
    from data_utils.stochastic_utils.distributions.basic_distributions import WeibullDistribution

    ar = ARProcess(0, [0.9], 1)
    # fit = ARProcess.fit(ar, 1, 200)
    print(ar)
    r = ar.random_process(first=[0.5, 0.3, 0.4, 0.3, 0.2], num=100).tolist()
    print(r)
    # print(ARProcess.fit(r, p=3, epoch=200))

    # ar = ARProcess([0.1, 0.2])
    # ma = MAProcess()
    # arma = ARMAProcess()
    # arima = ARIMAProcess(0, [0.1], [0.2], d=1)
    # print(ar.forward(0))
    # par = ar.random_process().tolist()
    # pma = ma.random_process().tolist()
    # parma = ma.random_process().tolist()
    # parima = arima.random_process(noise_dist=NormalDistribution(-1, 5)).tolist()
    # print(p)
    # ip = ar.noize(p)
    # print(ip.tolist())
    # print(par)
    # print(pma)
    # print(parma)
    # print(parima)
