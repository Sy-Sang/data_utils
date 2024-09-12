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
from data_utils.solve_utils.equationNSolve import gradient_descent, bfgs

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

    def noise(self, noise_dist: ABCDistribution = None, sigma: float = None, num: int = 100):
        """
        生成噪音序列, 通常为正态分布, 可以通过使用<noise_dist=True>指定噪音分布
        """
        if noise_dist is None:
            if sigma is None:
                return numpy.zeros(num)
            else:
                noise_dist = NormalDistribution(0, sigma)

        return noise_dist.rvf(num)

    @abstractmethod
    def first_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, no_noize=False, *args,
                    **kwargs) -> numpy.ndarray:
        pass

    @abstractmethod
    def high_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, no_noize=False, *args,
                   **kwargs) -> numpy.ndarray:
        pass

    @classmethod
    @abstractmethod
    def design_matrix(cls, data: Union[list, tuple, numpy.ndarray], p: int = 1) -> tuple[numpy.ndarray, numpy.ndarray]:
        """设计矩阵"""
        pass

    @classmethod
    @abstractmethod
    def fit(cls, data: Union[list, tuple, numpy.ndarray], p: int = 1, *args, **kwargs) -> Self:
        """从数据拟合模型"""
        pass

    def random_process(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, no_noize: bool = False,
                       *args, **kwargs) -> numpy.ndarray:
        """随机过程"""
        if self.order == 1:
            return self.first_order(first=first, num=num, no_noize=no_noize, *args, **kwargs)
        elif self.order > 1:
            return self.high_order(first=first, num=num, no_noize=no_noize, *args, **kwargs)
        else:
            raise Exception(f"order error: {self.order}")

    def data_noise(self, data: Union[list, tuple, numpy.ndarray]) -> float:
        """逆推噪音分布"""
        n = len(data)
        data = numpy.array(data).astype(float)
        noise = []
        for t in range(self.order, n):
            first = data[:t]
            walk = self.random_process(first=first, num=len(first) + 1, no_noize=True)
            noise.append(data[t] - walk[t])
        return numpy.std(noise, ddof=1)


class ARProcess(TimeSeriesProcess):
    """自回归过程"""

    def __init__(self, mu: float = 0, phi: Union[list, tuple, numpy.ndarray] = [0.1], sigma: float = 1.0):
        super().__init__(mu=mu, phi=phi, sigma=sigma)
        self.mu = mu
        self.phi = numpy.array(phi).astype(float)
        self.sigma = sigma
        self.order = len(phi)

    def first_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, no_noize: bool = False, *args,
                    **kwargs) -> numpy.ndarray:
        """一阶过程的随机序列"""
        p = numpy.array(list(first) + [0] * (num - len(first))).astype(float)
        if no_noize is True:
            eps = self.noise(num=num)
        else:
            eps = self.noise(sigma=self.sigma, num=num, *args, **kwargs)
        for i in range(len(first), num):
            p[i] = self.mu + p[i - 1] * self.phi[0] + eps[i]
        return p

    def high_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, no_noize: bool = False, *args,
                   **kwargs) -> numpy.ndarray:
        """高阶过程的随机序列"""
        p = numpy.array(list(first) + [0] * (num - len(first))).astype(float)
        if no_noize is True:
            eps = self.noise(num=num)
        else:
            eps = self.noise(sigma=self.sigma, num=num, *args, **kwargs)
        for t in range(max(self.order, len(first)), num):
            p[t] = self.mu + sum(self.phi[i] * p[t - 1 - i] for i in range(self.order)) + eps[t]
        return p

    @classmethod
    def design_matrix(cls, data: Union[list, tuple, numpy.ndarray], p: int = 1) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        设计矩阵
        """
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
        sigma = new_ar.data_noise(data)
        return cls(mu=mu, phi=x, sigma=sigma)


class MAProcess(TimeSeriesProcess):
    def __init__(self, mu: float = 0, theta: Union[list, tuple, numpy.ndarray] = [1], sigma: float = 1):
        super().__init__(mu=mu, theta=theta, sigma=sigma)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.order = len(theta)

    def first_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, no_noize: bool = False, *args,
                    **kwargs):
        if no_noize is True:
            eps = self.noise(num=num)
        else:
            eps = self.noise(sigma=self.sigma, num=num, *args, **kwargs)
        p = numpy.array(list(first) + [0] * (num - len(first))).astype(float)
        for i in range(len(first), num):
            p[i] = self.mu + eps[i] + self.theta[0] * eps[i - 1]
        return p

    def high_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, no_noize: bool = False, *args,
                   **kwargs) -> numpy.ndarray:
        p = numpy.array(list(first) + [0] * (num - len(first))).astype(float)
        if no_noize is True:
            eps = self.noise(num=num)
        else:
            eps = self.noise(sigma=self.sigma, num=num, *args, **kwargs)
        for t in range(max(self.order, len(first)), num):
            p[t] = self.mu + eps[t] + sum(self.theta[i] * p[t - 1 - i] for i in range(self.order))
        return p

    @classmethod
    def design_matrix(cls, data: Union[list, tuple, numpy.ndarray], p: int = 1) -> tuple[numpy.ndarray, numpy.ndarray]:
        raise Exception(f"{cls.__name__} has no design matrix")

    @classmethod
    def fit(cls, data: Union[list, tuple, numpy.ndarray], p: int = 1, epoh=1000, *args, **kwargs) -> Self:
        data = numpy.array(data).astype(float)

        def log_likelihood(params):
            mu = params[0]
            thetas = params[1:p + 1]
            sigma2 = numpy.abs(params[p + 1])

            n = len(data)
            errors = numpy.zeros(n)

            # 初始误差设为0
            for t in range(p, n):
                errors[t] = data[t] - (mu + numpy.dot(thetas, errors[t - p:t][::-1]))

            # 计算对数似然函数值
            logL = -0.5 * n * numpy.log(2 * numpy.pi * sigma2) - (numpy.sum(errors ** 2) / (2 * sigma2))
            return -logL  # 最小化负对数似然函数

        x0 = numpy.array([0] + [1] * p + [1]).astype(float)
        x = bfgs(log_likelihood, x0, epoch=epoh)
        return x


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

    def first_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, no_noize: bool = False, *args,
                    **kwargs) -> numpy.ndarray:
        ar = ARProcess(mu=self.mu, phi=self.phi, sigma=self.sigma).random_process(first, num, no_noize=no_noize, *args,
                                                                                  **kwargs)
        ma = MAProcess(mu=self.mu, theta=self.theta, sigma=self.sigma).random_process(first, num, no_noize=no_noize,
                                                                                      *args, **kwargs)
        return ar + ma

    def high_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, no_noize: bool = False, *args,
                   **kwargs) -> numpy.ndarray:
        return self.first_order(first, num, no_noize=no_noize, *args, **kwargs)


class ARIMAProcess(ARMAProcess):
    """ARIMA差分自回归移动平均过程"""

    def __init__(self, mu, phi: Union[list, tuple, numpy.ndarray] = [0.1],
                 theta: Union[list, tuple, numpy.ndarray] = [1],
                 sigma=1, d=1):
        super().__init__(mu, phi, theta, sigma)
        self.d = d

    def first_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, no_noize: bool = False, *args,
                    **kwargs) -> numpy.ndarray:
        arma = super().first_order(first, num, no_noize=no_noize, *args, **kwargs)
        for _ in range(self.d):
            arma = numpy.cumsum(arma)
        return arma

    def high_order(self, first: Union[list, tuple, numpy.ndarray] = [0], num: int = 100, no_noize: bool = False, *args,
                   **kwargs) -> numpy.ndarray:
        return self.first_order(first, num, no_noize=no_noize, *args, **kwargs)


if __name__ == "__main__":
    from data_utils.stochastic_utils.distributions.basic_distributions import WeibullDistribution

    ar = MAProcess(0, [0.1], 3)
    # fit = ARProcess.fit(ar, 1, 200)
    r = ar.random_process(first=[0], num=2000).tolist()
    # nr = ar.random_process(first=[0.5, 0.3, 0.4, 0.3, 0.2], num=100, no_noize=True).tolist()
    # print(r)
    # print(nr)
    print(MAProcess.fit(r, p=1))

    # print(ar.random_process(num=12))
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
