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

class Optimizer:
    """优化器"""

    def __init__(self, *args, **kwargs):
        pass

    def next(self, grad: numpy.ndarray, *args, **kwargs) -> numpy.ndarray:
        """
        梯度下降
        """
        return grad


class Adam(Optimizer):
    """adam优化器"""

    def __init__(self, x: numpy.ndarray, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = numpy.zeros_like(x)
        self.v = numpy.zeros_like(x)
        self.m_hat = 0
        self.v_hat = 0
        self.t = 0

    def next(self, grad: numpy.ndarray, *args, **kwargs) -> numpy.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * numpy.array(grad)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (numpy.array(grad) ** 2)
        self.m_hat = self.m / (1 - self.beta1 ** self.t)
        self.v_hat = self.v / (1 - self.beta2 ** self.t)
        return self.m_hat / (numpy.sqrt(self.v_hat) + self.epsilon)


if __name__ == "__main__":
    adam = Adam(numpy.array([1, 2, 3]))
    grad = numpy.random.uniform(3)
    print(adam.next(grad))
    print(adam.t)
