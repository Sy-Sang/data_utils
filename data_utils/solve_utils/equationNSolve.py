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
from typing import Union, Self, Type
from collections import namedtuple
import decimal

# 项目模块
from data_utils.solve_utils.lossFunction import mae, mse
from data_utils.solve_utils.equationSloveOptimizer import Optimizer, Adam
from easy_datetime.temporal_utils import timer

# 外部模块
import numpy

# 代码块

Eps = numpy.finfo(float).eps


def print_loss(f):
    """是否打印损失"""

    def wrapper(*args, **kwargs):
        s, l = f(*args, **kwargs)
        if "print_loss" in kwargs.keys() and kwargs["print_loss"] is True:
            return s, l
        else:
            return s

    return wrapper


@print_loss
def newton_method(f: callable, x0: Union[float, int, decimal.Decimal, numpy.floating],
                  tol=1e-6, max_iter=100, dx=1e-6, *args, **kwargs) -> tuple[float, float]:
    """
    牛顿法
    :param f:
    :param x0:
    :param tol:
    :param max_iter:
    :param dx:
    :return:
    """
    x = x0
    iter_count = 0
    while abs(f(x, *args, **kwargs)) > tol and iter_count < max_iter:
        f_prime = (f(x + dx, *args, **kwargs) - f(x, *args, **kwargs)) / dx
        if f_prime == 0:
            break
        else:
            x = x - f(x, *args, **kwargs) / f_prime
            iter_count += 1
    return x, abs(f(x, *args, **kwargs)) - tol


@timer
def bfgs(
        f: callable,
        x0: Union[list, tuple, numpy.ndarray],
        eps: float = 1e-6,
        tol: float = 1e-6,
        epoch: int = 1000,
        *args, **kwargs
):
    """
    拟牛顿法
    """

    def finite_diff_gradient(f, x, eps=1e-8):
        """使用有限差分计算梯度"""
        grad = numpy.zeros_like(x)
        fx = f(x)
        for i in range(len(x)):
            x_eps = numpy.array(x, dtype=float)
            x_eps[i] += eps
            grad[i] = (f(x_eps) - fx) / eps
        return grad

    n = len(x0)
    x = numpy.array(x0).astype(float)
    I = numpy.eye(n)  # 单位矩阵
    H = I  # 初始化为单位矩阵的逆
    grad = finite_diff_gradient(f, x, eps)
    for i in range(epoch):
        if numpy.linalg.norm(grad) < tol:
            break

        p = -H.dot(grad)
        alpha = 1
        while f(x + alpha * p) > f(x) + 1e-4 * alpha * grad.dot(p):
            alpha *= 0.5  # 缩小步长
        x_new = x + alpha * p
        grad_new = finite_diff_gradient(f, x_new)
        # BFGS 公式更新 H 矩阵
        s = x_new - x
        y = grad_new - grad

        if numpy.dot(s, y) > 1e-10:  # 确保数值稳定性，防止除以零或接近零的情况
            rho = 1.0 / numpy.dot(y, s)
            Hy = H.dot(y)
            H = (I - rho * numpy.outer(s, y)).dot(H).dot(I - rho * numpy.outer(y, s)) + rho * numpy.outer(s, s)

        x = x_new
        grad = grad_new

    return x


@timer
@print_loss
def gradient_descent(
        f: callable,
        x: Union[list, tuple, numpy.ndarray],
        y: Union[list, tuple, numpy.ndarray],
        loss_function: callable = mae,
        optimizer: Type[Optimizer] = Adam,
        eps: float = 1e-5,
        lr: float = 0.1,
        epoch: int = 200,
        *args,
        **kwargs
) -> numpy.ndarray:
    """基础梯度下降求解"""
    grad = numpy.zeros(len(x))
    x = numpy.array(x).astype(float)
    y = numpy.array(y).astype(float)
    loss_list = []
    opt = optimizer(x)

    for ep in range(epoch):
        for i, xi in enumerate(x):
            dx_plus = xi + eps
            dx_minus = xi - eps

            delta_x = numpy.array([[xj, xj] if j != i else [dx_plus, dx_minus] for j, xj in enumerate(x)])

            loss_plus = loss_function(f(delta_x[:, 0]), y)
            loss_minus = loss_function(f(delta_x[:, 1]), y)

            grad[i] = (loss_plus - loss_minus) / (2 * eps)
        optimized_grad = opt.next(grad)
        x -= optimized_grad * lr
        loss = loss_function(f(x), y)
        loss_list.append([ep, loss])
    return x, loss_list


if __name__ == "__main__":
    def f(x):
        return x ** 2 + x + 1


    def g(x):
        return x ** 2 - 10


    print(gradient_descent(f, [0], [10.0], epoch=200, optimizer=Adam, timer=True, print_loss=True))
    # print(g(bfgs(g, [1], timer=True)))
    print(newton_method(g, 0.0))
