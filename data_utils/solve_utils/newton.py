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
import decimal

# 项目模块

# 外部模块
import numpy


# 代码块

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


if __name__ == "__main__":
    pass
