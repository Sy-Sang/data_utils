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
from data_utils.stochastic_utils.vdistributions.nonparametric.continuous.kernel2 import KernelMixDistribution
from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution

# 外部模块
import numpy
from scipy.optimize import differential_evolution, minimize
from scipy.stats import skew, kurtosis


# 代码块

def moment_loss(x, mu_target, var_target, skew_target, kurt_target):
    mean = numpy.mean(x)
    var = numpy.var(x)
    s = skew(x)
    k = kurtosis(x, fisher=False)  # 设置为False保留总体峰度（不是excess）
    loss = ((mean - mu_target) ** 2 +
            (var - var_target) ** 2 +
            (s - skew_target) ** 2 +
            (k - kurt_target) ** 2)
    return loss


def moment_constrained_sampler(mu_target, var_target, skew_target, kurt_target, num=100):
    seed_data = NormalDistribution(mu_target, var_target ** 0.5).rvf(num)
    bounds = [(-numpy.inf, numpy.inf)] * num
    result = minimize(moment_loss, seed_data, bounds=bounds, args=(mu_target, var_target, skew_target, kurt_target))
    return KernelMixDistribution(result.x)


def moment_matching_sampler(mu_target, var_target, skew_target, kurt_target, num=100, max_iter=1000, perturb_scale=0.2):
    x = NormalDistribution(mu_target, var_target ** 0.5).rvf(num)
    best_x = x.copy()
    best_loss = moment_loss(x, mu_target, var_target, skew_target, kurt_target)
    loss_trace = [best_loss]
    replace_size = int(num * 0.05)
    replace_dist = moment_constrained_sampler(mu_target, var_target, skew_target, kurt_target, num)
    for i in range(max_iter):
        x_new = x.copy()
        idx = numpy.random.choice(num, size=replace_size, replace=False)
        # x_new[idx] += numpy.random.normal(0, perturb_scale, size=replace_size)
        x_new[idx] = replace_dist.rvf(replace_size)

        new_loss = moment_loss(x_new, mu_target, var_target, skew_target, kurt_target)

        T = 1.0 * (1 - i / max_iter)
        # if new_loss < best_loss or numpy.random.rand() < numpy.exp((best_loss - new_loss) / (T + 1e-6)):
        if new_loss < best_loss:
            x = x_new
            best_loss = new_loss
            best_x = x_new.copy()
            loss_trace.append(best_loss)

        if i % int(max_iter / 20) == 0:
            perturb_scale *= 0.95

    return best_x, loss_trace


if __name__ == "__main__":
    data, loss_trace = moment_matching_sampler(0, 1, 3, 1)
    print(data.tolist())
    print(loss_trace)