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
from data_utils.stochastic_utils.vdistributions.abstract import AbstractDistribution, eps

# 外部模块
import numpy
from scipy.integrate import quad


# 代码块

def kl_divergence_continuous(dist_0: AbstractDistribution, dist_1: AbstractDistribution):
    """kl散度"""

    def integrand(x):
        px = dist_0.pdf(x)
        qx = dist_1.pdf(x)
        if px <= 1e-12 or qx <= 1e-12:
            return 0.0
        return px * numpy.log(px / qx)

    domain_0 = dist_0.domain()
    domain_1 = dist_1.domain()

    domain_min = numpy.max([domain_0.min, domain_1.min])
    domain_max = numpy.min([domain_0.max, domain_1.max])

    result, _ = quad(integrand, domain_min, domain_max)
    return result


def crps(dist: AbstractDistribution, value):
    """Continuous Ranked Probability Score"""

    def f(x):
        return (dist.cdf(x) - numpy.where(x >= value, 1, 0)) ** 2

    domain_min, domain_max = dist.domain()
    result, _ = quad(f, domain_min, domain_max)
    return result


def quantile_RMSE(dist_0: AbstractDistribution, dist_1: AbstractDistribution):
    q = numpy.arange(0.01, 0.99, 0.01)
    q0 = dist_0.ppf(q)
    q1 = dist_1.ppf(q)
    return numpy.sqrt(numpy.sum((q0 - q1) ** 2))


if __name__ == "__main__":
    from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution

    # print(
    #     kl_divergence_continuous(
    #         NormalDistribution(0, 1),
    #         NormalDistribution(0, 100)
    #     )
    # )
    print(crps(NormalDistribution(0, 0.1), 0))
