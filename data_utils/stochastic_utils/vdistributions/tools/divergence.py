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
        x = float(x)
        return (dist.cdf(x) - (1 if x >= value else 0)) ** 2

    domain_min, domain_max = dist.domain()
    result, _ = quad(f, domain_min, domain_max)
    return result


def quantile_RMSE(dist_0: AbstractDistribution, dist_1: AbstractDistribution):
    q = numpy.arange(0.01, 0.99, 0.01)
    q0 = dist_0.ppf(q)
    q1 = dist_1.ppf(q)
    return numpy.sqrt(numpy.sum((q0 - q1) ** 2))


def js_divergence_continuous(dist_0, dist_1):
    def m_pdf(x):
        return 0.5 * (dist_0.pdf(x) + dist_1.pdf(x))

    def integrand(x, pdf_a):
        p = pdf_a(x)
        m = m_pdf(x)
        mask = (p > 1e-12) & (m > 1e-12)
        out = numpy.zeros_like(p)
        out[mask] = p[mask] * numpy.log(p[mask] / m[mask])
        return out

    domain_min = max(dist_0.domain().min, dist_1.domain().min)
    domain_max = min(dist_0.domain().max, dist_1.domain().max)
    x = numpy.linspace(domain_min, domain_max, 500)
    y = integrand(x, dist_0.pdf)
    z = integrand(x, dist_1.pdf)

    kl_p = numpy.trapz(y, x)
    kl_q = numpy.trapz(z, x)

    return 0.5 * (kl_p + kl_q)


def tv_divergence(dist_0: AbstractDistribution, dist_1: AbstractDistribution):
    domain_min = min(dist_0.domain().min, dist_1.domain().min)
    domain_max = max(dist_0.domain().max, dist_1.domain().max)
    x = numpy.linspace(domain_min, domain_max, 500)
    z = numpy.trapz(
        numpy.abs(dist_0.pdf(x) - dist_1.pdf(x)), x
    )
    return 0.5 * z


if __name__ == "__main__":
    from data_utils.stochastic_utils.vdistributions.parameter.continuous.basic import NormalDistribution

    # print(
    #     kl_divergence_continuous(
    #         NormalDistribution(0, 1),
    #         NormalDistribution(0, 100)
    #     )
    # )
    print(crps(NormalDistribution(0, 0.1), 0))
