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

import numpy

# 项目模块
from data_utils.stochastic_utils.vdistributions.abstract import eps
from data_utils.stochastic_utils.vdistributions.parameter.abstract import ParameterDistribution, DistributionParams

# 外部模块
import numpy as np


# 代码块

class UniformDistribution(ParameterDistribution):
    """均匀分布"""

    def __init__(self, min: float, max: float):
        super().__init__(min, max, **{"min": min, "max": max})
        self.min = min
        self.max = max

    def ppf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        result = numpy.where(
            (x > 0) & (x < 1),
            self.min + (self.max - self.min) * x,
            numpy.nan
        )
        return result

    def pdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        result = numpy.where(
            (x >= self.min) & (x <= self.max),
            1.0 / (self.max - self.min),
            0.0
        )
        return result

    def cdf(self, x, *args, **kwargs):
        x = numpy.asarray(x)
        result = numpy.where(
            x < self.min,
            0.0,
            numpy.where(
                x > self.max,
                1.0,
                (x - self.min) / (self.max - self.min)
            )
        )
        return result

    def get_param_constraints(self, args) -> list[DistributionParams]:
        return [
            DistributionParams("min", -numpy.inf, numpy.inf),
            DistributionParams("max", args[0] + eps, numpy.inf)
        ]


if __name__ == "__main__":
    u = UniformDistribution(0, 1)
    print(u.parameter_verification([-5, 0]))
    print(u.cdf([0.1, 0.2]))
    print(u.ppf([0.1, 0.2, 1]))
