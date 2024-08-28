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
from data_utils.stochastic_utils.dist_utils import ABCDistribution
from data_utils.stochastic_utils.basic_distributions import NormalDistribution
from data_utils.serial_utils.series_trans_utils import MinMax, ZScore

# 外部模块
import numpy
from sklearn.preprocessing import PowerTransformer

# 代码块


def uniformed(data:Union[list, tuple, numpy.ndarray], mm_min:float=0, mm_max:float=1) -> numpy.ndarray:
    """
    将数据变为均匀分布
    """
    array = numpy.array(data).reshape(-1, 1)
    nd = NormalDistribution(0, 1)

    transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    fit = transformer.fit_transform(array)
    transformed_data = fit.flatten()
    return nd.cdf(transformed_data).y


def dist_switch(data:Union[list, tuple, numpy.ndarray], dist:ABCDistribution) -> numpy.ndarray:
    """
    将数据变为特定分布
    """
    return dist.ppf(uniformed(data)).y






if __name__ == "__main__":
    from data_utils.stochastic_utils.basic_distributions import WeibullDistribution
    print(WeibullDistribution.parameterlength)

    w = WeibullDistribution(2, 5)
    # print(uniformed(w.rvf(100)).tolist())
    print(dist_switch(w.rvf(1000), NormalDistribution(0,1)).tolist())
