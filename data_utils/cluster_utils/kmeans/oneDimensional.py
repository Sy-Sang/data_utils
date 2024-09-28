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
from easy_utils.obj_utils.enumerable_utils import flatten

# 外部模块
import numpy
from sklearn.cluster import KMeans


# 代码块

def find_cluster(data: Union[list, tuple, numpy.ndarray], k: int = 3):
    """聚类"""
    x = numpy.array(data)
    cluster = KMeans(n_clusters=k, random_state=0)
    cluster.fit(x.reshape(len(x), -1))
    label_index = numpy.argsort(flatten(cluster.cluster_centers_))
    label = [label_index.tolist().index(i) for i in cluster.labels_]


if __name__ == "__main__":
    pass
