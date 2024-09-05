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
from easy_datetime.timestamp import TimeStamp, TimeLine
from data_utils.serial_utils.time_series import TimeSeries
from easy_datetime.temporal_utils import timer

# 外部模块
import numpy


# 代码块

class TimeTable(TimeSeries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.order_tuple = [
            [x.year, x.month, x.day, x.hour, x.minute, x.second, x.microsecond] for x in self.time_stamp.data
        ]

    @timer
    def line_up(self, other: Self, *args, **kwargs):
        """对齐数据表"""
        up_array = numpy.array([])
        other_array = other.get_array()
        s = 0
        for i, oi in enumerate(self.order_tuple):
            matched = False
            for j in range(s, len(other.order_tuple)):
                if oi == other.order_tuple[j]:
                    if i == 0:
                        up_array = numpy.concatenate((up_array, other_array[s][1:]))
                    else:
                        up_array = numpy.column_stack((up_array, other_array[s][1:]))
                    matched = True
                    s = j + 1
                    break
                else:
                    pass
            if matched is False:
                if i == 0:
                    up_array = numpy.concatenate((up_array, numpy.zeros(other.width - 1)))
                else:
                    up_array = numpy.column_stack((up_array, numpy.zeros(other.width - 1)))

        key_list = self.key_list + other.key_list[1:]
        array = numpy.column_stack((self.get_array(), up_array.T))
        return self._comb_by_key_and_value(key_list, array)


if __name__ == "__main__":
    drange = TimeStamp.timestamp_range("2021-1-1", "2022-1-1", "day", 1, include_last=True)
    print(TimeStamp.now())
    drange2 = TimeStamp.timestamp_range("2021-1-1", "2022-1-1", "hour", 1, include_last=True)
    print(TimeStamp.now())
    data = numpy.arange(0, len(drange) * 3).reshape(3, -1)
    data2 = numpy.arange(0, len(drange2) * 3).reshape(3, -1)

    tb = TimeTable(drange, *data)
    print(TimeStamp.now())
    tb2 = TimeTable(drange2, d1=data2[0], d2=data2[1], d3=data2[2])
    print(TimeStamp.now())
    new_tb = tb.line_up(tb2, timer=True)
    print(TimeStamp.now())

    print(new_tb.aggregate(["month", 1]))
    print(new_tb.aggregate(["month", 1]).column_entangled(sum, ["column_1", "d1"], "[[{},{}]]"))

    TimeTable.comb_by_key_and_value()
