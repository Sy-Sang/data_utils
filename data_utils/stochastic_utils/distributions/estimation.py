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

# 项目模块
from data_utils.stochastic_utils.distributions.baseclass import ABCDistribution
from data_utils.stochastic_utils.distributions.basic_distributions import NormalDistribution, LogNormalDistribution
from data_utils.stochastic_utils.distributions.nonParametricDistribution import HistogramDist, LogHisDist, \
    SmoothHisDist
from easy_utils.number_utils import calculus_utils
from data_utils.solve_utils.equationNSolve import gradient_descent

from easy_datetime.temporal_utils import timer

# 外部模块
import numpy


# 代码块

def sign_log(x):
    """
    Log with Sign
    """
    sign = numpy.array([0 if i == 0 else (1 if i > 0 else -1) for i in x])
    x += numpy.finfo(float).eps
    return sign * numpy.log(numpy.abs(x))


@timer
def moment_ed(
        target: dict,
        dist: Type[ABCDistribution],
        diff: float = 1e-5,
        lr: float = 0.1,
        epoch: int = 200,
        x_len: int = None,
        init_x: Union[list, tuple, numpy.ndarray] = None,
        *args, **kwargs
) -> tuple[ABCDistribution, list]:
    """
    参数估计至指定阶距
    """
    if init_x is None:
        init_param = []
        for item in dist.parameter_range.items():
            init_param.append(float(item[1].default))
        x = numpy.array(init_param) if x_len is None else numpy.array(init_param[:x_len])
    else:
        x = numpy.array(init_x) if x_len is None else numpy.array(init_x[:x_len])

    def moment(x):
        _dist = dist(*[float(i) for i in x])
        return numpy.array([_dist.__getattribute__(k)() for k in target.keys()])

    target_moment = numpy.array([float(i) for i in target.values()])
    # guess, adam_param = calculus_utils.adam_method(moment, x, target_moment, diff, lr, epoch, loss="mae")
    guess = gradient_descent(moment, x, target_moment, diff=diff, lr=lr, epoch=epoch)
    new_dist = dist(*[float(i) for i in guess])
    return new_dist, [float(new_dist.__getattribute__(k)()) for k in target.keys()]


def sang_ed(
        data: Union[list, tuple, numpy.ndarray],
        dist: Type[ABCDistribution],
        diff: float = 1e-5,
        lr: float = 0.1,
        epoch: int = 200,
        x_len: int = None,
        kernel_len: int = None,
        init_x: Union[list, tuple, numpy.ndarray] = None,
        *args, **kwargs
) -> tuple[ABCDistribution, float, float]:
    """
    快速参数估计
    """

    his_dist = LogHisDist(data, kernel_len=kernel_len)

    if init_x is None:
        init_param = []
        for item in dist.parameter_range.items():
            init_param.append(float(item[1].default))
        x = numpy.array(init_param) if x_len is None else numpy.array(init_param[:x_len])
    else:
        x = numpy.array(init_x) if x_len is None else numpy.array(init_x[:x_len])

    def moment(x):
        new_dist = dist(*[float(i) for i in x])
        return numpy.array([new_dist.mean(), new_dist.std()])

    def f(x):
        new_dist = dist(*[float(i) for i in x])
        _curve = new_dist.ppf(first=0.1, end=0.9, step=0.1)
        # _curve = new_dist.ppf()
        return sign_log(_curve.y)

    curve = his_dist.ppf(first=0.1, end=0.9, step=0.1)
    # curve = his_dist.ppf()
    y = sign_log(curve.y)

    if init_x is None:
        target_moment = numpy.array([numpy.mean(data), numpy.std(data, ddof=1)])
        guess, _ = calculus_utils.adam_method(moment, x, target_moment, diff, lr, epoch, loss="mae")
    else:
        guess = x

    parameter, adam_param = calculus_utils.adam_method(f, guess, y, diff, lr, epoch, loss="mae")
    # return dist(*[float(i) for i in parameter]), numpy.mean((f(parameter) - y) ** 2), adam_param
    return dist(*[float(i) for i in parameter]), numpy.mean(abs(f(parameter) - y)), adam_param


class SangDistEstimated:
    """
    参数估计
    """

    def __init__(
            self,
            data: Union[list, tuple, numpy.ndarray],
            loss: float = 0.01
    ):
        self.data = numpy.array(data)
        self.target_abs_loss = loss

    @timer
    def data_estimate(
            self,
            dist: Type[ABCDistribution],
            diff: float = 1e-5,
            lr: float = 0.1,
            epoch: int = 200,
            x_len: int = None,
            kernel_len: int = None,
            max_try=100,
            *args, **kwargs
    ):
        """
        参数估计
        """
        fit_dist, loss, adam_param = sang_ed(data=self.data, dist=dist, diff=diff, lr=lr, epoch=epoch, x_len=x_len,
                                             kernel_len=kernel_len, init_x=None, timer=False
                                             )
        dist_param_list = list(fit_dist.kwargs.values())
        loop_init_x = numpy.array(dist_param_list[:x_len])

        counter = 0
        for i in range(max_try):
            fit_dist, loss, adam_param = sang_ed(data=self.data, dist=dist, diff=diff, lr=lr, epoch=epoch, x_len=x_len,
                                                 kernel_len=kernel_len, init_x=loop_init_x, timer=False)
            dist_param_list = list(fit_dist.kwargs.values())
            loop_init_x -= lr * numpy.array(dist_param_list[:x_len]) * adam_param

            counter += 1

            if loss <= self.target_abs_loss:
                break
            else:
                pass
        return fit_dist, loss, counter

    # @timer
    # def ensemble_estimate(
    #         self,
    #         dist: Type[ABCDistribution],
    #         diff: float = 1e-5,
    #         lr: float = 0.1,
    #         epoch: int = 200,
    #         x_len: int = None,
    #         kernel_len: int = None,
    #         max_try=50,
    #         ensemble_num=10,
    #         *args, **kwargs
    # ):
    #     """集成参数估计"""
    #     dlist = []
    #     llist = []
    #     for i in range(ensemble_num):
    #         d, l = self.forced_estimate(
    #             data=self.data, dist=dist, diff=diff, lr=lr, epoch=epoch, x_len=x_len,
    #             kernel_len=kernel_len, init_x=None, max_try=max_try, timer=False
    #         )
    #         dlist.append(d)
    #         llist.append(l)
    #     lindex = numpy.argsort(llist)
    #     print(llist)
    #     return dlist[lindex[0]], llist[lindex[0]]


if __name__ == "__main__":
    from matplotlib import pyplot
    from data_utils.stochastic_utils.distributions.basic_distributions import NormalDistribution, LogNormalDistribution, \
        WeibullDistribution, StudentTDistribution

    # wd = WeibullDistribution(0.1, 1)
    # wd = WeibullDistribution(150, 500)
    # wd = NormalDistribution(-5, 100)
    # wd = NormalDistribution(-5, 0.1)
    wd = WeibullDistribution(5, 6)
    rwd = wd.rvf(2000)
    est = SangDistEstimated(rwd, loss=0.01)
    lhd = LogHisDist(rwd)
    hd = HistogramDist(rwd)
    # fit, loss = est.forced_estimate(WeibullDistribution, x_len=2, epoch=5000, max_try=1, timer=True)
    fit, loss, c = est.data_estimate(WeibullDistribution, x_len=2, epoch=1000, max_try=10, timer=True)
    print(fit)
    print(c)
    pyplot.plot(fit.ppf().y)
    pyplot.plot(lhd.ppf().y)
    pyplot.plot(hd.ppf().y)
    pyplot.plot(wd.ppf().y)
    pyplot.legend(["fit", "lhd", "hd", "wd"])
    pyplot.show()

    pyplot.hist(sign_log(rwd), density=True)
    pyplot.show()
    pyplot.scatter(x=fit.pdf().x, y=fit.pdf().y)
    pyplot.scatter(x=lhd.pdf().x, y=lhd.pdf().y)
    pyplot.scatter(x=hd.pdf().x, y=hd.pdf().y)
    pyplot.scatter(x=wd.pdf().x, y=wd.pdf().y)
    pyplot.legend(["fit", "lhd", "hd", "wd"])
    pyplot.show()
    # print(loos_list)
    print(loss)
    print(est.target_abs_loss)
