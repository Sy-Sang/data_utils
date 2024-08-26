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
from dataclasses import dataclass

# 项目模块
from easy_datetime.timestamp import TimeStamp

# 外部模块
import numpy
import httpx


# 代码块


class HistoricalDim:
    """
    历史数据维度
    """
    temperature_2m = "temperature_2m"
    relative_humidity_2m = "relative_humidity_2m"
    dew_point_2m = "dew_point_2m"
    apparent_temperature = "apparent_temperature"
    pressure_msl = "pressure_msl"
    surface_pressure = "surface_pressure"
    precipitation = "precipitation"
    rain = "rain"
    snowfall = "snowfall"
    cloud_cover = "cloud_cover"
    cloud_cover_low = "cloud_cover_low"
    cloud_cover_mid = "cloud_cover_mid"
    cloud_cover_high = "cloud_cover_high"
    shortwave_radiation = "shortwave_radiation"
    direct_radiation = "direct_radiation"
    direct_normal_irradiance = "direct_normal_irradiance"
    diffuse_radiation = "diffuse_radiation"
    global_tilted_irradiance = "global_tilted_irradiance"
    sunshine_duration = "sunshine_duration"
    wind_speed_10m = "wind_speed_10m"
    wind_speed_100m = "wind_speed_100m"
    wind_direction_10m = "wind_direction_10m"
    wind_direction_100m = "wind_direction_100m"
    wind_gusts_10m = "wind_gusts_10m"
    et0_fao_evapotranspiration = "et0_fao_evapotranspiration"
    weather_code = "weather_code"
    snow_depth = "snow_depth"
    vapour_pressure_deficit = "vapour_pressure_deficit"
    soil_temperature_0_to_7cm = "soil_temperature_0_to_7cm"
    soil_temperature_7_to_28cm = "soil_temperature_7_to_28cm"
    soil_temperature_28_to_100cm = "soil_temperature_28_to_100cm"
    soil_temperature_100_to_255cm = "soil_temperature_100_to_255cm"
    soil_moisture_0_to_7cm = "soil_moisture_0_to_7cm"
    soil_moisture_7_to_28cm = "soil_moisture_7_to_28cm"
    soil_moisture_28_to_100cm = "soil_moisture_28_to_100cm"
    soil_moisture_100_to_255cm = "soil_moisture_100_to_255cm"


def historical(lon: float, lat: float, start: str, end: str, *args) -> dict:
    """

    :param lon:
    :param lat:
    :param dim:
    :param start:
    :param end:
    :return:
    """
    dim = ""
    for i, a in enumerate(args):
        dim += f"{a}," if i < len(args) - 1 else f"{a}"
    start_date = TimeStamp(start).get_date_string()
    end_date = TimeStamp(end).get_date_string()
    request_url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}"
        f"&end_date={end_date}&hourly={dim}&timezone=Asia%2FShanghai&windspeed_unit=ms&timeformat=unixtime&models=era5"
    )
    res = httpx.get(
        request_url
    )
    return json.loads(res.text)


if __name__ == "__main__":
    print(historical(
        100, 40, "2024-1-1", "2024-2-1", HistoricalDim.wind_direction_10m, HistoricalDim.wind_speed_100m
    ))
