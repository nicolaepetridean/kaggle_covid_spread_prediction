import numpy as np
import pandas as pd


def growth_factor(country_df, target):
    confirmed = country_df[target]
    confirmed_day_minus1 = confirmed.shift(1, axis=0)
    confirmed_day_minus2 = confirmed.shift(2, axis=0)
    return (confirmed - confirmed_day_minus1) / (
        confirmed_day_minus1 - confirmed_day_minus2
    )


def new_cases(country_df, target):
    confirmed = country_df[target]
    return confirmed - confirmed.shift(1)


def active_cases(all_data):
    return all_data["ConfirmedCases"] - all_data["Recovered"] - all_data["Deaths"]


def growth_ratio(country_df, target):

    confirmed = country_df[target]
    confirmed_iminus1 = confirmed.shift(1, axis=0)
    return confirmed / confirmed_iminus1


def growth_rate(all_data, target):
    all_data["GrowthRate"] = np.gradient(np.log(all_data[target]))


def average_column_entries(country_df, target, window_size, std=2):
    return country_df[target].rolling(window_size, win_type="gaussian").mean(std=std)


def smoother(inputdata, w, imax):
    data = 1.0 * inputdata
    data = data.replace(np.nan, 1)
    data = data.replace(np.inf, 1)

    smoothed = 1.0 * data
    normalization = 1
    for i in range(-imax, imax + 1):
        if i == 0:
            continue
        smoothed += (w ** abs(i)) * data.shift(i, axis=0)
        normalization += w ** abs(i)
    smoothed /= normalization
    return smoothed
