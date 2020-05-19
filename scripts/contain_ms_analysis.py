import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from core.data.utils import preprocess_data
from core.data.stats import smoother, growth_factor, growth_ratio, new_cases
from scripts.preprocess_contain_ms import CLUSTERS

# used to map J.H. Country Names to measure data country names
COUNTRY_MAPPING = {"US": "United States", "Korea, South": "South Korea"}


def get_keywords(df):
    keys = df[df["Keywords"].notna()]
    cats = (
        pd.DataFrame(keys["Keywords"].unique())
        .apply(lambda x: x.str.split(", ").explode())
        .reset_index()
    )[0].unique()
    return sorted(cats)


def plot_spread_df_ln(country_df, target, outname="out", confirmed_treshhold=1):

    outname = f"{outname}.png"
    plot_df = country_df[country_df["ConfirmedCases"] > confirmed_treshhold]
    sns.set()
    sns.set_palette("colorblind")
    plt.figure(figsize=(14, 7))
    sns.lineplot(plot_df["Date"], country_df[target], label="")
    # plt.savefig(outname)
    plt.show()


def check_ms_for_country(
    country_df, measures_df, country_name, target="Smooth_Growth_Factor"
):
    measures_df[["Date Start", "Date end intended"]] = measures_df[
        ["Date Start", "Date end intended"]
    ].apply(pd.to_datetime)

    ms_dict = dict()
    ms_not_implemented = []
    if country_name in COUNTRY_MAPPING:
        country_ms_name = COUNTRY_MAPPING[country_name]
    else:
        country_ms_name = country_name

    for measures in CLUSTERS:
        ms_name = measures[0]
        measures_for_country = measures_df[
            (measures_df[ms_name] > 0.0) & (measures_df["Country"] == country_ms_name)
        ]
        measure_levels = measures_for_country[ms_name].unique()
        if len(measures_for_country) == 0:
            ms_not_implemented.append(ms_name)
            continue
        measure_date = measures_for_country[measures_for_country[ms_name] > 0.0][
            "Date Start"
        ].min()
        week_one = measure_date + pd.DateOffset(days=7)
        week_two = measure_date + pd.DateOffset(days=14)
        weak_tre = measure_date + pd.DateOffset(days=21)

        zero2week_one = (
            country_df[country_df["Date"] == week_one][target].values[0]
            - country_df[country_df["Date"] == measure_date][target].values[0]
        )
        zero2week_two = (
            country_df[country_df["Date"] == week_two][target].values[0]
            - country_df[country_df["Date"] == measure_date][target].values[0]
        )
        zero2week_tre = (
            country_df[country_df["Date"] == weak_tre][target].values[0]
            - country_df[country_df["Date"] == measure_date][target].values[0]
        )
        week_one2week_two = (
            country_df[country_df["Date"] == week_two][target].values[0]
            - country_df[country_df["Date"] == week_one][target].values[0]
        )

        ms_dict[ms_name] = {
            "From Implemntation to 1st week": zero2week_one,
            "From Implemntation to 2nd week": zero2week_two,
            "From Implemntation to 3rd week": zero2week_tre,
            "From 1st week to 2rd week": week_one2week_two,
            "Start_date": measure_date,
        }
    return ms_dict


if __name__ == "__main__":
    countries = [
        "France",
        "Poland",
        "Russia",
        "Sweden",
        "Germany",
        "US",
        "Spain",
        "Italy",
        "Korea, South",
        "Czechia",
        "Romania",
        "United Kingdom",
        "Turkey",
        "Canada",
        "Brazil",
        "Japan",
        "New Zealand",
    ]

    spread_pth = "assets/covid_spread.csv"
    measures_pth = "assets/contain_ms.csv"
    spread_df = preprocess_data(pd.read_csv(spread_pth))
    spread_df["Date"] = pd.to_datetime(spread_df["Date"])
    measures_df = pd.read_csv(measures_pth)

    for country in countries:
        country_df = spread_df[spread_df["Province_State"] == country]
        spread_df.loc[
            spread_df["Province_State"] == country, "Growth_Factor"
        ] = growth_factor(country_df, "ConfirmedCases")
        spread_df.loc[
            spread_df["Province_State"] == country, "Growth_Ratio"
        ] = growth_ratio(country_df, "ConfirmedCases")
        spread_df.loc[
            spread_df["Province_State"] == country, "New_ConfirmedCases"
        ] = new_cases(country_df, "ConfirmedCases")
        country_df = spread_df[spread_df["Province_State"] == country]
        spread_df.loc[
            spread_df["Province_State"] == country, "Growth_Factor"
        ] = smoother(country_df["Growth_Factor"], 0.5, 3)
        country_df = spread_df[spread_df["Province_State"] == country]
        plot_spread_df_ln(country_df, "Growth_Factor", f"{country}Growth_Factor")
        country_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        country_df.fillna({"Growth_Factor": 0, "Growth_Ratio": 0,}, inplace=True)

        ms_dict = check_ms_for_country(
            country_df, measures_df, country, "Growth_Factor"
        )

        # plot_ms_dict(ms_dict, country)
