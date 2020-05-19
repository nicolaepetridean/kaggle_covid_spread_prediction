import pandas as pd
import numpy as np

# Cluster defintions for keyword clusters
CLUSTERS = [
    ["education_closure", "university closure|school closure"],
    ["outdoor_gatherings_banned", "outdoor gatherings banned"],
    ["sports cancellation", "sports cancellation"],
    ["compulsory isolation", "compulsory isolation"],
    [
        "international travel ban",
        "international travel ban - risk countries|international travel ban - all countries",
        [
            "international travel ban - risk countries",
            "international travel ban - all countries",
        ],
    ],
    [
        "business suspension",
        "general nonessential business suspension|limited nonessential business suspension|closure nonessential stores",
    ],
    [
        "blanket isolation",
        "blanket isolation - no symptoms|blanket isolation - symptoms",
        ["blanket isolation - symptoms", "blanket isolation - no symptoms"],
    ],
    ["blanket curfew", "blanket curfew "],
    ["social distancing", "social distancing"],
]


def load_containment_measures(ms_path):
    ms_df = pd.read_csv(ms_path)
    ms_df = ms_df.drop(
        [
            "Target city",
            "Target state",
            "Source",
            "Applies To",
            "Exceptions",
            "Implementing City",
            "Target country",
            "Quantity",
        ],
        axis=1,
    )
    ms_df = ms_df.dropna(subset=["Keywords"])

    out = pd.DataFrame()
    # cluster keywords according to CLUSTER
    for cluster in CLUSTERS:
        cluster_name = cluster[0]
        cluster_query = cluster[1]

        if len(cluster) == 2:
            ms_df.loc[ms_df["Keywords"].str.contains(cluster_query), cluster_name] = 1

        else:
            cluster_cats = cluster[2]
            for i, cat in enumerate(cluster_cats):
                ms_df.loc[ms_df["Keywords"].str.contains(cat), cluster_name] = i + 1

        out = pd.concat(
            [out, ms_df[ms_df["Keywords"].str.contains(cluster[1])]], sort=False
        )

    out["Date Start"] = pd.to_datetime(out["Date Start"])
    out["Date end intended"] = pd.to_datetime(out["Date end intended"])
    return out


if __name__ == "__main__":
    measures_pth = "assets/contain_ms.csv"
    measures_df = load_containment_measures("assets/contain_ms_raw.csv")
    measures_df.to_csv(measures_pth)
