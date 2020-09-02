from typing import Dict
from collections import defaultdict


import pandas as pd


def build_colleges_by_fips(
    college_df: pd.DataFrame, full_time_enrollement_threshold=1000
) -> Dict[str, Dict]:
    """Builds dictionary of colleges by fips passing an enrollment threshold.

    Args:
        college_df: DataFrame of college data.
        full_time_enrollement_threshold: enrollment threshold.

    Returns: Details on colleges per fips.
    """

    has_on_campus_housing = college_df.HOUSING == 1
    not_closed = college_df.CLOSE_DATE == "-2"
    grants_degree = college_df.DEG_GRANT == 1

    college_df["fips"] = college_df["STFIPS"] + college_df["COFIPS"]
    college_df["NAME"] = college_df["NAME"].str.capitalize()

    colleges = college_df[has_on_campus_housing & not_closed & grants_degree]
    colleges = colleges[["fips", "NAME", "FT_ENROLL"]].rename(
        {"NAME": "name", "FT_ENROLL": "ft_enroll"}, axis="columns"
    )
    colleges_by_fips = defaultdict(list)
    for college in colleges.to_dict(orient="records"):
        if college["ft_enroll"] < full_time_enrollement_threshold:
            continue

        colleges_by_fips[college["fips"]].append(college)

    return colleges_by_fips
