import json
import os

import pandas as pd

from libs.datasets import combined_datasets
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets.dataset_utils import AggregationLevel
from pyseir import DATA_DIR


def _quantile_range(x: pd.Series) -> float:
    """Compute the quantile range for the cumulative case data.  We sometimes get single points of
    a timeseries that are obviously wrong. Taking the quantile range is less sensitive to outliers
    than either just "last minus first" or "maximum minus minimum"
    """
    QUANTILES = [0.05, 0.95]  # We sometimes get single point obviously wrong input.
    # Quantile range is less sensitive than max-min or last-first.
    lower, upper = x.quantile(q=QUANTILES).values
    return upper - lower


def calculate_case_based_weights() -> dict:
    LOOKBACK_DAYS = 31
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=LOOKBACK_DAYS)
    ts = combined_datasets.load_us_timeseries_dataset().get_subset(
        after=cutoff_date, aggregation_level=AggregationLevel.COUNTY
    )

    last_month_cum_cases = ts.data.groupby("fips")[CommonFields.CASES].apply(_quantile_range)
    last_month_cum_cases.name = "summed_cases"

    df = last_month_cum_cases.reset_index().dropna()
    df["state_fips"] = df["fips"].str.slice(0, 2)  # Apply State Labels to Groupby
    # Normalize the cases based on the groupby total
    df["weight"] = df.groupby("state_fips")["summed_cases"].transform(lambda x: x / x.sum())
    df["weight"] = df["weight"].round(5)
    # Convert to dict mapping
    output = df.set_index("fips")["weight"].to_dict()
    return output
