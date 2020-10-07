import pandas as pd

from libs.datasets import combined_datasets, timeseries
from covidactnow.datapublic.common_fields import CommonFields


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
    SUMMED_CASES_LABEL = "summed_cases"
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=LOOKBACK_DAYS)
    region_groupby = (
        combined_datasets.load_us_timeseries_dataset()
        .get_counties(after=cutoff_date)
        .groupby_region()
    )

    last_month_cum_cases = region_groupby[CommonFields.CASES].apply(_quantile_range)
    last_month_cum_cases.name = SUMMED_CASES_LABEL

    df = last_month_cum_cases.reset_index().dropna()
    timeseries._add_fips_if_missing(df)
    # Example location_id value = 'iso1:us#iso2:us-ak#fips:02013'
    df["state_location_id"] = df[CommonFields.LOCATION_ID.value].str.split("#").str[1]
    # Normalize the cases based on the groupby total
    df["weight"] = df.groupby("state_location_id")[SUMMED_CASES_LABEL].transform(
        lambda x: x / x.sum()
    )
    df["weight"] = df["weight"].round(4)
    # Convert to dict mapping
    output = df.set_index(CommonFields.FIPS.value)["weight"].to_dict()
    return output
