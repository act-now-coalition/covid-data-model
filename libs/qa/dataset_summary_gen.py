import pandas as pd


def generate_field_summary(series: pd.Series) -> pd.Series:

    has_value = not series.isnull().all()
    min_date = None
    max_date = None
    max_value = None
    min_value = None
    latest_value = None
    num_observations = 0
    largest_delta = None
    largest_delta_date = None

    if has_value:
        min_date = series.first_valid_index()
        max_date = series.last_valid_index()
        latest_value = series[series.notnull()].iloc[-1]
        max_value = series.max()
        min_value = series.min()
        num_observations = len(series[series.notnull()])
        largest_delta = series.diff().abs().max()
        # If a
        if len(series.diff().abs().dropna()):
            largest_delta_date = series.diff().abs().max()

    results = {
        "has_value": has_value,
        "min_date": min_date,
        "max_date": max_date,
        "max_value": max_value,
        "min_value": min_value,
        "latest_value": latest_value,
        "num_observations": num_observations,
        "largest_delta": largest_delta,
        "largest_delta_date": largest_delta_date,
    }
    return pd.Series(results)
