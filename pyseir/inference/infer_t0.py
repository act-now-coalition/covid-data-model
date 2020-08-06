import pandas as pd
from libs.datasets import combined_datasets
from libs.datasets import CommonFields


def infer_t0(fips, method="first_case", default=pd.Timestamp("2020-02-01")):
    """
    Infer t0 for a given fips under given methods:
       - first_case: t0 is set as time of first observed case.
       - impute: t0 is imputed.
    Returns default value if neither method works.

    Parameters
    ----------
    fips : str
        County fips
    method : str
        The method to determine t0.
    default : pd.Timestamp
        Default t0 if neither method works.

    Returns
    -------
    t0 : pd.Timestamp
        Inferred t0 for given fips.
    """

    if method == "first_case":
        fips_timeseries = combined_datasets.get_timeseries_for_fips(
            fips, columns=[CommonFields.CASES], min_range_with_some_value=True
        )
        if not fips_timeseries.empty:
            t0 = fips_timeseries[CommonFields.DATE].min()
        else:
            t0 = default
    elif method == "reference_date":
        t0 = default
    else:
        raise ValueError(f"Invalid method {method} for t0 inference")
    return t0
