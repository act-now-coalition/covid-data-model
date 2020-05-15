import pandas as pd
from pyseir.load_data import load_county_case_data
from pyseir.inference import fit_results


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

    if method == "impute":
        t0 = fit_results.load_t0(fips)
    elif method == "first_case":
        case_data = load_county_case_data()
        if fips in case_data.fips:
            t0 = case_data[case_data.fips == fips].date.min()
        else:
            t0 = default
    elif method == "reference_date":
        t0 = default
    else:
        raise ValueError(f"Invalid method {method} for t0 inference")
    return t0
