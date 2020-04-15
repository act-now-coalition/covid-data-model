import os
from pyseir.load_data import load_county_metadata
from pyseir import OUTPUT_DIR
import pandas as pd
from datetime import datetime
import us


def load_t0(fips):
    """
    Load the simulation start time by county.

    Parameters
    ----------
    fips: str
        County FIPS

    Returns
    -------
    : datetime
        t0(C=1) cases.
    """
    county_metadata = load_county_metadata().set_index("fips")
    state = county_metadata.loc[fips]["state"]
    fit_results = os.path.join(
        OUTPUT_DIR, "pyseir", state, "data", f"summary__{state}_imputed_start_times.pkl"
    )
    return datetime.fromtimestamp(
        pd.read_pickle(fit_results).set_index("fips").loc[fips]["t0_date"].timestamp()
    )


def load_inference_result(fips):
    """
    Load fit results by state or county fips code.

    Parameters
    ----------
    fips: str
        2 or 5 digit fips code.

    Returns
    -------
    : dict
        Dictionary of fit result information.
    """

    if len(fips) == 2:
        state = us.states.lookup(fips).name
        state_output_file = os.path.join(
            OUTPUT_DIR,
            "pyseir",
            "data",
            "state_summary",
            f"summary_{state}_state_only__mle_fit_results.json",
        )
    else:
        raise NotImplementedError()

    return pd.read_json(state_output_file).iloc[0].to_dict()
