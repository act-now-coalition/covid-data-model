import os
from pyseir.load_data import load_county_metadata
from pyseir import OUTPUT_DIR
import pandas as pd
from datetime import datetime


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
    county_metadata = load_county_metadata().set_index('fips')
    state = county_metadata.loc[fips]['state']
    fit_results = os.path.join(OUTPUT_DIR, state, 'data', f'summary__{state}_imputed_start_times.pkl')
    return datetime.fromtimestamp(pd.read_pickle(fit_results).set_index('fips').loc[fips]['t0_date'].timestamp())
