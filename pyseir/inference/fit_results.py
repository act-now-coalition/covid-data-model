import os
from libs import us_state_abbrev
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import combined_datasets
from pyseir import OUTPUT_DIR
import pandas as pd
from datetime import datetime
from pyseir.utils import get_run_artifact_path, RunArtifact


def load_t0(fips):
    """
    Load the simulation start time by county from fit_results

    Parameters
    ----------
    fips: str
        County FIPS

    Returns
    -------
    : datetime
        t0(C=1) cases.
    """
    fips_data = combined_datasets.get_us_latest_for_fips(fips)
    state = fips_data[CommonFields.STATE]
    state_full_name = us_state_abbrev.ABBREV_US_STATE[state]

    fit_results = os.path.join(
        OUTPUT_DIR, "pyseir", state, "data", f"summary__{state_full_name}_imputed_start_times.pkl"
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
        State or County FIPS code.

    Returns
    -------
    : dict
        Dictionary of fit result information.
    """
    output_file = get_run_artifact_path(fips, RunArtifact.MLE_FIT_RESULT)
    df = pd.read_json(output_file, dtype={"fips": "str"})
    if len(fips) == 2:
        return df.iloc[0].to_dict()
    else:
        return df.set_index("fips").loc[fips].to_dict()
