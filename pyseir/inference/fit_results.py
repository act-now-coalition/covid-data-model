import os
from pyseir.load_data import load_county_metadata
import simplejson
from pyseir import OUTPUT_DIR
import pandas as pd
from datetime import datetime
from pyseir.utils import get_run_artifact_path, RunArtifact


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
    fit_results = os.path.join(OUTPUT_DIR, 'pyseir', state, 'data', f'summary__{state}_imputed_start_times.pkl')
    return datetime.fromtimestamp(pd.read_pickle(fit_results).set_index('fips').loc[fips]['t0_date'].timestamp())


def convert_fips_to_str(record):
    """
    Returns the same record with a stringified fips key.
    Intended to run on a single json record

    Arguments:
        record {dict} -- {'fips': {'01': 101},
                    'R0': {'01': 3.4387109255},
                    ... }

    Returns:
        record {dict} -- {'fips': {'01': '101'},
                    'R0': {'01': 3.4387109255},
                    ... }

    """
    d = {}
    for col, vdict in record.items():
        if col=='fips':
            d.update({col: {k:str(v) for k,v in vdict.items()}})
        else:
            d.update({col:vdict})
    return d



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
    json_file = open(output_file, 'r')
    d = simplejson.load(json_file, object_hook=lambda x: convert_fips_to_str(x))
    df = pd.DataFrame.from_dict(d)
    if len(fips) == 2:
        return df.iloc[0].to_dict()
    else:
        return df.set_index('fips').loc[fips].to_dict()


def load_Rt_result(fips):
    """
    Load the Rt inference result.

    Parameters
    ----------
    fips: str
        State or County FIPS code.

    Returns
    -------
    results: pd.DataFrame
        DataFrame containing the R_t inferences.
    """
    path = get_run_artifact_path(fips, RunArtifact.RT_INFERENCE_RESULT)
    return pd.read_json(path)
