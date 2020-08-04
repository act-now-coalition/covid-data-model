from pyseir.utils import get_run_artifact_path, RunArtifact


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
