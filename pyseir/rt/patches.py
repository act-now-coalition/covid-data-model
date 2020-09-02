import pandas as pd
import numpy as np

from libs.datasets import combined_datasets
from libs.datasets.combined_datasets import CommonFields
import pyseir.utils


def patch_aggregate_rt_results(fips_superset: list) -> pd.DataFrame:
    """Return the population weighted rt dataframe results for the given fips_superset

    Parameters
    ----------
    fips_superset
        A list of fips to combine for rt calculation

    Returns
    -------
    dataframe
        With columns "Rt_MAP_composite" and "Rt_ci95_composite"
    """

    def load_and_append_population(fips: str) -> pd.DataFrame:
        path = pyseir.utils.get_run_artifact_path(
            fips, pyseir.utils.RunArtifact.RT_INFERENCE_RESULT
        )
        tmp = pd.read_json(path)
        tmp["population"] = combined_datasets.get_us_latest_for_fips(fips)[CommonFields.POPULATION]
        return tmp

    combined_df = pd.concat([load_and_append_population(x) for x in fips_superset])

    def f(x):
        """Helper function to apply to weighted arithmetic means to groupby objects"""
        # TODO: Decide whether Rt_ci95_composite should be changed to being combined in quadrature
        #  http://ipl.physics.harvard.edu/wp-uploads/2013/03/PS3_Error_Propagation_sp13.pdf instead
        #  of the population weighted arithmetic mean

        weighted_data = dict(
            Rt_MAP_composite=np.average(x["Rt_MAP_composite"], weights=x["population"]),
            Rt_ci95_composite=np.average(x["Rt_ci95_composite"], weights=x["population"]),
            date=x.name,
        )

        return pd.Series(weighted_data)

    return combined_df.groupby(CommonFields.DATE).apply(f)
