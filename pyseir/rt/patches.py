from typing import Mapping

import pandas as pd
import numpy as np

from libs import pipeline
from libs.datasets.combined_datasets import CommonFields


def patch_aggregate_rt_results(
    infection_rate_map: Mapping[pipeline.Region, pd.DataFrame],
    population_map: Mapping[pipeline.Region, float],
) -> pd.DataFrame:
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
    assert set(infection_rate_map.keys()) == set(population_map.keys())

    def load_and_append_population(region: pipeline.Region) -> pd.DataFrame:
        tmp = infection_rate_map[region]
        tmp["population"] = population_map[region]
        return tmp

    combined_df = pd.concat(load_and_append_population(x) for x in infection_rate_map.keys())

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
