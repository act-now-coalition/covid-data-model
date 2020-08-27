import os
from typing import List

import pandas as pd
import numpy as np
import logging

from libs import pipeline

from libs.datasets.timeseries import TimeseriesDataset
from pyseir import load_data
from datetime import datetime
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import AggregationLevel
from pyseir.utils import get_run_artifact_path, RunArtifact
from pandarallel import pandarallel

VISIBIBLE_PROGRESS_BAR = os.environ.get("PYSEIR_VERBOSITY") == "True"
pandarallel.initialize(progress_bar=VISIBIBLE_PROGRESS_BAR)


class WhitelistGenerator:
    """
    This class applies filters to inference results to determine a set of
    Counties to whitelist in the API.

    Parameters
    ----------
    total_cases: int
        Min number of total cases to whitelist.
    total_deaths: int
        Min number of total cases to whitelist.
    nonzero_case_datapoints: int
        Nonzero case datapoints.
    nonzero_death_datapoints: int
        Minimum number of nonzero death datapoints in the time series to allow
        display.
    """

    def __init__(
        self, total_cases=50, total_deaths=0, nonzero_case_datapoints=5, nonzero_death_datapoints=0
    ):
        self.df_whitelist = None

        self.total_cases = total_cases
        self.total_deaths = total_deaths
        self.nonzero_case_datapoints = nonzero_case_datapoints
        self.nonzero_death_datapoints = nonzero_death_datapoints

    def generate_whitelist(self, timeseries: TimeseriesDataset) -> pd.DataFrame:
        """
        Generate a county whitelist based on the cuts above.

        Returns
        -------
        df: whitelist
        """
        logging.info("Generating county level whitelist...")

        counties = timeseries.get_data(aggregation_level=AggregationLevel.COUNTY).set_index(
            CommonFields.FIPS
        )
        df_candidates = (
            counties.groupby(CommonFields.FIPS)
            .apply(_whitelist_candidates_per_fips)
            .reset_index(drop=True)
        )

        df_candidates["inference_ok"] = (
            (df_candidates.nonzero_case_datapoints >= self.nonzero_case_datapoints)
            & (df_candidates.nonzero_death_datapoints >= self.nonzero_death_datapoints)
            & (df_candidates.total_cases >= self.total_cases)
            & (df_candidates.total_deaths >= self.total_deaths)
        )
        output_path = get_run_artifact_path(
            fips="06", artifact=RunArtifact.WHITELIST_RESULT  # Dummy fips since not used here...
        )
        df_whitelist = df_candidates[["fips", "state", "county", "inference_ok"]]
        df_whitelist.to_json(output_path)

        return df_whitelist


def _whitelist_candidates_per_fips(combined_data: pd.DataFrame):
    assert not combined_data.empty
    fips = combined_data.name
    (times, observed_new_cases, observed_new_deaths,) = load_data.calculate_new_case_data_by_region(
        TimeseriesDataset(combined_data.reset_index()), t0=datetime(day=1, month=1, year=2020),
    )
    record = dict(
        fips=fips,
        # Get the state and county values from the first row of the dataframe.
        state=combined_data.iat[0, combined_data.columns.get_loc(CommonFields.STATE)],
        county=combined_data.iat[0, combined_data.columns.get_loc(CommonFields.COUNTY)],
        total_cases=observed_new_cases.sum(),
        total_deaths=observed_new_deaths.sum(),
        nonzero_case_datapoints=np.sum(observed_new_cases[~np.isnan(observed_new_cases)] > 0),
        nonzero_death_datapoints=np.sum(observed_new_deaths[~np.isnan(observed_new_deaths)] > 0),
    )
    return pd.Series(record)


def regions_in_states(
    states: List[pipeline.Region], whitelist_df: pd.DataFrame, fips: str = None,
) -> List[pipeline.Region]:
    """Finds all whitelisted regions in a list of states.

    Args:
        states: List of states to run on.
        fips: Optional county fips code to restrict results to.
        whitelist_df: A whitelist used to filter counties

    Returns: List of counties in all states, represented as `Region` objects.
    """
    states_values = [r.state_obj().abbr for r in states]
    fips_in_states = whitelist_df.loc[
        whitelist_df["inference_ok"] & whitelist_df[CommonFields.STATE].isin(states_values),
        CommonFields.FIPS,
    ].to_list()
    if fips:
        if fips in fips_in_states:
            return [pipeline.Region.from_fips(fips)]
        else:
            print(f"{fips} not in {fips_in_states}")
            return []
    else:
        return [pipeline.Region.from_fips(f) for f in fips_in_states]
