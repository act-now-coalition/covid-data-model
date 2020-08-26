import os
import pandas as pd
import numpy as np
import logging
from pyseir import load_data
from datetime import datetime
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import AggregationLevel
from libs.datasets import combined_datasets
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

    def generate_whitelist(self) -> pd.DataFrame:
        """
        Generate a county whitelist based on the cuts above.

        Returns
        -------
        df: whitelist
        """
        logging.info("Generating county level whitelist...")

        counties = (
            combined_datasets.load_us_timeseries_dataset()
            .get_subset(aggregation_level=AggregationLevel.COUNTY)
            .set_index(CommonFields.FIPS)
        )
        df_candidates = counties.groupby(CommonFields.FIPS).apply(_whitelist_candidates_per_fips)

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
    fips = combined_data.fips.iloc[0]
    (times, observed_new_cases, observed_new_deaths,) = load_data.calculate_new_case_data_by_region(
        combined_data, t0=datetime(day=1, month=1, year=2020),
    )
    record = dict(
        fips=fips,
        state=combined_data.iat[0, combined_data.columns.get_loc(CommonFields.STATE)],
        county=combined_data.iat[0, combined_data.columns.get_loc(CommonFields.COUNTY)],
        total_cases=observed_new_cases.sum(),
        total_deaths=observed_new_deaths.sum(),
        nonzero_case_datapoints=np.sum(observed_new_cases > 0),
        nonzero_death_datapoints=np.sum(observed_new_deaths > 0),
    )
    return pd.Series(record)
