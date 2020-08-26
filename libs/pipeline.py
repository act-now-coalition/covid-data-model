"""
Code that is used to help move information around in the pipeline, starting with `Region` which
represents a geographical area (state, county, metro area, etc).
"""

from dataclasses import dataclass
from typing import Mapping, Any

import pandas as pd
import structlog
import us

import pyseir
import pyseir.rt.patches
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import combined_datasets
from libs.datasets import timeseries
from pyseir.utils import RunArtifact

_log = structlog.get_logger()


@dataclass(frozen=True)
class Region:
    """Identifies a geographical area."""

    # The FIPS identifier for the region, either 2 digits for a state or 5 digits for a county.
    # TODO(tom): Add support for regions other than states and counties.
    fips: str

    @staticmethod
    def from_fips(fips: str) -> "Region":
        return Region(fips=fips)

    @staticmethod
    def from_state(state: str) -> "Region":
        """Creates a Region object from a state abbreviation, name or 2 digit FIPS code."""
        state_obj = us.states.lookup(state)
        fips = state_obj.fips
        return Region(fips=fips)

    def is_county(self):
        return len(self.fips) == 5

    def is_state(self):
        return len(self.fips) == 2

    def state_obj(self):
        if self.is_state():
            return us.states.lookup(self.fips)
        elif self.is_county():
            return us.states.lookup(self.fips[:2])
        else:
            raise ValueError(f"No state_obj for {self}")

    def get_state_region(self) -> "Region":
        """Returns a Region object for the state of a county, otherwise raises a ValueError."""
        if len(self.fips) != 5:
            raise ValueError(f"No state for {self}")
        return Region(fips=self.fips[:2])

    def run_artifact_path_to_read(self, run_artifact: pyseir.utils.RunArtifact) -> str:
        """Returns the path of given artifact, to be used for reading.

        Call this function instead of directly passing a fips to get_run_artifact_path to reduce
        the amount of code that handles a fips. `run_artifact_path_to_write` has identical
        behavior but using the appropriate function helps track down inputs and outputs.
        """
        return pyseir.utils.get_run_artifact_path(self.fips, run_artifact)

    def run_artifact_path_to_write(self, run_artifact: pyseir.utils.RunArtifact) -> str:
        """Returns the path of given artifact, to be used for reading.

        Call this function instead of directly passing a fips to get_run_artifact_path to reduce
        the amount of code that handles a fips. `run_artifact_path_to_read` has identical
        behavior but using the appropriate function helps track down inputs and outputs.
        """
        return pyseir.utils.get_run_artifact_path(self.fips, run_artifact)


@dataclass(frozen=True)
class RegionalCombinedData:
    """Identifies a geographical area and wraps access to `combined_datasets` of it."""

    region: Region

    @staticmethod
    def from_region(region: Region) -> "RegionalCombinedData":
        return RegionalCombinedData(region=region)

    def get_us_latest(self):
        """Gets latest values for a given state or county fips code."""
        us_latest = combined_datasets.load_us_latest_dataset()
        return us_latest.get_record_for_fips(self.region.fips)

    def get_timeseries(self) -> timeseries.TimeseriesDataset:
        """Gets latest values for a given state or county fips code."""
        us_latest = combined_datasets.load_us_timeseries_dataset()
        return us_latest.get_subset(fips=self.region.fips)

    @property
    def population(self) -> int:
        """Gets the population for this region."""
        return self.get_us_latest()[CommonFields.POPULATION]

    @property  # TODO(tom): Change to cached_property when we're using Python 3.8
    def display_name(self) -> str:
        record = self.get_us_latest()
        county = record[CommonFields.COUNTY]
        state = record[CommonFields.STATE]
        if county:
            return f"{county}, {state}"
        return state


def load_inference_result(region: Region) -> Mapping[str, Any]:
    """
    Load fit results by state or county fips code.

    Returns
    -------
    : dict
        Dictionary of fit result information.
    """
    output_file = region.run_artifact_path_to_read(RunArtifact.MLE_FIT_RESULT)
    df = pd.read_json(output_file, dtype={"fips": "str"})
    if region.is_state():
        return df.iloc[0].to_dict()
    else:
        return df.set_index("fips").loc[region.fips].to_dict()
