"""
Code that is used to help move information around in the pipeline, starting with `Region` which
represents a geographical area (state, county, metro area, etc).
"""

from typing import Optional, Mapping, Any
import json
import os
from dataclasses import dataclass

import pandas as pd
import structlog

import pyseir
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import combined_datasets
from pyseir.rt.utils import NEW_ORLEANS_FIPS
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

    def is_county(self):
        return len(self.fips) == 5

    def is_state(self):
        return len(self.fips) == 2

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
    def from_fips(fips: str) -> "RegionalCombinedData":
        return RegionalCombinedData(region=Region.from_fips(fips))

    def get_us_latest(self):
        """Gets latest values for a given state or county fips code."""
        us_latest = combined_datasets.load_us_latest_dataset()
        return us_latest.get_record_for_fips(self.region.fips)

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


@dataclass(frozen=True)
class RegionalWebUIInput:
    """Identifies a geographical area and wraps access to any related data read by the WebUIDataAdaptorV1."""

    region: Region

    _combined_data: RegionalCombinedData

    @staticmethod
    def from_fips(fips: str) -> "RegionalWebUIInput":
        return RegionalWebUIInput(
            region=Region.from_fips(fips), _combined_data=RegionalCombinedData.from_fips(fips)
        )

    @property
    def population(self):
        return self._combined_data.population

    @property
    def fips(self) -> str:
        return self.region.fips

    def get_us_latest(self):
        return self._combined_data.get_us_latest()

    def load_inference_result(self) -> Mapping[str, Any]:
        """
        Load fit results by state or county fips code.

        Returns
        -------
        : dict
            Dictionary of fit result information.
        """
        return load_inference_result(self.region)

    def load_ensemble_results(self) -> Optional[dict]:
        """Retrieves ensemble results for this region."""
        output_filename = self.region.run_artifact_path_to_write(
            pyseir.utils.RunArtifact.ENSEMBLE_RESULT
        )
        if not os.path.exists(output_filename):
            return None

        with open(output_filename) as f:
            return json.load(f)

    def load_rt_result(self) -> Optional[pd.DataFrame]:
        """Loads the Rt inference result.

        Returns
        -------
        results: pd.DataFrame
            DataFrame containing the R_t inferences.
        """
        if self.fips in NEW_ORLEANS_FIPS:
            _log.info("Applying New Orleans Patch")
            return pyseir.rt.patches.patch_aggregate_rt_results(NEW_ORLEANS_FIPS)

        path = self.region.run_artifact_path_to_read(pyseir.utils.RunArtifact.RT_INFERENCE_RESULT)
        if not os.path.exists(path):
            return None
        return pd.read_json(path)

    def is_county(self):
        return self.region.is_county()


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
