"""
Code that is used to help move information around in the pipeline, starting with `Region` which
represents a geographical area (state, county, metro area, etc).
"""


import json
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import structlog
from pydantic import BaseModel

import pyseir
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import FIPSPopulation
from libs.datasets import combined_datasets
from pyseir.rt.utils import NEW_ORLEANS_FIPS
from pyseir.utils import get_run_artifact_path, RunArtifact


overwrite_params_df = pd.read_csv(
    "./pyseir_data/pyseir_fitter_initial_conditions.csv", dtype={"fips": str}
).set_index("fips")


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

    def load_inference_result(self):
        """
        Load fit results by state or county fips code.

        Returns
        -------
        : dict
            Dictionary of fit result information.
        """
        output_file = get_run_artifact_path(self.fips, RunArtifact.MLE_FIT_RESULT)
        df = pd.read_json(output_file, dtype={"fips": "str"})
        if len(self.fips) == 2:
            return df.iloc[0].to_dict()
        else:
            return df.set_index("fips").loc[self.fips].to_dict()

    def load_ensemble_results(self) -> Optional[dict]:
        """Retrieves ensemble results for this region."""
        output_filename = pyseir.utils.get_run_artifact_path(
            self.fips, pyseir.utils.RunArtifact.ENSEMBLE_RESULT
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

        path = pyseir.utils.get_run_artifact_path(
            self.fips, pyseir.utils.RunArtifact.RT_INFERENCE_RESULT
        )
        if not os.path.exists(path):
            return None
        return pd.read_json(path)

    def load_new_case_data_by_fips(self, t0):
        return load_data.load_new_case_data_by_fips(self.fips, t0)

    def load_hospitalization_data(
        self, t0: datetime, category: HospitalizationCategory = HospitalizationCategory.HOSPITALIZED
    ):
        return load_data.load_hospitalization_data(self.fips, t0, category=category)

    def is_county(self):
        return len(self.fips) == 5

    def load_inference_result_of_state(self):
        if not self.is_county():
            raise AssertionError(f"Attempt to find state of {self}")
        return Region.from_fips(self.fips[:2]).load_inference_result()

    def get_pyseir_fitter_initial_conditions(self, params: List[str]) -> Mapping[str, Any]:
        if self.fips in overwrite_params_df.index:
            return overwrite_params_df.loc[self.fips, params].to_dict()
        else:
            return {}
