import json
import os
from dataclasses import dataclass
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Optional

import structlog
from datetime import timedelta, datetime
import numpy as np
import pandas as pd
from multiprocessing import Pool

import pyseir
from libs import pipeline
from libs.pipeline import Region
from libs.pipeline import Region
from libs.pipeline import RegionalCombinedData
from libs.pipeline import RegionalCombinedData
from libs.pipeline import _log
from libs.pipeline import load_inference_result
from pyseir.deployment import model_to_observed_shim as shim
from pyseir.icu import infer_icu
from pyseir.inference import model_fitter
from pyseir.models import seir_model
from pyseir.rt.utils import NEW_ORLEANS_FIPS
from pyseir.rt.utils import NEW_ORLEANS_FIPS
from pyseir.utils import get_run_artifact_path, RunArtifact, RunMode
from libs.enums import Intervention
from libs.datasets import CommonFields
import libs.datasets.can_model_output_schema as schema


log = structlog.get_logger()

# Value of orient argument in pandas dataframe json output.
OUTPUT_JSON_ORIENT = "split"


@dataclass(frozen=True)
class RegionalInput:
    """Identifies a geographical area and wraps access to any related data read by the WebUIDataAdaptorV1."""

    region: Region

    _combined_data: RegionalCombinedData
    _mle_fit_result: Optional[Mapping[str, Any]] = None

    @staticmethod
    def from_region(region: Region) -> "RegionalInput":
        return RegionalInput(region=region, _combined_data=RegionalCombinedData.from_region(region))

    @staticmethod
    def from_model_fitter(fitter: model_fitter.ModelFitter) -> "RegionalInput":
        return RegionalInput(
            region=fitter.region,
            _combined_data=fitter.regional_input._combined_data,
            _mle_fit_result=fitter.fit_results,
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
        if self._mle_fit_result is not None:
            return self._mle_fit_result
        return pipeline.load_inference_result(self.region)

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


class WebUIDataAdaptorV1:
    """
    Map pyseir output data model to the format required by CovidActNow's webUI.

    Parameters
    ----------
    include_imputed:
        If True, map the outputs for imputed counties as well as those with
        no data.
    """

    def __init__(
        self, output_interval_days=4, run_mode="can-before", output_dir=None, include_imputed=False,
    ):
        self.output_interval_days = output_interval_days
        self.run_mode = RunMode(run_mode)
        self.include_imputed = include_imputed
        self.output_dir = output_dir

    def map_fips(self, regional_input: RegionalInput) -> None:
        """Generates the CAN UI output format for a given region.

        Args:
            regional_input: the region and its data
        """
        # Get the latest observed values to use in calculating shims
        observed_latest_dict = regional_input.get_us_latest()

        state = observed_latest_dict[CommonFields.STATE]
        log.info("Mapping output to WebUI.", state=state, fips=regional_input.fips)
        shim_log = structlog.getLogger(fips=regional_input.fips)
        pyseir_outputs = regional_input.load_ensemble_results()

        try:
            fit_results = regional_input.load_inference_result()
            t0_simulation = datetime.fromisoformat(fit_results["t0_date"])
        except (KeyError, ValueError):
            log.error("Fit result not found for fips. Skipping...", fips=regional_input.fips)
            return
        population = regional_input.population

        # We will shim all suppression policies by the same amount (since historical tracking error
        # for all policies is the same).
        baseline_policy = "suppression_policy__inferred"  # This could be any valid policy

        # We need the index in the model's temporal frame.
        idx_offset = int(fit_results["t_today"] - fit_results["t0"])

        observed_death_latest = observed_latest_dict[CommonFields.DEATHS]
        observed_total_hosps_latest = observed_latest_dict[CommonFields.CURRENT_HOSPITALIZED]
        observed_icu_latest = observed_latest_dict[CommonFields.CURRENT_ICU]

        # For Deaths
        model_death_latest = pyseir_outputs[baseline_policy]["total_deaths"]["ci_50"][idx_offset]
        model_acute_latest = pyseir_outputs[baseline_policy]["HGen"]["ci_50"][idx_offset]
        model_icu_latest = pyseir_outputs[baseline_policy]["HICU"]["ci_50"][idx_offset]
        model_total_hosps_latest = model_acute_latest + model_icu_latest

        death_shim = shim.calculate_strict_shim(
            model=model_death_latest,
            observed=observed_death_latest,
            log=shim_log.bind(type=CommonFields.DEATHS),
        )

        total_hosp_shim = shim.calculate_strict_shim(
            model=model_total_hosps_latest,
            observed=observed_total_hosps_latest,
            log=shim_log.bind(type=CommonFields.CURRENT_HOSPITALIZED),
        )

        # For ICU This one is a little more interesting since we often don't have ICU. In this case
        # we use information from the same aggregation level (intralevel) to keep the ratios
        # between general hospitalization and icu hospitalization
        icu_shim = shim.calculate_intralevel_icu_shim(
            model_acute=model_acute_latest,
            model_icu=model_icu_latest,
            observed_icu=observed_icu_latest,
            observed_total_hosps=observed_total_hosps_latest,
            log=shim_log.bind(type=CommonFields.CURRENT_ICU),
        )
        # ICU PATCH
        icu_patch_ts = infer_icu.get_icu_timeseries(
            fips=regional_input.fips, weight_by=infer_icu.ICUWeightsPath.ONE_MONTH_TRAILING_CASES
        )

        # Iterate through each suppression policy.
        # Model output is interpolated to the dates desired for the API.
        suppression_policies = [
            key for key in pyseir_outputs.keys() if key.startswith("suppression_policy")
        ]
        for suppression_policy in suppression_policies:
            output_for_policy = pyseir_outputs[suppression_policy]
            output_model = pd.DataFrame()
            t_list = output_for_policy["t_list"]
            t_list_downsampled = range(0, int(max(t_list)), self.output_interval_days)

            output_model[schema.DAY_NUM] = t_list_downsampled
            output_model[schema.DATE] = [
                (t0_simulation + timedelta(days=t)).date().strftime("%Y-%m-%d")
                for t in t_list_downsampled
            ]
            output_model[schema.TOTAL] = population
            output_model[schema.TOTAL_SUSCEPTIBLE] = np.interp(
                t_list_downsampled, t_list, output_for_policy["S"]["ci_50"]
            )
            output_model[schema.EXPOSED] = np.interp(
                t_list_downsampled, t_list, output_for_policy["E"]["ci_50"]
            )
            output_model[schema.INFECTED] = np.interp(
                t_list_downsampled,
                t_list,
                np.add(output_for_policy["I"]["ci_50"], output_for_policy["A"]["ci_50"]),
            )  # Infected + Asympt.
            output_model[schema.INFECTED_A] = output_model[schema.INFECTED]

            interpolated_model_acute_values = np.interp(
                t_list_downsampled, t_list, output_for_policy["HGen"]["ci_50"]
            )
            output_model[schema.INFECTED_B] = interpolated_model_acute_values

            raw_model_icu_values = output_for_policy["HICU"]["ci_50"]
            interpolated_model_icu_values = np.interp(
                t_list_downsampled, t_list, raw_model_icu_values
            )

            # 21 August 2020: The line assigning schema.INFECTED_C in the output_model is
            # commented out while the Linear Regression estimator is patched through this pipeline
            # to be consumed downstream by the ICU utilization calculations. It is left here as a
            # marker for the future if the ICU utilization calculations is dis-entangled from the
            # PySEIR model outputs.

            # output_model[schema.INFECTED_C] = (icu_shim+interpolated_model_icu_values).clip(min=0)

            # Applying Patch for ICU Linear Regression
            infer_icu_patch = icu_patch_ts.reindex(
                [pd.Timestamp(x) for x in output_model[schema.DATE]]
            )
            output_model[schema.INFECTED_C] = infer_icu_patch.to_numpy()

            # General + ICU beds. don't include vent here because they are also counted in ICU
            output_model[schema.ALL_HOSPITALIZED] = (
                interpolated_model_acute_values + interpolated_model_icu_values + total_hosp_shim
            ).clip(min=0)

            output_model[schema.ALL_INFECTED] = output_model[schema.INFECTED]

            # Shim Deaths to Match Observed
            raw_model_deaths_values = output_for_policy["total_deaths"]["ci_50"]
            interp_model_deaths_values = np.interp(
                t_list_downsampled, t_list, raw_model_deaths_values
            )
            output_model[schema.DEAD] = (interp_model_deaths_values + death_shim).clip(min=0)

            # Continue mapping
            final_beds = np.mean(output_for_policy["HGen"]["capacity"])
            output_model[schema.BEDS] = final_beds
            output_model[schema.CUMULATIVE_INFECTED] = np.interp(
                t_list_downsampled,
                t_list,
                np.cumsum(output_for_policy["total_new_infections"]["ci_50"]),
            )

            if fit_results:
                output_model[schema.Rt] = np.interp(
                    t_list_downsampled,
                    t_list,
                    fit_results["eps2"] * fit_results["R0"] * np.ones(len(t_list)),
                )
                output_model[schema.Rt_ci90] = np.interp(
                    t_list_downsampled,
                    t_list,
                    2 * fit_results["eps2_error"] * fit_results["R0"] * np.ones(len(t_list)),
                )
            else:
                output_model[schema.Rt] = 0
                output_model[schema.Rt_ci90] = 0

            output_model[schema.CURRENT_VENTILATED] = (
                icu_shim
                + np.interp(t_list_downsampled, t_list, output_for_policy["HVent"]["ci_50"])
            ).clip(min=0)
            output_model[schema.POPULATION] = population
            # Average capacity.
            output_model[schema.ICU_BED_CAPACITY] = np.mean(output_for_policy["HICU"]["capacity"])
            output_model[schema.VENTILATOR_CAPACITY] = np.mean(
                output_for_policy["HVent"]["capacity"]
            )

            # Truncate date range of output.
            output_dates = pd.to_datetime(output_model["date"])
            output_model = output_model[
                (output_dates >= datetime(month=3, day=3, year=2020))
                & (output_dates < datetime.today() + timedelta(days=90))
            ]
            output_model = output_model.fillna(0)

            # Fill in results for the Rt indicator.
            rt_results = regional_input.load_rt_result()
            if rt_results is not None:
                rt_results.index = rt_results["Rt_MAP_composite"].index.strftime("%Y-%m-%d")
                merged = output_model.merge(
                    rt_results[["Rt_MAP_composite", "Rt_ci95_composite"]],
                    right_index=True,
                    left_on="date",
                    how="left",
                )
                output_model[schema.RT_INDICATOR] = merged["Rt_MAP_composite"]

                # With 90% probability the value is between rt_indicator - ci90
                # to rt_indicator + ci90
                output_model[schema.RT_INDICATOR_CI90] = (
                    merged["Rt_ci95_composite"] - merged["Rt_MAP_composite"]
                )
            else:
                log.warning(
                    "No Rt Results found, clearing Rt in output.",
                    fips=regional_input.fips,
                    suppression_policy=suppression_policy,
                )
                output_model[schema.RT_INDICATOR] = "NaN"
                output_model[schema.RT_INDICATOR_CI90] = "NaN"

            output_model[[schema.RT_INDICATOR, schema.RT_INDICATOR_CI90]] = output_model[
                [schema.RT_INDICATOR, schema.RT_INDICATOR_CI90]
            ].fillna("NaN")

            int_columns = [
                col
                for col in output_model.columns
                if col
                not in (
                    schema.DATE,
                    schema.Rt,
                    schema.Rt_ci90,
                    schema.RT_INDICATOR,
                    schema.RT_INDICATOR_CI90,
                    schema.FIPS,
                )
            ]
            output_model.loc[:, int_columns] = output_model[int_columns].fillna(0).astype(int)
            output_model.loc[
                :, [schema.Rt, schema.Rt_ci90, schema.RT_INDICATOR, schema.RT_INDICATOR_CI90]
            ] = output_model[
                [schema.Rt, schema.Rt_ci90, schema.RT_INDICATOR, schema.RT_INDICATOR_CI90]
            ].fillna(
                0
            )

            output_model[schema.FIPS] = regional_input.fips
            intervention = Intervention.from_webui_data_adaptor(suppression_policy)
            output_model[schema.INTERVENTION] = intervention.value
            output_path = get_run_artifact_path(
                regional_input.fips, RunArtifact.WEB_UI_RESULT, output_dir=self.output_dir
            )
            output_path = output_path.replace("__INTERVENTION_IDX__", str(intervention.value))
            output_model.to_json(output_path, orient=OUTPUT_JSON_ORIENT)

    def generate_state(
        self, regional_input: RegionalInput, whitelisted_county_fips: list, states_only=False,
    ):
        """
        Generate the output for the webUI for the given state, and counties in that state if
        states_only=False.

        Parameters
        ----------
        regional_input: RegionalInput
            Input for a state
        whitelisted_county_fips:
            Counties in the state
        states_only: bool
            If True only run the state level.
        """

        self.map_fips(regional_input)

        if states_only:
            return
        else:
            with Pool(maxtasksperchild=1) as p:
                regions = map(pipeline.Region.from_fips, whitelisted_county_fips)
                regional_inputs = map(RegionalInput.from_region, regions)
                p.map(self.map_fips, regional_inputs)

            return


if __name__ == "__main__":
    # Need to have a whitelist pre-generated
    # Need to have state output already built
    mapper = WebUIDataAdaptorV1(output_interval_days=1, run_mode="can-inference-derived")
    mapper.generate_state("TX", whitelisted_county_fips=["48201"], states_only=False)
