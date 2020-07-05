import ujson as json
import structlog
import us
from typing import Tuple
from datetime import timedelta, datetime
import numpy as np
import pandas as pd
from multiprocessing import Pool
from pyseir import load_data
from pyseir.deployment import model_to_observed_shim as shim
from pyseir.inference.fit_results import load_inference_result, load_Rt_result
from pyseir.utils import get_run_artifact_path, RunArtifact, RunMode
from libs.enums import Intervention
from libs.datasets import CommonFields
from libs.datasets import FIPSPopulation, combined_datasets, dataset_cache
import libs.datasets.can_model_output_schema as schema


log = structlog.get_logger()


class WebUIDataAdaptorV1:
    """
    Map pyseir output data model to the format required by CovidActNow's webUI.

    Parameters
    ----------
    state: str
        State to map outputs for.
    include_imputed:
        If True, map the outputs for imputed counties as well as those with
        no data.
    """

    def __init__(
        self,
        state,
        output_interval_days=4,
        run_mode="can-before",
        output_dir=None,
        include_imputed=False,
    ):

        self.output_interval_days = output_interval_days
        self.state = state
        self.run_mode = RunMode(run_mode)
        self.include_imputed = include_imputed
        self.state_abbreviation = us.states.lookup(state).abbr
        self.population_data = FIPSPopulation.local().population()
        self.output_dir = output_dir

    @staticmethod
    def _get_county_hospitalization(fips: str, t0_simulation: datetime) -> Tuple[float, float]:
        """
        Fetches the latest county hospitalization and icu utilization.

        If current data is available, we return that.
        If not, current values are estimated from cumulative.
        """
        _, county_hosp = load_data.get_current_hospitalized(
            fips, t0_simulation, category=load_data.HospitalizationCategory.HOSPITALIZED,
        )
        _, county_icu = load_data.get_current_hospitalized(
            fips, t0_simulation, category=load_data.HospitalizationCategory.ICU
        )

        return county_hosp, county_icu

    @staticmethod
    def _is_valid_count_metric(metric: float) -> bool:
        return metric is not None and metric > 0

    def _get_population(self, fips: str) -> int:
        if len(fips) == 5:
            return self.population_data.get_record_for_fips(fips)[CommonFields.POPULATION]
        return self.population_data.get_record_for_state(self.state_abbreviation)[
            CommonFields.POPULATION
        ]

    def _get_model_to_dataset_conversion_factors(self, t0_simulation, fips, pyseir_outputs):
        """
        Return scaling factors to convert model hospitalization and model icu numbers to match
        the most current values provided in combined_datasets.

        Parameters
        ----------
        t0_simulation
        fips
        pyseir_outputs

        Returns
        -------
        hosp_rescaling_factor
        icu_rescaling_factor
        """
        state_fips = fips[:2]

        t_latest_hosp_data, current_hosp_count = load_data.get_current_hospitalized(
            fips=state_fips,
            t0=t0_simulation,
            category=load_data.HospitalizationCategory.HOSPITALIZED,
        )

        _, current_state_icu = load_data.get_current_hospitalized(
            fips=state_fips, t0=t0_simulation, category=load_data.HospitalizationCategory.ICU,
        )

        if current_hosp_count is not None:
            t_latest_hosp_data_date = t0_simulation + timedelta(days=int(t_latest_hosp_data))

            state_hosp_gen = load_data.get_compartment_value_on_date(
                fips=state_fips, compartment="HGen", date=t_latest_hosp_data_date
            )
            state_hosp_icu = load_data.get_compartment_value_on_date(
                fips=state_fips, compartment="HICU", date=t_latest_hosp_data_date
            )

            if len(fips) == 5:
                (current_county_hosp, current_county_icu,) = self._get_county_hospitalization(
                    fips, t0_simulation
                )
                log.info(
                    "Actual county hospitalizations",
                    fips=fips,
                    hospitalized=current_county_hosp,
                    icu=current_county_icu,
                )
                inferred_county_hosp = load_data.get_compartment_value_on_date(
                    fips=fips,
                    compartment="HGen",
                    date=t_latest_hosp_data_date,  # this could be off by a day from the hosp data
                    ensemble_results=pyseir_outputs,
                )

                county_hosp = inferred_county_hosp

                inferred_county_icu = load_data.get_compartment_value_on_date(
                    fips=fips,
                    compartment="HICU",
                    date=t_latest_hosp_data_date,
                    ensemble_results=pyseir_outputs,
                )
                log.info(
                    "Inferred county hospitalized for fips.",
                    fips=fips,
                    hospitalized=inferred_county_hosp,
                    icu=inferred_county_icu,
                )
                county_icu = inferred_county_icu
                if self._is_valid_count_metric(current_county_hosp):
                    # use actual instead of adjusted
                    county_hosp = current_county_hosp

                if self._is_valid_count_metric(current_county_icu):
                    county_icu = current_county_icu

                # Rescale the county level hospitalizations by the expected
                # ratio of county / state hospitalizations from simulations.
                # We use ICU data if available too.
                current_hosp_count *= (county_hosp + county_icu) / (state_hosp_gen + state_hosp_icu)

            hosp_rescaling_factor = current_hosp_count / (state_hosp_gen + state_hosp_icu)

            # Some states have covidtracking issues. We shouldn't ground ICU cases
            # to zero since so far these have all been bad reporting.
            if len(fips) == 5 and self._is_valid_count_metric(current_county_icu):
                icu_rescaling_factor = current_county_icu / inferred_county_icu
            elif self._is_valid_count_metric(current_state_icu):
                icu_rescaling_factor = current_state_icu / state_hosp_icu
            else:
                icu_rescaling_factor = current_hosp_count / (state_hosp_gen + state_hosp_icu)
        else:
            hosp_rescaling_factor = 1.0
            icu_rescaling_factor = 1.0
        return hosp_rescaling_factor, icu_rescaling_factor

    def map_fips(self, fips: str) -> None:
        """
        For a given fips code, for either a county or state, generate the CAN UI output format.

        Parameters
        ----------
        fips: str
            FIPS code to map.
        """
        log.info("Mapping output to WebUI.", state=self.state, fips=fips)
        pyseir_outputs = load_data.load_ensemble_results(fips)

        try:
            fit_results = load_inference_result(fips)
            t0_simulation = datetime.fromisoformat(fit_results["t0_date"])
        except (KeyError, ValueError):
            log.error("Fit result not found for fips. Skipping...", fips=fips)
            return
        population = self._get_population(fips)

        # Get multiplicative conversion factors to scale model output to fit dataset current values
        hosp_rescaling_factor, icu_rescaling_factor = self._get_model_to_dataset_conversion_factors(
            t0_simulation=t0_simulation, fips=fips, pyseir_outputs=pyseir_outputs,
        )

        # We will shim all suppression policies by the same amount (since historical tracking error
        # for all policies is the same).
        baseline_policy = "suppression_policy__inferred"  # This could be any valid policy
        model_death_ts = pyseir_outputs[baseline_policy]["total_deaths"]["ci_50"]
        # We need the index in the model's temporal frame.
        idx_offset = int(fit_results["t_today"] - fit_results["t0"])
        # Get the latest observed values to shim to.
        observed_latest_dict = combined_datasets.get_us_latest_for_fips(fips)
        shim_log = structlog.getLogger(fips=fips)

        death_shim = shim.shim_deaths_model_to_observations(
            model_death_ts=model_death_ts,
            idx=idx_offset,
            observed_latest=observed_latest_dict,
            log=shim_log,
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
            output_model[schema.INFECTED_B] = hosp_rescaling_factor * np.interp(
                t_list_downsampled, t_list, output_for_policy["HGen"]["ci_50"]
            )  # Hosp General

            raw_model_icu_values = output_for_policy["HICU"]["ci_50"]
            interpolated_model_icu_values = np.interp(
                t_list_downsampled, t_list, raw_model_icu_values
            )
            final_derived_model_value = icu_rescaling_factor * interpolated_model_icu_values
            output_model[schema.INFECTED_C] = final_derived_model_value
            # General + ICU beds. don't include vent here because they are also counted in ICU
            output_model[schema.ALL_HOSPITALIZED] = np.add(
                output_model[schema.INFECTED_B], output_model[schema.INFECTED_C]
            )
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

            output_model[schema.CURRENT_VENTILATED] = icu_rescaling_factor * np.interp(
                t_list_downsampled, t_list, output_for_policy["HVent"]["ci_50"]
            )
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
            rt_results = load_Rt_result(fips)
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
                    fips=fips,
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

            output_model[schema.FIPS] = fips
            intervention = Intervention.from_webui_data_adaptor(suppression_policy)
            output_model[schema.INTERVENTION] = intervention.value
            output_path = get_run_artifact_path(
                fips, RunArtifact.WEB_UI_RESULT, output_dir=self.output_dir
            )
            output_path = output_path.replace("__INTERVENTION_IDX__", str(intervention.value))
            output_model.to_json(output_path)

    def generate_state(self, whitelisted_county_fips: list, states_only=False):
        """
        Generate the output for the webUI for the given state, and counties in that state if
        states_only=False.

        Parameters
        ----------
        whitelisted_county_fips
        states_only: bool
            If True only run the state level.
        """

        state_fips = us.states.lookup(self.state).fips
        self.map_fips(state_fips)

        if states_only:
            return
        else:
            with Pool(maxtasksperchild=1) as p:
                p.map(self.map_fips, whitelisted_county_fips)

            return


if __name__ == "__main__":
    dataset_cache.set_pickle_cache_dir()
    # Need to have a whitelist pre-generated
    # Need to have state output already built
    mapper = WebUIDataAdaptorV1(
        state="California", output_interval_days=1, run_mode="can-inference-derived"
    )
    mapper.generate_state(whitelisted_county_fips=["06037", "06075", "06059"], states_only=False)
