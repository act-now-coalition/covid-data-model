import numpy as np
import pandas as pd
import ujson as json
import logging
import us
from datetime import timedelta, datetime, date
from multiprocessing import Pool
from pyseir import load_data
from pyseir.inference.fit_results import load_inference_result, load_Rt_result
from pyseir.utils import get_run_artifact_path, RunArtifact, RunMode
from libs.enums import Intervention
from libs.datasets import CommonFields
from libs.datasets import FIPSPopulation, JHUDataset, CDSDataset
from libs.datasets.dataset_utils import build_aggregate_county_data_frame
from libs.datasets.dataset_utils import AggregationLevel
import libs.datasets.can_model_output_schema as schema

from typing import Tuple

log = logging.getLogger(__name__)


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
        jhu_dataset=None,
        cds_dataset=None,
        include_imputed=False,
    ):

        self.output_interval_days = output_interval_days
        self.state = state
        self.run_mode = RunMode(run_mode)
        self.include_imputed = include_imputed
        self.state_abbreviation = us.states.lookup(state).abbr
        self.population_data = FIPSPopulation.local().population()
        self.output_dir = output_dir

        self.jhu_local = jhu_dataset or JHUDataset.local()
        self.cds_dataset = cds_dataset or CDSDataset.local()

        self.county_timeseries = build_aggregate_county_data_frame(self.jhu_local, self.cds_dataset)
        self.county_timeseries["date"] = self.county_timeseries["date"].dt.normalize()

        state_timeseries = self.jhu_local.timeseries().get_subset(AggregationLevel.STATE)
        self.state_timeseries = state_timeseries.data["date"].dt.normalize()
        self.df_whitelist = load_data.load_whitelist()
        self.df_whitelist = self.df_whitelist[self.df_whitelist["inference_ok"] == True]

    @staticmethod
    def _get_county_hospitalization_from_actuals(fips, t0_simulation):
        county_hosp = load_data.get_current_hospitalized_for_county(
            fips, t0_simulation, category="hospitalized"
        )[1]
        county_icu = load_data.get_current_hospitalized_for_county(
            fips, t0_simulation, category="icu"
        )[1]

        return county_hosp, county_icu

    @staticmethod
    def _is_valid_count_metric(metric: float) -> bool:
        return metric is not None and metric > 0

    def _calculate_hospitalization_scaling_factors(
        self, fips: str, t0_simulation: datetime, pyseir_outputs
    ) -> Tuple[float, float]:
        """
        Rescale hospitalization and icu usage
        based on the population ratio...
        
        This could be swapped to use infection ratio later?
        
        """
        t_latest_hosp_data, current_hosp_count = load_data.get_current_hospitalized_for_state(
            state=self.state_abbreviation, t0=t0_simulation, category="hospitalized"
        )

        _, current_state_icu = load_data.get_current_hospitalized_for_state(
            state=self.state_abbreviation, t0=t0_simulation, category="icu",
        )

        if current_hosp_count is not None:
            state_fips = fips[:2]
            t_latest_hosp_data_date = t0_simulation + timedelta(days=int(t_latest_hosp_data))

            state_hosp_gen = load_data.get_compartment_value_on_date(
                fips=state_fips, compartment="HGen", date=t_latest_hosp_data_date
            )
            state_hosp_icu = load_data.get_compartment_value_on_date(
                fips=state_fips, compartment="HICU", date=t_latest_hosp_data_date
            )

            if len(fips) == 5:
                (
                    current_county_hosp,
                    current_county_icu,
                ) = self._get_county_hospitalization_from_actuals(fips, t0_simulation)
                log.info(
                    "Actual county hospitalizations: %s, icu: %s",
                    current_county_hosp,
                    current_county_icu,
                )
                inferred_county_hosp = load_data.get_compartment_value_on_date(
                    fips=fips,
                    compartment="HGen",
                    date=t_latest_hosp_data_date,
                    ensemble_results=pyseir_outputs,
                )
                inferred_county_icu = load_data.get_compartment_value_on_date(
                    fips=fips,
                    compartment="HICU",
                    date=t_latest_hosp_data_date,
                    ensemble_results=pyseir_outputs,
                )
                log.info(
                    "Inferred county hospitalized: %s, icu: %s",
                    inferred_county_hosp,
                    inferred_county_icu,
                )
                # Rescale the county level hospitalizations by the expected
                # ratio of county / state hospitalizations from simulations.
                # We use ICU data if available too.
                current_hosp_count *= (inferred_county_hosp + inferred_county_icu) / (
                    state_hosp_gen + state_hosp_icu
                )

            hosp_rescaling_factor = current_hosp_count / (state_hosp_gen + state_hosp_icu)

            # Some states have covidtracking issues. We shouldn't ground ICU cases
            # to zero since so far these have all been bad reporting.
            if self._is_valid_count_metric(current_state_icu):
                icu_rescaling_factor = current_state_icu / state_hosp_icu
            else:
                icu_rescaling_factor = current_hosp_count / (state_hosp_gen + state_hosp_icu)
        else:
            hosp_rescaling_factor = 1.0
            icu_rescaling_factor = 1.0

        return hosp_rescaling_factor, icu_rescaling_factor

    def _get_population(self, fips: str):
        if len(fips) == 5:
            return self.population_data.get_record_for_fips(fips)[CommonFields.POPULATION]
        return self.population_data.get_record_for_state(self.state_abbreviation)[
            CommonFields.POPULATION
        ]

    def map_fips(self, fips):
        """
        For a given county fips code, generate the CAN UI output format.

        Parameters
        ----------
        fips: str
            County FIPS code to map.
        """
        log.info("Mapping output to WebUI for %s, %s", self.state, fips)
        pyseir_outputs = load_data.load_ensemble_results(fips)

        if len(fips) == 5 and fips not in self.df_whitelist.fips.values:
            log.info("Excluding %s due to white list...", fips)
            return
        try:
            fit_results = load_inference_result(fips)
            t0_simulation = datetime.fromisoformat(fit_results["t0_date"])
        except (KeyError, ValueError):
            log.error("Fit result not found for %s. Skipping...", fips)
            return
        population = self._get_population(fips)

        (
            hosp_rescaling_factor,
            icu_rescaling_factor,
        ) = self._calculate_hospitalization_scaling_factors(fips, t0_simulation, pyseir_outputs)
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
                (t0_simulation + timedelta(days=t)).date().strftime("%m/%d/%y")
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
            log.info("Raw icu: %s", raw_model_icu_values)
            final_derived_model_value = icu_rescaling_factor * interpolated_model_icu_values
            log.info("Final icu value from model: %s", final_derived_model_value[-1])
            output_model[schema.INFECTED_C] = final_derived_model_value
            # General + ICU beds. don't include vent here because they are also counted in ICU
            output_model[schema.ALL_HOSPITALIZED] = np.add(
                output_model[schema.INFECTED_B], output_model[schema.INFECTED_C]
            )
            output_model[schema.ALL_INFECTED] = output_model[schema.INFECTED]
            output_model[schema.DEAD] = np.interp(
                t_list_downsampled, t_list, output_for_policy["total_deaths"]["ci_50"]
            )
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
                    fit_results["eps"] * fit_results["R0"] * np.ones(len(t_list)),
                )
                output_model[schema.Rt_ci90] = np.interp(
                    t_list_downsampled,
                    t_list,
                    2 * fit_results["eps_error"] * fit_results["R0"] * np.ones(len(t_list)),
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
            try:
                rt_results = load_Rt_result(fips)
                rt_results.index = rt_results["Rt_MAP_composite"].index.strftime("%m/%d/%y")
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
            except (ValueError, KeyError) as e:
                log.warning("Clearing Rt in output for fips %s", fips, exc_info=e)
                output_model[schema.RT_INDICATOR] = "NaN"
                output_model[schema.RT_INDICATOR_CI90] = "NaN"

            output_model[[schema.RT_INDICATOR, schema.RT_INDICATOR_CI90]] = output_model[
                [schema.RT_INDICATOR, schema.RT_INDICATOR_CI90]
            ].fillna("NaN")

            # Truncate floats and cast as strings to match data model.
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
                )
            ]
            output_model.loc[:, int_columns] = (
                output_model[int_columns].fillna(0).astype(int).astype(str)
            )
            output_model.loc[
                :, [schema.Rt, schema.Rt_ci90, schema.RT_INDICATOR, schema.RT_INDICATOR_CI90]
            ] = (
                output_model[
                    [schema.Rt, schema.Rt_ci90, schema.RT_INDICATOR, schema.RT_INDICATOR_CI90]
                ]
                .fillna(0)
                .round(decimals=4)
                .astype(str)
            )

            # Convert the records format to just list(list(values))
            output_model = [
                [val for val in timestep.values()]
                for timestep in output_model.to_dict(orient="records")
            ]

            output_path = get_run_artifact_path(
                fips, RunArtifact.WEB_UI_RESULT, output_dir=self.output_dir
            )
            policy_enum = Intervention.from_webui_data_adaptor(suppression_policy)
            output_path = output_path.replace("__INTERVENTION_IDX__", str(policy_enum.value))
            with open(output_path, "w") as f:
                json.dump(output_model, f)

    def generate_state(self, all_fips=[], states_only=False):
        """
        Generate for each county in a state, the output for the webUI.

        Parameters
        ----------
        states_only: bool
            If True only run the state level.
        """
        state_fips = us.states.lookup(self.state).fips
        self.map_fips(state_fips)

        if not states_only:
            all_fips = load_data.get_all_fips_codes_for_a_state(self.state)

            if not self.include_imputed:
                # Filter...
                fips_with_cases = self.jhu_local.timeseries().get_data(
                    AggregationLevel.COUNTY, country="USA", state=self.state_abbreviation
                )
                fips_with_cases = fips_with_cases[fips_with_cases.cases > 0].fips.unique().tolist()
                all_fips = [fips for fips in all_fips if fips in fips_with_cases]

            p = Pool()
            p.map(self.map_fips, all_fips)
            p.close()
            p.join()


if __name__ == "__main__":
    mapper = WebUIDataAdaptorV1("California", output_interval_days=4)
    mapper.generate_state()
