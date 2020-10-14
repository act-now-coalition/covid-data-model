from typing import Mapping, Optional, List, Union
import dataclasses
import pathlib
import sys
import os
from dataclasses import dataclass
import logging
from multiprocessing import Pool

import us
import pandas as pd
import click

from covidactnow.datapublic.common_fields import CommonFields

from covidactnow.datapublic import common_init
from libs import pipeline
from libs.datasets import AggregationLevel
from libs.datasets import combined_datasets
from libs.datasets.timeseries import MultiRegionTimeseriesDataset
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from pyseir.deployment import webui_data_adaptor_v1
from pyseir.inference import whitelist
from pyseir.rt import infer_rt
from pyseir.icu import infer_icu
import pyseir.rt.patches
from pyseir.ensembles import ensemble_runner
from pyseir.inference import model_fitter
from pyseir.deployment.webui_data_adaptor_v1 import WebUIDataAdaptorV1
import pyseir.utils
from pyseir.inference.whitelist import WhitelistGenerator
from pyseir.rt.utils import NEW_ORLEANS_FIPS

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

root = logging.getLogger()

DEFAULT_RUN_MODE = "can-inference-derived"
ALL_STATES: List[str] = [state_obj.abbr for state_obj in us.STATES] + ["PR"]


def _cache_global_datasets():
    # Populate cache for combined latest and timeseries.  Caching pre-fork
    # will make sure cache is populated for subprocesses.  Return value
    # is not needed as the only goal is to populate the cache.
    combined_datasets.load_us_latest_dataset()
    combined_datasets.load_us_timeseries_dataset()
    infer_icu.get_region_weight_map()


@click.group()
def entry_point():
    """Basic entrypoint for cortex subcommands"""
    common_init.configure_logging()


def _generate_whitelist() -> pd.DataFrame:
    gen = WhitelistGenerator()
    all_us_timeseries = combined_datasets.load_us_timeseries_dataset()
    return gen.generate_whitelist(all_us_timeseries)


def _states_region_list(state: Optional[str], default: List[str]) -> List[pipeline.Region]:
    """Create a list of Region objects containing just state or default."""
    if state:
        return [pipeline.Region.from_state(state)]
    else:
        return [pipeline.Region.from_state(s) for s in default]


@dataclass
class StatePipeline:
    """Runs the pipeline for one state and stores the output."""

    region: pipeline.Region
    # infer_df provides support for the infection rate result.
    # TODO(tom): Rename when not refactoring it.
    infer_df: pd.DataFrame

    icu_data: OneRegionTimeseriesDataset
    fitter: model_fitter.ModelFitter
    ensemble: ensemble_runner.EnsembleRunner

    @staticmethod
    def run(region: pipeline.Region) -> "StatePipeline":
        assert region.is_state()
        infer_df = infer_rt.run_rt(infer_rt.RegionalInput.from_region(region))

        # Run ICU adjustment
        icu_input = infer_icu.RegionalInput.from_regional_data(
            combined_datasets.RegionalData.from_region(region)
        )
        icu_data = infer_icu.get_icu_timeseries_from_regional_input(
            icu_input, weight_by=infer_icu.ICUWeightsPath.ONE_MONTH_TRAILING_CASES
        )

        fitter_input = model_fitter.RegionalInput.from_state_region(region)
        fitter = model_fitter.ModelFitter.run_for_region(fitter_input)
        ensembles_input = ensemble_runner.RegionalInput.for_state(fitter)
        ensemble = ensemble_runner.make_and_run(ensembles_input)
        return StatePipeline(
            region=region, infer_df=infer_df, icu_data=icu_data, fitter=fitter, ensemble=ensemble
        )


@dataclass
class SubStateRegionPipelineInput:
    region: pipeline.Region
    run_fitter: bool
    state_fitter: model_fitter.ModelFitter
    regional_combined_dataset: combined_datasets.RegionalData

    @staticmethod
    def build_all(
        state_fitter_map: Mapping[pipeline.Region, model_fitter.ModelFitter],
        fips: Optional[str] = None,
        states: Optional[List[str]] = None,
    ) -> List["SubStateRegionPipelineInput"]:
        """For each region smaller than a state, build the input object used to run the pipeline."""
        # TODO(tom): Pass in the combined dataset instead of reading it from a global location.
        # Calculate the whitelist for the infection rate metric which makes no promises
        # about it's relationship to the SEIR subset
        if fips:  # A single Fips string was passed as a flag. Just run for that fips.
            infer_rt_regions = {pipeline.Region.from_fips(fips)}
        else:  # Default to the full infection rate whitelist
            infer_rt_regions = {
                *combined_datasets.get_subset_regions(
                    aggregation_level=AggregationLevel.COUNTY,
                    exclude_county_999=True,
                    states=states,
                )
            }
        # Now calculate the pyseir dependent whitelist
        whitelist_df = _generate_whitelist()
        # Make Region objects for all sub-state regions (counties, MSAs etc) that pass the whitelist
        # and parameters used to select subsets of regions.
        whitelist_regions = set(
            whitelist.regions_in_states(
                list(state_fitter_map.keys()), fips=fips, whitelist_df=whitelist_df
            )
        )

        pipeline_inputs = [
            SubStateRegionPipelineInput(
                region=region,
                run_fitter=region in whitelist_regions,
                state_fitter=state_fitter_map.get(region.get_state_region()),
                regional_combined_dataset=combined_datasets.RegionalData.from_region(region),
            )
            for region in (infer_rt_regions | whitelist_regions)
        ]
        return pipeline_inputs


@dataclass
class SubStatePipeline:
    """Runs the pipeline for one region smaller than a state and stores the output."""

    region: pipeline.Region
    infer_df: pd.DataFrame
    icu_data: Optional[OneRegionTimeseriesDataset]
    _combined_data: combined_datasets.RegionalData
    fitter: Optional[model_fitter.ModelFitter] = None
    ensemble: Optional[ensemble_runner.EnsembleRunner] = None

    @staticmethod
    def run(input: SubStateRegionPipelineInput) -> "SubStatePipeline":
        assert not input.region.is_state()
        # `infer_df` does not have the NEW_ORLEANS patch applied. TODO(tom): Rename to something like
        # infection_rate.
        infer_rt_input = infer_rt.RegionalInput.from_region(input.region)
        infer_df = infer_rt.run_rt(infer_rt_input)

        # Run ICU adjustment
        icu_input = infer_icu.RegionalInput.from_regional_data(input.regional_combined_dataset)
        try:
            icu_data = infer_icu.get_icu_timeseries_from_regional_input(
                icu_input, weight_by=infer_icu.ICUWeightsPath.ONE_MONTH_TRAILING_CASES
            )
        except KeyError:
            icu_data = None
            root.exception(f"Failed to run icu data for {input.region}")

        if input.run_fitter:
            fitter_input = model_fitter.RegionalInput.from_substate_region(
                input.region, input.state_fitter
            )
            fitter = model_fitter.ModelFitter.run_for_region(fitter_input)
            ensembles_input = ensemble_runner.RegionalInput.for_substate(
                fitter, state_fitter=input.state_fitter
            )
            ensemble = ensemble_runner.make_and_run(ensembles_input)
        else:
            fitter = None
            ensemble = None

        return SubStatePipeline(
            region=input.region,
            infer_df=infer_df,
            icu_data=icu_data,
            fitter=fitter,
            ensemble=ensemble,
            _combined_data=input.regional_combined_dataset,
        )

    def population(self) -> float:
        return self._combined_data.latest[CommonFields.POPULATION]


def _patch_substatepipeline_nola_infection_rate(
    pipelines: List[SubStatePipeline],
) -> List[SubStatePipeline]:
    """Returns a new list of pipeline objects with New Orleans infection rate patched."""
    pipeline_map = {p.region: p for p in pipelines}

    input_regions = set(pipeline_map.keys())
    new_orleans_regions = set(pipeline.Region.from_fips(f) for f in NEW_ORLEANS_FIPS)
    regions_to_patch = input_regions & new_orleans_regions
    if regions_to_patch:
        root.info("Applying New Orleans Patch")
        if len(regions_to_patch) != len(new_orleans_regions):
            root.warning(
                f"Missing New Orleans counties break patch: {new_orleans_regions - input_regions}"
            )

        nola_input_pipelines = [pipeline_map[fips] for fips in regions_to_patch]
        infection_rate_map = {p.region: p.infer_df for p in nola_input_pipelines}
        population_map = {p.region: p.population() for p in nola_input_pipelines}

        # Aggregate the results created so far into one timeseries of metrics in a DataFrame
        nola_infection_rate = pyseir.rt.patches.patch_aggregate_rt_results(
            infection_rate_map, population_map
        )

        for region in regions_to_patch:
            this_fips_infection_rate = nola_infection_rate.copy()
            this_fips_infection_rate.insert(0, CommonFields.LOCATION_ID, region.location_id)
            # Make a new SubStatePipeline object with the new infer_df
            pipeline_map[region] = dataclasses.replace(
                pipeline_map[region], infer_df=this_fips_infection_rate,
            )

    return list(pipeline_map.values())


def _write_pipeline_output(
    pipelines: List[Union[SubStatePipeline, StatePipeline]],
    output_dir: str,
    output_interval_days: int = 4,
):

    infection_rate_metric_df = pd.concat((p.infer_df for p in pipelines), ignore_index=True)
    multiregion_rt = MultiRegionTimeseriesDataset.from_timeseries_df(infection_rate_metric_df)
    output_path = pathlib.Path(output_dir) / pyseir.utils.SummaryArtifact.RT_METRIC_COMBINED.value
    multiregion_rt.to_csv(output_path)
    root.info(f"Saving Rt results to {output_path}")

    icu_df = pd.concat((p.icu_data.data for p in pipelines if p.icu_data), ignore_index=True)
    multiregion_icu = MultiRegionTimeseriesDataset.from_timeseries_df(icu_df)
    output_path = pathlib.Path(output_dir) / pyseir.utils.SummaryArtifact.ICU_METRIC_COMBINED.value
    multiregion_icu.to_csv(output_path)
    root.info(f"Saving ICU results to {output_path}")

    # does not parallelize well, because web_ui mapper doesn't serialize efficiently
    # TODO: Remove intermediate artifacts and paralellize artifacts creation better
    # Approximately 40% of the processing time is taken on this step
    web_ui_mapper = WebUIDataAdaptorV1(
        output_interval_days=output_interval_days, output_dir=output_dir,
    )
    webui_inputs = [
        webui_data_adaptor_v1.RegionalInput.from_results(p.fitter, p.ensemble, p.infer_df)
        for p in pipelines
        if p.fitter
    ]

    with Pool(maxtasksperchild=1) as p:
        p.map(web_ui_mapper.write_region_safely, webui_inputs)


def _build_all_for_states(
    states: List[str], states_only=False, fips: Optional[str] = None,
) -> List[Union[StatePipeline, SubStatePipeline]]:
    # prepare data
    _cache_global_datasets()

    # do everything for just states in parallel
    with Pool(maxtasksperchild=1) as pool:
        states_regions = [pipeline.Region.from_state(s) for s in states]
        state_pipelines: List[StatePipeline] = pool.map(StatePipeline.run, states_regions)
        state_fitter_map = {p.region: p.fitter for p in state_pipelines}

    if states_only:
        return state_pipelines

    substate_inputs = SubStateRegionPipelineInput.build_all(
        state_fitter_map, fips=fips, states=states
    )

    with Pool(maxtasksperchild=1) as p:
        root.info(f"executing pipeline for {len(substate_inputs)} counties")

        substate_pipelines = p.map(SubStatePipeline.run, substate_inputs)

    substate_pipelines = _patch_substatepipeline_nola_infection_rate(substate_pipelines)

    return state_pipelines + substate_pipelines


@entry_point.command()
def generate_whitelist():
    _generate_whitelist()


@entry_point.command()
@click.option(
    "--state", help="State to generate files for. If no state is given, all states are computed."
)
@click.option(
    "--states-only",
    default=False,
    is_flag=True,
    type=bool,
    help="Warning: This flag is unused and the function always defaults to only state "
    "level regions",
)
def run_infer_rt(state, states_only):
    for state in _states_region_list(state=state, default=ALL_STATES):
        infer_rt.run_rt(infer_rt.RegionalInput.from_region(state))


@entry_point.command()
@click.option(
    "--states",
    "-s",
    multiple=True,
    help="a list of states to generate files for. If no state is given, all states are computed.",
)
@click.option(
    "--output-interval-days",
    default=1,
    type=int,
    help="Number of days between outputs for the WebUI payload.",
)
@click.option(
    "--skip-whitelist", default=False, is_flag=True, type=bool, help="Skip the whitelist phase."
)
@click.option(
    "--fips",
    help=(
        "County level fips code to restrict runs to. "
        "This does not restrict the states that run, so also specifying states with "
        "`--states` is recommended."
    ),
)
@click.option("--states-only", is_flag=True, help="If set, only runs on states.")
@click.option("--output-dir", default="output/", type=str, help="Directory to deploy webui output.")
def build_all(
    states, output_interval_days, output_dir, skip_whitelist, states_only, fips,
):
    # split columns by ',' and remove whitespace
    states = [c.strip() for c in states]
    states = [us.states.lookup(state).abbr for state in states]
    states = [state for state in states if state in ALL_STATES]
    if not len(states):
        states = ALL_STATES

    pipelines = _build_all_for_states(states, states_only=states_only, fips=fips,)
    _write_pipeline_output(pipelines, output_dir, output_interval_days=output_interval_days)


if __name__ == "__main__":
    try:
        entry_point()  # pylint: disable=no-value-for-parameter
    except Exception:
        # According to https://github.com/getsentry/sentry-python/issues/480 Sentry is expected
        # to create an event when this is called.
        logging.exception("Exception reached __main__")
        raise
