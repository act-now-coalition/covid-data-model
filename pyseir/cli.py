import itertools
import sys
import os
from dataclasses import dataclass
from typing import Mapping
from typing import Optional
from typing import List
import logging
from multiprocessing import Pool

import us
import pandas as pd
import click

from covidactnow.datapublic import common_init
from libs import pipeline
from pyseir.deployment import webui_data_adaptor_v1
from pyseir.inference import whitelist
from pyseir.rt import infer_rt
from pyseir.ensembles import ensemble_runner
from pyseir.inference import model_fitter
from pyseir.deployment.webui_data_adaptor_v1 import WebUIDataAdaptorV1
from libs.datasets import combined_datasets
import pyseir.utils
from pyseir.inference.whitelist import WhitelistGenerator


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
    fitter: model_fitter.ModelFitter
    ensemble: ensemble_runner.EnsembleRunner

    @staticmethod
    def run(region: pipeline.Region) -> "StatePipeline":
        assert region.is_state()
        infer_df = infer_rt.run_rt(infer_rt.RegionalInput.from_region(region))
        fitter_input = model_fitter.RegionalInput.from_state_region(region)
        fitter = model_fitter.ModelFitter.run_for_region(fitter_input)
        ensembles_input = ensemble_runner.RegionalInput.for_state(fitter)
        ensemble = ensemble_runner.make_and_run(ensembles_input)
        return StatePipeline(region=region, infer_df=infer_df, fitter=fitter, ensemble=ensemble)


@dataclass
class SubStateRegionPipelineInput:
    region: pipeline.Region
    run_fitter: bool
    state_fitter: model_fitter.ModelFitter

    @staticmethod
    def build_all(
        state_fitter_map: Mapping[pipeline.Region, model_fitter.ModelFitter],
        fips: Optional[str] = None,
    ) -> List["SubStateRegionPipelineInput"]:
        """For each region smaller than a state, build the input object used to run the pipeline."""
        # TODO(tom): Pass in the combined dataset instead of reading it from a global location.
        # Calculate the whitelist for the infection rate metric which makes no promises
        # about it's relationship to the SEIR subset
        if fips:  # A single Fips string was passed as a flag. Just run for that fips.
            infer_rt_regions = {pipeline.Region.from_fips(fips)}
        else:  # Default to the full infection rate whitelist
            infer_rt_regions = {
                pipeline.Region.from_fips(x)
                for x in combined_datasets.load_us_latest_dataset().all_fips
                if len(x) == 5
                and "25" != x[:2]  # Counties only  # Masking MA Counties (2020-08-27) due to NaNs
                and "999" != x[-3:]  # Remove placeholder fips that have no data
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
                run_fitter=(region in whitelist_regions),
                state_fitter=state_fitter_map.get(region.get_state_region()),
            )
            for region in (infer_rt_regions | whitelist_regions)
        ]
        return pipeline_inputs


@dataclass
class SubStatePipeline:
    """Runs the pipeline for one region smaller than a state and stores the output."""

    region: pipeline.Region
    infer_df: pd.DataFrame
    fitter: Optional[model_fitter.ModelFitter]
    ensemble: ensemble_runner.EnsembleRunner

    @staticmethod
    def run(input: SubStateRegionPipelineInput) -> "SubStatePipeline":
        assert not input.region.is_state()
        infer_df = infer_rt.run_rt(infer_rt.RegionalInput.from_region(input.region))
        fitter_input = model_fitter.RegionalInput.from_substate_region(
            input.region, input.state_fitter
        )
        fitter = model_fitter.ModelFitter.run_for_region(fitter_input)
        ensembles_input = ensemble_runner.RegionalInput.for_substate(
            fitter, state_fitter=input.state_fitter
        )
        ensemble = ensemble_runner.make_and_run(ensembles_input)
        return SubStatePipeline(
            region=input.region, infer_df=infer_df, fitter=fitter, ensemble=ensemble
        )


def _build_all_for_states(
    states: List[str],
    output_interval_days=4,
    output_dir=None,
    states_only=False,
    fips: Optional[str] = None,
):
    # prepare data
    _cache_global_datasets()

    # do everything for just states in parallel
    with Pool(maxtasksperchild=1) as p:
        states_regions = [pipeline.Region.from_state(s) for s in states]
        state_pipelines: List[StatePipeline] = p.map(StatePipeline.run, states_regions)
        state_fitter_map = {p.region: p.fitter for p in state_pipelines}

    if states_only:
        root.info("Only executing for states. returning.")
        return

    substate_inputs = SubStateRegionPipelineInput.build_all(state_fitter_map, fips=fips)

    with Pool(maxtasksperchild=1) as p:
        root.info(f"executing pipeline for {len(substate_inputs)} counties")
        substate_pipelines = p.map(SubStatePipeline.run, substate_inputs)

    infection_rate_metric_df = pd.concat(
        [p.infer_df for p in itertools.chain(state_pipelines, substate_pipelines)]
    )

    infection_rate_metric_df.to_csv(
        path_or_buf=pyseir.utils.get_summary_artifact_path(
            pyseir.utils.SummaryArtifact.RT_METRIC_COMBINED
        ),
        index=False,
    )

    # output it all
    output_interval_days = int(output_interval_days)
    _cache_global_datasets()

    root.info(f"outputting web results for states and {len(substate_pipelines)} counties")

    # does not parallelize well, because web_ui mapper doesn't serialize efficiently
    # TODO: Remove intermediate artifacts and paralellize artifacts creation better
    # Approximately 40% of the processing time is taken on this step
    web_ui_mapper = WebUIDataAdaptorV1(
        output_interval_days=output_interval_days, output_dir=output_dir,
    )

    webui_inputs = [
        webui_data_adaptor_v1.RegionalInput.from_model_fitter(p.fitter)
        for p in itertools.chain(state_pipelines, substate_pipelines)
        if p.fitter
    ]
    with Pool(maxtasksperchild=1) as p:
        p.map(web_ui_mapper.write_region, webui_inputs)


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
@click.option("--output-dir", default=None, type=str, help="Directory to deploy webui output.")
def build_all(
    states, output_interval_days, output_dir, skip_whitelist, states_only, fips,
):
    # split columns by ',' and remove whitespace
    states = [c.strip() for c in states]
    states = [us.states.lookup(state).abbr for state in states]
    states = [state for state in states if state in ALL_STATES]
    if not len(states):
        states = ALL_STATES

    _build_all_for_states(
        states,
        output_interval_days=output_interval_days,
        output_dir=output_dir,
        states_only=states_only,
        fips=fips,
    )


if __name__ == "__main__":
    try:
        entry_point()  # pylint: disable=no-value-for-parameter
    except Exception:
        # According to https://github.com/getsentry/sentry-python/issues/480 Sentry is expected
        # to create an event when this is called.
        logging.exception("Exception reached __main__")
        raise
