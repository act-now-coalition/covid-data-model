import pathlib
from typing import Optional, List
import dataclasses
import sys
import os
import logging
import us
import click


from datapublic import common_init
from datapublic.common_fields import CommonFields

from libs.pipelines import api_v2_pipeline
from libs import parallel_utils
from libs import pipeline
from libs.datasets import AggregationLevel
from libs.datasets import combined_datasets
import pyseir.rt.patches

import pyseir.utils
from pyseir.rt import infer_rt
from pyseir.rt.utils import NEW_ORLEANS_FIPS
from pyseir.run import OneRegionPipeline

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

root = logging.getLogger()

ALL_STATES: List[str] = [state_obj.abbr for state_obj in us.STATES] + ["PR"]


def _cache_global_datasets():
    # Populate cache for combined timeseries.  Caching pre-fork
    # will make sure cache is populated for subprocesses.  Return value
    # is not needed as the only goal is to populate the cache.
    # Access data here to populate the property cache.
    combined_datasets.load_us_timeseries_dataset()


@click.group()
def entry_point():
    """Basic entrypoint for cortex subcommands"""
    common_init.configure_logging()


def _states_region_list(state: Optional[str], default: List[str]) -> List[pipeline.Region]:
    """Create a list of Region objects containing just state or default."""
    if state:
        return [pipeline.Region.from_state(state)]
    else:
        return [pipeline.Region.from_state(s) for s in default]


def _patch_nola_infection_rate_in_pipelines(
    pipelines: List[OneRegionPipeline],
) -> List[OneRegionPipeline]:
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
    "--fips",
    help=(
        "County level fips code to restrict runs to. "
        "This does not restrict the states that run, so also specifying states with "
        "`--states` is recommended."
    ),
)
@click.option("--level", "-l", type=AggregationLevel)
@click.option(
    "--output-dir",
    default="output/",
    type=pathlib.Path,
    help="Directory to deploy " "webui output.",
)
@click.option(
    "--location-id-matches",
    help="If set only location_id matching this regular expression are processed",
)
@click.option(
    "--generate-api-v2",
    default=False,
    is_flag=True,
    type=bool,
    help="Generate API v2 output after PySEIR finishes",
)
def build_all(states, output_dir, level, fips, location_id_matches: str, generate_api_v2: bool):
    # split columns by ',' and remove whitespace
    states = [c.strip() for c in states]
    states = [us.states.lookup(state).abbr for state in states]
    states = [state for state in states if state in ALL_STATES]

    # prepare data
    _cache_global_datasets()

    regions_dataset = combined_datasets.load_us_timeseries_dataset().get_subset(
        fips=fips,
        aggregation_level=level,
        exclude_county_999=True,
        states=states,
        location_id_matches=location_id_matches,
    )
    regions = [one_region for _, one_region in regions_dataset.iter_one_regions()]
    root.info(f"Executing pipeline for {len(regions)} regions")
    region_pipelines: List[OneRegionPipeline] = parallel_utils.parallel_map(
        OneRegionPipeline.run, regions
    )
    region_pipelines = _patch_nola_infection_rate_in_pipelines(region_pipelines)

    model_output = pyseir.run.PyseirOutputDatasets.from_pipeline_output(region_pipelines)
    model_output.write(output_dir, root)

    if generate_api_v2:
        api_v2_pipeline.generate_from_loaded_data(model_output, output_dir, regions_dataset, root)


if __name__ == "__main__":
    try:
        entry_point()  # pylint: disable=no-value-for-parameter
    except Exception:
        # According to https://github.com/getsentry/sentry-python/issues/480 Sentry is expected
        # to create an event when this is called.
        logging.exception("Exception reached __main__")
        raise
