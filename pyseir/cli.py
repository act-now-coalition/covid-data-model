from typing import Optional, List
import dataclasses
import pathlib
import sys
import os
from dataclasses import dataclass
import logging

import us
import pandas as pd
import click

from covidactnow.datapublic.common_fields import CommonFields

from covidactnow.datapublic import common_init
from libs import parallel_utils
from libs import pipeline
from libs.datasets import AggregationLevel
from libs.datasets import combined_datasets
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.timeseries import MultiRegionTimeseriesDataset
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from pyseir.rt import infer_rt
from pyseir.icu import infer_icu
import pyseir.rt.patches

import pyseir.utils
from pyseir.rt.utils import NEW_ORLEANS_FIPS

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

root = logging.getLogger()

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


def _states_region_list(state: Optional[str], default: List[str]) -> List[pipeline.Region]:
    """Create a list of Region objects containing just state or default."""
    if state:
        return [pipeline.Region.from_state(state)]
    else:
        return [pipeline.Region.from_state(s) for s in default]


def _build_region_pipeline_input(region: pipeline.Region):
    return RegionPipelineInput(
        region=region, regional_combined_dataset=combined_datasets.RegionalData.from_region(region),
    )


@dataclass
class RegionPipelineInput:
    region: pipeline.Region
    regional_combined_dataset: combined_datasets.RegionalData

    @staticmethod
    def build_all(
        fips: Optional[str] = None,
        states: Optional[List[str]] = None,
        level: AggregationLevel = None,
    ) -> List["RegionPipelineInput"]:
        """For each region smaller than a state, build the input object used to run the pipeline."""
        # TODO(tom): Pass in the combined dataset instead of reading it from a global location.
        regions = {
            *combined_datasets.get_subset_regions(
                fips=fips, aggregation_level=level, exclude_county_999=True, states=states,
            )
        }
        pipeline_inputs = parallel_utils.parallel_map(_build_region_pipeline_input, regions)
        return list(pipeline_inputs)


@dataclass
class RegionPipeline:
    """Runs the pipeline for one region smaller than a state and stores the output."""

    region: pipeline.Region
    infer_df: pd.DataFrame
    icu_data: Optional[OneRegionTimeseriesDataset]
    _combined_data: combined_datasets.RegionalData

    @staticmethod
    def run(input: RegionPipelineInput) -> "RegionPipeline":
        # `infer_df` does not have the NEW_ORLEANS patch applied. TODO(tom): Rename to something like
        # infection_rate.
        infer_rt_input = infer_rt.RegionalInput.from_region(input.region)
        try:
            infer_df = infer_rt.run_rt(infer_rt_input)
        except Exception:
            root.exception(f"run_rt failed for {input.region}")
            raise

        # Run ICU adjustment
        icu_input = infer_icu.RegionalInput.from_regional_data(input.regional_combined_dataset)
        try:
            icu_data = infer_icu.get_icu_timeseries_from_regional_input(
                icu_input, weight_by=infer_icu.ICUWeightsPath.ONE_MONTH_TRAILING_CASES
            )
        except KeyError:
            icu_data = None
            root.exception(f"Failed to run icu data for {input.region}")

        return RegionPipeline(
            region=input.region,
            infer_df=infer_df,
            icu_data=icu_data,
            _combined_data=input.regional_combined_dataset,
        )

    @property
    def fips(self) -> str:
        return self.region.fips

    def population(self) -> float:
        return self._combined_data.latest[CommonFields.POPULATION]


def _patch_nola_infection_rate_in_pipelines(
    pipelines: List[RegionPipeline],
) -> List[RegionPipeline]:
    """Returns a new list of pipeline objects with New Orleans infection rate patched."""
    pipeline_map = {p.fips: p for p in pipelines}

    input_fips = set(pipeline_map.keys())
    fips_to_patch = input_fips & set(NEW_ORLEANS_FIPS)
    if fips_to_patch:
        root.info("Applying New Orleans Patch")
        if len(fips_to_patch) != len(NEW_ORLEANS_FIPS):
            root.warning(
                f"Missing New Orleans counties break patch: {set(NEW_ORLEANS_FIPS) - input_fips}"
            )

        nola_input_pipelines = [pipeline_map[fips] for fips in fips_to_patch]
        infection_rate_map = {p.region: p.infer_df for p in nola_input_pipelines}
        population_map = {p.region: p.population() for p in nola_input_pipelines}

        # Aggregate the results created so far into one timeseries of metrics in a DataFrame
        nola_infection_rate = pyseir.rt.patches.patch_aggregate_rt_results(
            infection_rate_map, population_map
        )

        for fips in fips_to_patch:
            this_fips_infection_rate = nola_infection_rate.copy()
            this_fips_infection_rate.insert(0, CommonFields.FIPS, fips)
            # Make a new RegionPipeline object with the new infer_df
            pipeline_map[fips] = dataclasses.replace(
                pipeline_map[fips], infer_df=this_fips_infection_rate,
            )

    return list(pipeline_map.values())


def _write_pipeline_output(
    pipelines: List[RegionPipeline], output_dir: str,
):
    output_dir_path = pathlib.Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir()

    infection_rate_metric_df = pd.concat((p.infer_df for p in pipelines), ignore_index=True)
    # TODO: Use constructors in MultiRegionTimeseriesDataset
    timeseries_dataset = TimeseriesDataset(infection_rate_metric_df)
    latest = timeseries_dataset.latest_values_object()
    multiregion_rt = MultiRegionTimeseriesDataset.from_timeseries_and_latest(
        timeseries_dataset, latest
    )
    output_path = output_dir_path / pyseir.utils.SummaryArtifact.RT_METRIC_COMBINED.value
    multiregion_rt.to_csv(output_path)
    root.info(f"Saving Rt results to {output_path}")

    icu_df = pd.concat((p.icu_data.data for p in pipelines if p.icu_data), ignore_index=True)
    timeseries_dataset = TimeseriesDataset(icu_df)
    latest = timeseries_dataset.latest_values_object().data.set_index(CommonFields.LOCATION_ID)
    multiregion_icu = MultiRegionTimeseriesDataset(icu_df, latest)

    output_path = output_dir_path / pyseir.utils.SummaryArtifact.ICU_METRIC_COMBINED.value
    multiregion_icu.to_csv(output_path)
    root.info(f"Saving ICU results to {output_path}")


def _run_on_all_regions(
    states: List[str] = None, states_only: bool = False, fips: Optional[str] = None,
) -> List[RegionPipeline]:
    # prepare data
    _cache_global_datasets()

    level = None

    if states_only:
        level = AggregationLevel.STATE

    region_inputs = RegionPipelineInput.build_all(fips=fips, states=states, level=level)

    root.info(f"Executing pipeline for {len(region_inputs)} regions")
    region_pipelines = parallel_utils.parallel_map(RegionPipeline.run, region_inputs)

    region_pipelines = _patch_nola_infection_rate_in_pipelines(region_pipelines)

    return region_pipelines


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
@click.option("--states-only", is_flag=True, help="If set, only runs on states.")
@click.option("--output-dir", default="output/", type=str, help="Directory to deploy webui output.")
def build_all(states, output_dir, states_only, fips):
    # split columns by ',' and remove whitespace
    states = [c.strip() for c in states]
    states = [us.states.lookup(state).abbr for state in states]
    states = [state for state in states if state in ALL_STATES]

    if not len(states):
        states = ALL_STATES

    pipelines = _run_on_all_regions(states=states, states_only=states_only, fips=fips)
    _write_pipeline_output(pipelines, output_dir)


if __name__ == "__main__":
    try:
        entry_point()  # pylint: disable=no-value-for-parameter
    except Exception:
        # According to https://github.com/getsentry/sentry-python/issues/480 Sentry is expected
        # to create an event when this is called.
        logging.exception("Exception reached __main__")
        raise
