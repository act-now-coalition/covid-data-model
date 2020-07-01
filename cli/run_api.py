#!/usr/bin/env python
import click
import logging
import itertools
import os
import pathlib

from libs.pipelines import api_pipeline
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets import combined_datasets
from libs.enums import Intervention
from libs.datasets.dataset_utils import AggregationLevel

logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"


@click.command()
@click.option(
    "--disable-validation", "-dv", is_flag=True, help="Run the validation on the deploy command",
)
@click.option(
    "--input-dir",
    "-i",
    default="results",
    help="Input directory of state projections",
    type=pathlib.Path,
)
@click.option(
    "--output",
    "-o",
    default="results/output/states",
    help="Output directory for artifacts",
    type=pathlib.Path,
)
@click.option(
    "--summary-output",
    default="results/output",
    help="Output directory for state summaries.",
    type=pathlib.Path,
)
@click.option("--aggregation-level", "-l", type=AggregationLevel)
@click.option("--state")
def deploy_api(disable_validation, input_dir, output, summary_output, aggregation_level, state):
    """The entry function for invocation"""

    # check that the dirs exist before starting
    # for directory in [input_dir, output]:
    #     if not os.path.isdir(directory):
    #         raise NotADirectoryError(directory)

    us_latest = combined_datasets.build_us_latest_with_all_fields().get_subset(
        aggregation_level, state=state
    )
    us_timeseries = combined_datasets.build_us_timeseries_with_all_fields().get_subset(
        aggregation_level, state=state
    )

    for intervention in list(Intervention):
        if intervention is not Intervention.OBSERVED_INTERVENTION:
            continue

        logger.info(f"Running intervention {intervention.name}")
        all_timeseries = api_pipeline.run_on_all_fips_for_intervention(
            us_latest, us_timeseries, intervention, input_dir
        )
        county_timeseries = [
            output for output in all_timeseries if output.aggregate_level is AggregationLevel.COUNTY
        ]
        api_pipeline.deploy_single_level(
            intervention, county_timeseries, summary_output, summary_output
        )
        state_timeseries = [
            output for output in all_timeseries if output.aggregate_level is AggregationLevel.STATE
        ]
        api_pipeline.deploy_single_level(
            intervention, state_timeseries, summary_output, summary_output
        )
