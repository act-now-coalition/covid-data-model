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

logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"


@click.command("deploy-states-api")
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
    "--summary-output", default="results/output", help="Output directory for state summaries.",
)
def deploy_states_api(disable_validation, input_dir, output, summary_output):
    """The entry function for invocation"""

    # check that the dirs exist before starting
    # for directory in [input_dir, output]:
    #     if not os.path.isdir(directory):
    #         raise NotADirectoryError(directory)

    us_latest = combined_datasets.build_us_latest_with_all_fields()
    us_latest_states = us_latest.get_subset(AggregationLevel.STATE)

    us_timeseries = combined_datasets.build_us_timeseries_with_all_fields()
    us_timeseries_states = us_timeseries.get_subset(AggregationLevel.STATE)

    for intervention in list(Intervention):
        logger.info(f"Running intervention {intervention.name}")
        api_processing_results = api_pipeline.run_on_all_fips_for_intervention(
            us_latest_states, us_timeseries_states, intervention, input_dir
        )
        try:
            all_summaries, all_timeseries = zip(*api_processing_results)
        except Exception:
            # no results were found for intervention type, continuing
            continue

        all_summaries = (
            api_pipeline.deploy_results(intervention, area_result, output)
            for area_result in all_summaries
        )
        all_timeseries = (
            api_pipeline.deploy_results(intervention, area_result, output)
            for area_result in all_timeseries
        )
