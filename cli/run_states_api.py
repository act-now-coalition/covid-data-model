#!/usr/bin/env python
import click
import logging
import os


from libs.pipelines import api_pipeline
from libs.datasets.dataset_utils import AggregationLevel
from libs.enums import Intervention

logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"


@click.command("deploy-states-api")
@click.option(
    "--disable-validation", "-dv", is_flag=True, help="Run the validation on the deploy command",
)
@click.option(
    "--input-dir", "-i", default="results", help="Input directory of state projections",
)
@click.option(
    "--output", "-o", default="results/output/states", help="Output directory for artifacts",
)
@click.option(
    "--summary-output", default="results/output", help="Output directory for state summaries.",
)
def deploy_states_api(disable_validation, input_dir, output, summary_output):
    """The entry function for invocation"""

    # check that the dirs exist before starting
    for directory in [input_dir, output, summary_output]:
        if not os.path.isdir(directory):
            raise NotADirectoryError(directory)

    for intervention in list(Intervention):
        logger.info(f"Running intervention {intervention.name}")
        states_result = api_pipeline.run_projections(
            input_dir, AggregationLevel.STATE, intervention, run_validation=not disable_validation,
        )
        state_summaries, state_timeseries = api_pipeline.generate_api(states_result, input_dir)
        api_pipeline.deploy_results([*state_summaries, *state_timeseries], output)

        states_summary = api_pipeline.build_states_summary(state_summaries, intervention)
        states_timeseries = api_pipeline.build_states_timeseries(state_timeseries, intervention)
        summarized_timeseries = api_pipeline.build_prediction_header_timeseries_data(
            states_timeseries
        )
        api_pipeline.deploy_prediction_timeseries_csvs(summarized_timeseries, summary_output)
        api_pipeline.deploy_results([states_summary], summary_output, write_csv=True)
        api_pipeline.deploy_results([states_timeseries], summary_output)
