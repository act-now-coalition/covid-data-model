#!/usr/bin/env python
import click
import logging


from libs.pipelines import api_pipeline
from libs.datasets.dataset_utils import AggregationLevel
from libs.enums import Intervention

logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"


@click.command("deploy-states-api")
@click.option(
    "--disable-validation",
    "-dv",
    is_flag=True,
    help="Run the validation on the deploy command",
)
@click.option(
    "--input-dir", "-i", default="results", help="Input directory of state projections",
)
@click.option(
    "--output",
    "-o",
    default="results/output/states",
    help="Output directory for artifacts",
)
def deploy_states_api(disable_validation, input_dir, output):
    """The entry function for invocation"""

    for intervention in list(Intervention):
        states_result = api_pipeline.run_projections(
            input_dir,
            AggregationLevel.STATE,
            intervention,
            run_validation=not disable_validation,
        )
        states_results_api = api_pipeline.generate_api(states_result, input_dir)
        api_pipeline.deploy_results(states_results_api, output)

        logger.info("finished states job")
