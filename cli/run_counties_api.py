#!/usr/bin/env python
import click
import logging


from libs.pipelines import api_pipeline
from libs.datasets.dataset_utils import AggregationLevel
from libs.enums import Intervention

logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"


@click.command("deploy-counties-api")
@click.option(
    "--disable-validation",
    "-dv",
    is_flag=True,
    help="Run the validation on the deploy command",
)
@click.option(
    "--input-dir",
    "-i",
    default="results",
    help="Input directory of county projections",
)
@click.option(
    "--output",
    "-o",
    default="results/output/counties",
    help="Output directory for artifacts",
)
def deploy_counties_api(disable_validation, input_dir, output):
    """The entry function for invocation"""

    for intervention in list(Intervention):
        county_result = api_pipeline.run_projections(
            input_dir,
            AggregationLevel.COUNTY,
            intervention,
            run_validation=not disable_validation,
        )
        county_results_api = api_pipeline.generate_api(county_result, input_dir)
        api_pipeline.deploy_results(county_results_api, output)

        logger.info("finished top counties job")
