#!/usr/bin/env python
import click
import logging

from libs.pipelines import top_counties_pipeline

logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"


@click.command()
@click.option(
    "--run_validation",
    "-r",
    default=True,
    help="Run the validation on the deploy command",
)
@click.option(
    "--input-dir",
    "-i",
    default="results",
    help="Input directory of state/county projections",
)
@click.option(
    "--output",
    "-o",
    default="results/top_counties",
    help="Output directory for artifacts",
)
def deploy(run_validation, input_dir, output):
    """The entry function for invocation"""

    county_result = top_counties_pipeline.run_projections(
        input_dir, run_validation=True
    )
    county_results_api = top_counties_pipeline.generate_api(county_result)
    top_counties_pipeline.deploy_results(county_results_api, output)

    logger.info("finished top counties job")


if __name__ == "__main__":
    """Used for manual trigger

    # triggering persistance to s3
    AWS_PROFILE=covidactnow BUCKET_NAME=covidactnow-deleteme python deploy_top_counties_dataset.py

    # deploy to the data bucket
    AWS_PROFILE=covidactnow BUCKET_NAME=data.covidactnow.org python deploy_top_counties_dataset.py

    # triggering persistance to local
    python deploy_top_counties_dataset.py
    """
    logging.basicConfig(level=logging.INFO)
    # pylint: disable=no-value-for-parameter
    deploy()
