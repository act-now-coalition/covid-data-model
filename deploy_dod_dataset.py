#!/usr/bin/env python
import click
import logging

from libs.enums import Intervention
from libs.pipelines import dod_pipeline
logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"


@click.command()
@click.option('--run_validation', '-r', default=True, help='Run the validation on the deploy command')
@click.option('--input-dir', '-i', default='results', help='Input directory of state/county projections')
@click.option('--output', '-o', default='results/dod', help='Output directory for artifacts')
def deploy(run_validation, input_dir, output):
    """The entry function for invocation"""

    for intervention in list(Intervention):
        logger.info(f"Starting to generate files for {intervention.name}.")

        state_result, county_result = dod_pipeline.run_projections(
            input_dir, intervention, run_validation=run_validation
        )
        dod_pipeline.deploy_results(state_result, output)
        dod_pipeline.deploy_results(county_result, output)

    logger.info('finished dod job')


if __name__ == "__main__":
    """Used for manual trigger

    # triggering persistance to s3
    AWS_PROFILE=covidactnow BUCKET_NAME=covidactnow-deleteme python deploy_dod_dataset.py

    # deploy to the data bucket
    AWS_PROFILE=covidactnow BUCKET_NAME=data.covidactnow.org python deploy_dod_dataset.py

    # triggering persistance to local
    python deploy_dod_dataset.py
    """
    logging.basicConfig(level=logging.INFO)
    # pylint: disable=no-value-for-parameter
    deploy()
