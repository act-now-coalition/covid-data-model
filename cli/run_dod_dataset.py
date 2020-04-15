#!/usr/bin/env python

import click
import logging

from libs.enums import Intervention
from libs.pipelines import dod_pipeline
logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"


@click.command('deploy-dod')
@click.option('--disable-validation', is_flag=True, help='Disable validation of generated projection files')
@click.option('--input-state-dir', '-is', default='results/state', help='Input directory of state projections')
@click.option('--input-county-dir', '-ic', default='results/county', help='Input directory of county projections')
@click.option('--output', '-o', default='results/dod', help='Output directory for artifacts')
def deploy_dod_projections(disable_validation, input_state_dir, input_county_dir, output):

    """Generates and runs dod data projections from model outputs.

    Used for manual trigger

    # triggering persistance to s3
    AWS_PROFILE=covidactnow BUCKET_NAME=covidactnow-deleteme python run.py deploy-dod-projections

    # deploy to the data bucket
    AWS_PROFILE=covidactnow BUCKET_NAME=data.covidactnow.org python run.py deploy-dod-projections

    # triggering persistance to local
    python run.py deploy-dod-projections
    """

    for intervention in list(Intervention):
        logger.info(f"Starting to generate files for {intervention.name}.")

        state_result, county_result = dod_pipeline.run_projections(
            input_state_dir, input_county_dir, intervention, run_validation=not disable_validation
        )
        dod_pipeline.deploy_results(state_result, output)
        dod_pipeline.deploy_results(county_result, output)

    logger.info('finished dod job')
