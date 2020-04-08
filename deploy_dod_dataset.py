#!/usr/bin/env python
from io import BytesIO
import boto3
import click
import os
import logging

from libs.enums import Intervention
from libs.validate_results import validate_states_df, validate_counties_df, validate_states_shapefile, validate_counties_shapefile
from libs.build_dod_dataset import get_usa_by_county_with_projection_df, get_usa_by_states_df
from libs.functions import generate_shapefiles
from libs import dataset_deployer
logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"


@click.command()
@click.option('--run_validation', '-r', default=True, help='Run the validation on the deploy command')
@click.option('--input', '-i', default='results', help='Input directory of state/county projections')
@click.option('--output', '-o', default='results/dod', help='Output directory for artifacts')
def deploy(run_validation, input, output):
    """The entry function for invocation"""

    for intervention_enum in list(Intervention):
        logger.info(f"Starting to generate files for {intervention_enum.name}.")

        states_key_name = f'states.{intervention_enum.name}'
        states_df = get_usa_by_states_df(input, intervention_enum.value)
        if run_validation:
            validate_states_df(states_key_name, states_df)
        dataset_deployer.upload_csv(states_key_name, states_df.to_csv(), output)

        states_shp, states_shx, states_dbf = generate_shapefiles.get_usa_state_shapefile(
            states_df
        )
        if run_validation:
            validate_states_shapefile(states_key_name, states_shp, states_shx, states_dbf)
        dataset_deployer.deploy_shape_files(output, states_key_name, states_shp, states_shx, states_dbf)
        logger.info(f"Generated state shape files for {intervention_enum.name}")

        counties_key_name = f'counties.{intervention_enum.name}'
        counties_df = get_usa_by_county_with_projection_df(input, intervention_enum.value)
        if run_validation:
            validate_counties_df(counties_key_name, counties_df)
        dataset_deployer.upload_csv(counties_key_name, counties_df.to_csv(), output)

        counties_shp, counties_shx, counties_dbf = generate_shapefiles.get_usa_county_shapefile(
            counties_df
        )
        if run_validation:
            validate_counties_shapefile(counties_key_name, counties_shp, counties_shx, counties_dbf)
        dataset_deployer.deploy_shape_files(output, counties_key_name, counties_shp, counties_shx, counties_dbf)
        logger.info(f"Generated counties shape files for {intervention_enum.name}")

    print('finished dod job')


if __name__ == "__main__":
    """Used for manual trigger

    # triggering persistance to s3
    AWS_PROFILE=covidactnow BUCKET_NAME=covidactnow-deleteme python deploy_dod_dataset.py

    # deploy to the data bucket
    AWS_PROFILE=covidactnow BUCKET_NAME=data.covidactnow.org python deploy_dod_dataset.py

    # triggering persistance to local
    python deploy_dod_dataset.py
    """
    # pylint: disable=no-value-for-parameter
    logging.basicConfig(level=logging.INFO)
    deploy()
