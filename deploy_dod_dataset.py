#!/usr/bin/env python
from io import BytesIO
import boto3
import click
import os
import logging

from libs.build_params import OUTPUT_DIR_DOD
from libs.constants import INTERVENTIONS
from libs.validate_results import validate_states_df, validate_counties_df, validate_states_shapefile, validate_counties_shapefile
from libs.build_dod_dataset import get_usa_by_county_df, get_usa_by_states_df, get_usa_county_shapefile, get_usa_state_shapefile
from libs.build_dod_dataset import get_usa_by_county_with_projection_df

logger = logging.getLogger()
PROD_BUCKET = "data.covidactnow.org"

class DatasetDeployer():

    def __init__(self, key='filename.csv', body='a random data', output_dir='.'):
        self.s3 = boto3.client('s3')
        # Supplied by ENV on AWS
        # BUCKET_NAME format is s3://{BUCKET_NAME}
        self.bucket_name = os.environ.get('BUCKET_NAME')
        self.key = key
        self.body = body
        self.output_dir = output_dir

    def _persist_to_s3(self):
        """Persists specific data onto an s3 bucket.
        This method assumes versioned is handled on the bucket itself.
        """
        print('persisting {} to s3'.format(self.key))

        response = self.s3.put_object(Bucket=self.bucket_name,
                                      Key=self.key,
                                      Body=self.body,
                                      ACL='public-read')
        return response

    def _persist_to_local(self):
        """Persists specific data onto an s3 bucket.
        This method assumes versioned is handled on the bucket itself.
        """
        print('persisting {} to local'.format(self.key))

        with open(os.path.join(self.output_dir, self.key), 'wb') as f:
            # hack to allow the local writer to take either bytes or a string
            # note this assumes that all strings are given in utf-8 and not,
            # like, ASCII
            f.write(self.body.encode('UTF-8') if isinstance(self.body, str) else self.body)

        pass

    def persist(self):
        if self.bucket_name:
            self._persist_to_s3()
        else:
            self._persist_to_local()
        return

def upload_csv(key_name, csv): 
    blob = {
        'key': f'{key_name}.csv',
        'body': csv
        }
    obj = DatasetDeployer(**blob)
    obj.persist()
    logger.info(f"Generated csv for {key_name}")


@click.command()
@click.option('--input', '-i', default='results', help='Input directory of state/county projections')
@click.option('--output', '-o', default='', help='Output directory for artifacts')
def deploy(input, output, should_run_validation):
    """The entry function for invocation"""
    for intervention in INTERVENTIONS: 
        logger.info(f"Starting to generate files for {intervention['intervention_name']}.")

        states_key_name = f'states.{intervention["intervention_name"]}'
        states_df = get_usa_by_states_df(input, intervention["intervention_enum"])
        if should_run_validation: 
            validate_states_df(states_key_name, states_df)
        upload_csv(states_key_name, states_df.to_csv())

        states_shp = BytesIO()
        states_shx = BytesIO()
        states_dbf = BytesIO()
        get_usa_state_shapefile(input,  intervention["intervention_enum"], states_shp, states_shx, states_dbf)
        if should_run_validation: 
            validate_states_shapefile(states_key_name, states_shp, states_shx, states_dbf)
        DatasetDeployer(key=f'{states_key_name}.shp', body=states_shp.getvalue()).persist()
        DatasetDeployer(key=f'{states_key_name}.shx', body=states_shx.getvalue()).persist()
        DatasetDeployer(key=f'{states_key_name}.dbf', body=states_dbf.getvalue()).persist()
        logger.info(f"Generated state shape files for {intervention['intervention_name']}")

        counties_key_name = f'counties.{intervention["intervention_name"]}'
        counties_df = get_usa_by_county_with_projection_df(input, intervention["intervention_enum"])
        if should_run_validation: 
            validate_counties_df(counties_key_name, counties_df)
        upload_csv(counties_key_name, counties_df.to_csv())

        counties_shp = BytesIO()
        counties_shx = BytesIO()
        counties_dbf = BytesIO()
        get_usa_county_shapefile(input, intervention["intervention_enum"], counties_shp, counties_shx, counties_dbf)
        if should_run_validation: 
            validate_counties_shapefile(counties_key_name, counties_shp, counties_shx, counties_dbf)
        DatasetDeployer(key=f'{counties_key_name}.shp', body=counties_shp.getvalue()).persist()
        DatasetDeployer(key=f'{counties_key_name}.shp', body=counties_shx.getvalue()).persist()
        DatasetDeployer(key=f'{counties_key_name}.shp', body=counties_dbf.getvalue()).persist()
        logger.info(f"Generated counties shape files for {intervention['intervention_name']}")

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
    should_run_validation = os.environ.get('BUCKET_NAME') == PROD_BUCKET
    # pylint: disable=no-value-for-parameter
    deploy(should_run_validation)
