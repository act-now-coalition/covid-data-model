import boto3
import os

from libs.build_dod_dataset import get_usa_by_county_df, get_usa_by_states_df
s3 = boto3.client('s3')  # Create an S3 client
# Supplied by ENV on AWS
# BUCKET_NAME format is s3://{BUCKET_NAME}
bucket_name = os.environ.get('BUCKET_NAME')


def persist_to_s3(key='my_public_identifier', body='empty'):
    """Persists specific data onto an s3 bucket.
    This method assumes versioned is handled on the bucket itself.

    Keyword Arguments:
        key {str} -- the file name on s3 (default: {'my_public_identifier'})
        body {str} -- the file content as either json or csv (default: {'empty'})

    Returns:
        [ResponseMetadata] -- the AWS SDK response object
    """
    response = s3.put_object(Bucket=bucket_name,
                             Key=key,
                             Body=body,
                             ACL='public-read')
    return response


def handler(event, context):
    """The entry function for invokation

    Arguments:
        event {dict} -- Used by AWS to pass in event data
        context {} -- Used by AWs uses this parameter to provide runtime information

    Returns:
    """
    print('creating states.csv')
    states_csv_buffer = get_usa_by_states_df().to_csv()
    states_blob = {'key': 'states.csv', 'body': states_csv_buffer}
    print('persisting states.csv')
    persist_to_s3(**states_blob)

    print('creating counties.csv')
    counties_csv_buffer = get_usa_by_county_df().to_csv()
    counties_blob = {'key': 'counties.csv', 'body': counties_csv_buffer}
    print('persisting counties.csv')
    persist_to_s3(**counties_blob)

    print('finished job')


if __name__ == "__main__":
    """
    Used for local testing i.e.

    AWS_PROFILE=covidactnow BUCKET_NAME=covidactnow-models-staging python deploy_dod_dataset.py
    """
    handler({}, {})
