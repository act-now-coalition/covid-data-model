import boto3
import os

s3 = boto3.client('s3')  # Create an S3 client
# Supplied by ENV on AWS
# BUCKET_NAME format is s3://{BUCKET_NAME}
bucket_name = os.environ.get('BUCKET_NAME')


def make_dataset():
    """Used to create the target dataset.

    Returns:
        dict -- With a key {str} and body {str}
    """
    ########### FILL IN BELOW ###########

    result = {'key': 'my-key', 'body': 'my-data'}

    ########### FILL IN ABOVE ###########
    return result


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
        [ResponseMetadata] -- the AWS SDK response object
    """
    data = make_dataset()

    return persist_to_s3(**data)


if __name__ == "__main__":
    # used for locally testing
    handler({}, {})
