import boto3
import os


def make_dataset():
    """Used to create the target dataset.

    Returns:
        Dict[str, str]
    """
    ########### FILL IN BELOW ###########

    result = {'key': 'my-key', 'body': 'my-data'}

    ########### FILL IN ABOVE ###########
    return result


def persist_to_s3(s3_client, bucket_name: str, key: str = 'my_public_identifier', body: str = 'empty'):
    """Persists specific data onto an s3 bucket.

    This method assumes versioned is handled on the bucket itself.

    Args:
        key: the file name on s3 (default: {'my_public_identifier'})
        body: the file content as either json or csv (default: {'empty'})

    Returns:
        [ResponseMetadata] -- the AWS SDK response object
    """

    response = s3_client.put_object(Bucket=bucket_name,
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
    # Supplied by ENV on AWS
    # BUCKET_NAME format is s3://{BUCKET_NAME}
    bucket_name = os.environ.get('BUCKET_NAME')

    data = make_dataset()
    s3_client = boto3.client('s3')  # Create an S3 client

    return persist_to_s3(s3_client, bucket_name, **data)


if __name__ == "__main__":
    # used for locally testing
    handler({}, {})
