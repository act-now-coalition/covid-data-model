import json
import boto3
from awsauth.config import Config


class FirehoseClient:
    def __init__(self, client=None):
        self._client = client or boto3.client("firehose", region_name=Config.Constants.AWS_REGION)

    def put_data(self, stream: str, data: dict):

        record = {"Data": json.dumps(data) + "\n"}
        self._client.put_record(
            DeliveryStreamName=stream, Record=record,
        )
