from typing import Dict, Any
import os
import re
import uuid
import json
import logging
import boto3
from boto3.dynamodb.conditions import Key

_logger = logging.getLogger(__name__)

API_KEY_TABLE_NAME = os.environ["API_KEYS_TABLE"]
API_KEY_INDEX_NAME = "apiKeys"


class DynamoDBClient:
    def __init__(self, client=None):
        self._client = client or boto3.resource("dynamodb")

    def get_item(self, table, key):
        table = self._client.Table(table)
        result = table.get_item(Key=key)

        if "Item" not in result:
            return None

        return result["Item"]

    def query_index(self, table, index, key, value):
        table = self._client.Table(table)

        result = table.query(IndexName=index, KeyConditionExpression=Key(key).eq(value))
        return result["Items"]

    def put_item(self, table: str, item: Dict[str, Any]):
        table = self._client.Table(table)
        return table.put_item(Item=item)


class APIKeyRepo:
    @staticmethod
    def add_api_key(email, api_key):
        client = DynamoDBClient()
        obj = {"email": email, "api_key": api_key}
        client.put_item(API_KEY_TABLE_NAME, obj)

    @staticmethod
    def get_api_key(email):
        client = DynamoDBClient()

        key = {"email": email}
        api_key_item = client.get_item(API_KEY_TABLE_NAME, key)

        if not api_key_item:
            return None

        return api_key_item["api_key"]

    @staticmethod
    def get_record_for_api_key(api_key):
        client = DynamoDBClient()
        items = client.query_index(API_KEY_TABLE_NAME, API_KEY_INDEX_NAME, "api_key", api_key)
        if not items:
            return None

        if len(items) > 1:
            raise Exception("Multiple emails found for API key")

        return items[0]


class InvalidAPIKey(Exception):
    def __init__(self, api_key):
        super().__init__(f"Invalid API Key: {api_key}")
        self.api_key = api_key


def _create_api_key(email: str) -> str:
    return uuid.uuid4().hex


def _get_or_create_api_key(email):
    api_key = APIKeyRepo.get_api_key(email)
    if api_key:
        return api_key

    _logger.info(f"No API Key found for email {email}, creating new key")

    api_key = _create_api_key(email)
    APIKeyRepo.add_api_key(email, api_key)

    return api_key


def _check_api_key(api_key):
    if not APIKeyRepo.get_record_for_api_key(api_key):
        raise InvalidAPIKey(api_key)


def register(event, context):

    if not "email" in event:
        raise ValueError("Missing email parameter")

    email = event["email"]
    api_key = _get_or_create_api_key(email)
    body = {"api_key": api_key, "email": email}

    response = {"statusCode": 200, "body": json.dumps(body)}

    return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """


def _generate_accept_policy(user_record: dict, method_arn):
    *first, last = method_arn.split("/")
    method_arn = "/".join([*first, "*"])
    return {
        "principalId": user_record["email"],
        "policyDocument": {
            "Version": "2012-10-17",
            "Statement": [
                {"Action": "execute-api:Invoke", "Effect": "Allow", "Resource": method_arn}
            ],
        },
        "usageIdentifierKey": user_record["api_key"],
    }


def check_api_key(event, context):
    if not event["queryStringParameters"]["apiKey"]:
        raise Exception("Must have api key")

    api_key = event["queryStringParameters"]["apiKey"]

    record = APIKeyRepo.get_record_for_api_key(api_key)
    if not record:
        raise InvalidAPIKey(api_key)

    policy = _generate_accept_policy(record, event["methodArn"])

    return policy
