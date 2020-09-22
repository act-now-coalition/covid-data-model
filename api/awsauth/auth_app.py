import uuid
import json
import logging

from api_key_repo import APIKeyRepo


_logger = logging.getLogger(__name__)


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


def _generate_accept_policy(user_record: dict, method_arn):
    # The last part of the method arn does not give permission to the underlying resource.
    # Replacing with a wildcard to give access to paths below.
    *first, _ = method_arn.split("/")
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
    """Checks API Key included in request for registered value."""
    if not event["queryStringParameters"]["apiKey"]:
        raise InvalidAPIKey("Must have api key")

    api_key = event["queryStringParameters"]["apiKey"]

    record = APIKeyRepo.get_record_for_api_key(api_key)
    if not record:
        raise InvalidAPIKey(api_key)

    return _generate_accept_policy(record, event["methodArn"])
