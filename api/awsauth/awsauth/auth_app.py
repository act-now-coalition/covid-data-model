import uuid
import os
import re
import json
import pathlib
import logging

import sentry_sdk
from sentry_sdk.integrations.aws_lambda import AwsLambdaIntegration

from awsauth.api_key_repo import APIKeyRepo
from awsauth.email_repo import EmailRepo
from awsauth import ses_client

IS_LAMBDA = os.getenv("LAMBDA_TASK_ROOT")
WELCOME_EMAIL_PATH = pathlib.Path(__file__).parent / "welcome_email.html"


_logger = logging.getLogger(__name__)


def init():
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        integrations=[AwsLambdaIntegration()],
        traces_sample_rate=1.0,  # adjust the sample rate in production as needed
    )


if IS_LAMBDA:
    # Only running initialization inside of a lambda environment
    init()


# Fairly permissive email regex taken from
# https://stackoverflow.com/questions/8022530/how-to-check-for-valid-email-address#comment52453093_8022584
EMAIL_REGEX = r"[^@\s]+@[^@\s]+\.[a-zA-Z0-9]+$"


class InvalidAPIKey(Exception):
    def __init__(self, api_key):
        super().__init__(f"Invalid API Key: {api_key}")
        self.api_key = api_key


def _build_welcome_email(to_email: str, api_key: str) -> ses_client.EmailData:
    welcome_email_html = WELCOME_EMAIL_PATH.read_text()
    welcome_email_html = welcome_email_html.format(api_key=api_key)

    return ses_client.EmailData(
        subject="Welcome to the Covid Act Now API!",
        from_email="api@covidactnow.org",
        reply_to="api@covidactnow.org",
        to_email=to_email,
        html=welcome_email_html,
        configuration_set="api-welcome-emails",
    )


def _create_api_key(email: str) -> str:
    return uuid.uuid4().hex


def _get_or_create_api_key(email):
    api_key = APIKeyRepo.get_api_key(email)
    if api_key:
        return api_key

    _logger.info(f"No API Key found for email {email}, creating new key")

    api_key = _create_api_key(email)
    APIKeyRepo.add_api_key(email, api_key)

    welcome_email = _build_welcome_email(email, api_key)
    if not EmailRepo.send_email(welcome_email):
        _logger.error(f"Failed to send email to {email}")

    return api_key


def _check_api_key(api_key):
    if not APIKeyRepo.get_record_for_api_key(api_key):
        raise InvalidAPIKey(api_key)


def register(event, context):

    if not "email" in event:
        raise ValueError("Missing email parameter")

    email = event["email"]
    if not re.match(EMAIL_REGEX, email):
        return {"statusCode": 400, "errorMessage": "Invalid email"}

    api_key = _get_or_create_api_key(email)
    body = {"api_key": api_key, "email": email}

    response = {"statusCode": 200, "body": json.dumps(body)}

    return response


def _generate_accept_policy(user_record: dict, method_arn):
    # Want to give access to all data so that we can cache policy responses, not just the
    # level or file they requested.
    method_arn = re.sub(r"(.*/GET/).*", r"\1*", method_arn)
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


def _generate_deny_policy(method_arn):
    return {
        "policyDocument": {
            "Version": "2012-10-17",
            "Statement": [
                {"Action": "execute-api:Invoke", "Effect": "Deny", "Resource": method_arn}
            ],
        },
    }


def check_api_key(event, context):
    """Checks API Key included in request for registered value."""
    method_arn = event["methodArn"]

    if not event["queryStringParameters"]["apiKey"]:
        return _generate_deny_policy(method_arn)

    api_key = event["queryStringParameters"]["apiKey"]

    record = APIKeyRepo.get_record_for_api_key(api_key)
    if not record:
        return _generate_deny_policy(method_arn)

    return _generate_accept_policy(record, method_arn)
