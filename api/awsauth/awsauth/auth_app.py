import uuid
import urllib.parse
import datetime
import base64
import os
import re
import json
import pathlib
import logging

import sentry_sdk
from sentry_sdk.integrations.aws_lambda import AwsLambdaIntegration

from awsauth import ses_client
from awsauth.api_key_repo import APIKeyRepo
from awsauth.email_repo import EmailRepo
from awsauth.firehose_client import FirehoseClient
from awsauth.config import Config


IS_LAMBDA = os.getenv("LAMBDA_TASK_ROOT")
WELCOME_EMAIL_PATH = pathlib.Path(__file__).parent / "welcome_email.html"
os.environ["AWS_REGION"] = "us-east-1"


_logger = logging.getLogger(__name__)

FIREHOSE_CLIENT = None

# Headers needed to return for CORS OPTIONS request
CORS_OPTIONS_HEADERS = {
    "access-control-allow-origin": [{"key": "Access-Control-Allow-Origin", "value": "*"}],
    "access-control-allow-headers": [
        {
            "key": "Access-Control-Allow-Headers",
            "value": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
        }
    ],
    "access-control-allow-methods": [{"key": "Access-Control-Allow-Methods", "value": "POST"}],
    "access-control-max-age": [{"key": "Access-Control-Max-Age", "value": "86400"}],
}


def init():
    global FIREHOSE_CLIENT
    Config.init()

    FIREHOSE_CLIENT = FirehoseClient()

    sentry_sdk.init(
        dsn=Config.Constants.SENTRY_DSN,
        environment=Config.Constants.SENTRY_ENVIRONMENT,
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
        from_email="Covid Act Now API <api@covidactnow.org>",
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
    if EmailRepo.send_email(welcome_email):
        APIKeyRepo.record_email_sent(email)
    else:
        _logger.error(f"Failed to send email to {email}")

    return api_key


def _record_successful_request(request: dict, record: dict):
    data = {
        "timestamp": datetime.datetime.utcnow().isoformat().replace("T", " "),
        "email": record["email"],
        "path": request["uri"],
        "ip": request["clientIp"],
    }

    FIREHOSE_CLIENT.put_data(Config.Constants.FIREHOSE_TABLE_NAME, data)


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


def check_api_key_edge(event, context):
    request = event["Records"][0]["cf"]["request"]

    query_parameters = urllib.parse.parse_qs(request["querystring"])
    # parse query parameter by taking first api key in query string arg.
    api_key = None
    for api_key in query_parameters.get("apiKey") or []:
        break

    if not api_key:
        return {"status": 403, "statusDescription": "Unauthorized"}

    record = APIKeyRepo.get_record_for_api_key(api_key)
    if not record:
        return {"status": 403, "statusDescription": "Unauthorized"}

    _record_successful_request(request, record)

    # Return request, which forwards to S3 backend.
    return request


def register_edge(event, context):
    """API Registration function used in Lambda@Edge cloudfront distribution.

    Handles CORS for OPTIONS and POST requests.
    """
    # For more details on the structure of the event, see:
    # https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/lambda-event-structure.html#example-viewer-request
    request = event["Records"][0]["cf"]["request"]

    if request["method"] == "OPTIONS":
        return {"status": 200, "headers": CORS_OPTIONS_HEADERS}

    data = request["body"]["data"]

    # Data can either be 'text' or 'base64'. If 'base64' decode data.
    if request["body"]["encoding"] == "base64":
        data = base64.b64decode(data)

    data = json.loads(data)

    # Headers needed for CORS.
    headers = {
        "access-control-allow-origin": [{"key": "Access-Control-Allow-Origin", "value": "*"}],
    }
    if "email" not in data:
        return {"status": 400, "errorMessage": "Missing email parameter", "headers": headers}

    email = data["email"]
    if not re.match(EMAIL_REGEX, email):
        return {"status": 400, "errorMessage": "Invalid email", "headers": headers}

    api_key = _get_or_create_api_key(email)
    body = {"api_key": api_key, "email": email}

    headers["content-type"] = [{"key": "Content-Type", "value": "application/json"}]
    response = {"status": 200, "body": json.dumps(body), "headers": headers}

    return response
