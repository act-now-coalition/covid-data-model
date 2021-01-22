from typing import Optional
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
from awsauth.config import Config
from awsauth.the_registry import registry
from awsauth import hubspot_client


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
    registry.initialize()

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


def _get_or_create_api_key(email: str, is_crs_user: bool):
    api_key = APIKeyRepo.get_api_key(email)

    if api_key:
        # Record if existing users are covid response simulator users as they
        # may have registered not in the CRS.  This ensures that anyone who
        # gets their api key through the CRS registration will be recorded as a CRS user.
        if is_crs_user:
            APIKeyRepo.record_covid_response_simulator_user(email)

        return api_key

    _logger.info(f"No API Key found for email {email}, creating new key")

    api_key = _create_api_key(email)
    APIKeyRepo.add_api_key(email, api_key, is_crs_user)

    welcome_email = _build_welcome_email(email, api_key)
    if EmailRepo.send_email(welcome_email):
        APIKeyRepo.record_email_sent(email)
    else:
        _logger.error(f"Failed to send email to {email}")

    # attempt to add hubspot contact, but don't block reg on failure.
    try:
        registry.hubspot_client.create_contact(email)
    except hubspot_client.HubSpotAPICallFailed:
        _logger.error("HubSpot call failed")
        sentry_sdk.capture_exception()

    return api_key


def _record_successful_request(request: dict, record: dict):
    data = {
        "timestamp": datetime.datetime.utcnow().isoformat().replace("T", " "),
        "email": record["email"],
        "path": request["uri"],
        "ip": request["clientIp"],
        "is_covid_response_simulator_user": record.get("is_covid_response_simulator_user", False),
    }

    registry.firehose_client.put_data(Config.Constants.FIREHOSE_TABLE_NAME, data)


def _get_query_parameter(request, key) -> Optional[str]:
    query_parameters = urllib.parse.parse_qs(request["querystring"])
    # parse query parameter by taking first api key in query string arg.
    param = None
    for param in query_parameters.get(key) or []:
        break

    return param


def check_api_key_edge(event, context):
    request = event["Records"][0]["cf"]["request"]

    api_key = _get_query_parameter(request, "apiKey")
    if not api_key:
        return {"status": 403, "statusDescription": "Unauthorized"}

    record = APIKeyRepo.get_record_for_api_key(api_key)
    if not record:
        return {"status": 403, "statusDescription": "Unauthorized"}

    if record["email"] in Config.Constants.EMAIL_BLOCKLIST:
        error_message = {
            "error": "Unauthorized. Please contact api@covidactnow.org to restore access."
        }
        return {
            "status": 403,
            "body": json.dumps(error_message),
            "headers": {"content-type": [{"value": "application/json"}]},
            "bodyEncoding": "text",
            "statusDescription": (
                "Unauthorized. Please contact api@covidactnow.org to restore access."
            ),
        }

    _record_successful_request(request, record)

    # Check for days back parameter and if included and a supported file,
    # modify URI to point to file with truncated history. This is a simple way
    # of expanding the API to add a bit of flexibility with the files we serve and
    # let people query data that is more of a fixed size and is not unbounded in size.
    days_back = _get_query_parameter(request, "daysBack")

    # Currently only bulk timeseries json files support daysBack
    days_back_supported_regex = "/v2/([a-z]+)\.timeseries\.json"
    if days_back and re.match(days_back_supported_regex, request["uri"]):
        uri = request["uri"]
        *beginning, suffix = uri.split(".")
        final_uri = ".".join([*beginning, f"{days_back}d", suffix])
        request["uri"] = final_uri

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

    is_crs_user = data.get("is_crs_user", False)
    api_key = _get_or_create_api_key(email, is_crs_user)
    body = {"api_key": api_key, "email": email}

    headers["content-type"] = [{"key": "Content-Type", "value": "application/json"}]
    response = {"status": 200, "body": json.dumps(body), "headers": headers}

    return response
