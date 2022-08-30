from typing import Dict, Optional
import uuid
import dataclasses
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
    Config.init()
    registry.initialize()

    sentry_sdk.init(
        dsn=Config.Constants.SENTRY_DSN,
        environment=Config.Constants.SENTRY_ENVIRONMENT,
        integrations=[AwsLambdaIntegration()],
        # We don't care about performance traces and they use up our transaction quota.
        traces_sample_rate=0,
    )


if IS_LAMBDA:
    # Only running initialization inside of a lambda environment
    init()


class InvalidInputError(Exception):
    """Error raised on invalid input for registration form parameters."""


@dataclasses.dataclass(frozen=True)
class RegistrationArguments:

    email: str

    is_crs_user: bool

    hubspot_token: Optional[str]

    page_uri: Optional[str]

    use_case: Optional[str]

    @staticmethod
    def make_from_json_body(data: Dict) -> "RegistrationArguments":
        if "email" not in data:
            raise InvalidInputError("Missing email parameter")

        email = data["email"]

        if not re.match(EMAIL_REGEX, email):
            raise InvalidInputError("Invalid email")

        is_crs_user = data.get("is_crs_user", False)

        hubspot_token = data.get("hubspot_token")

        page_uri = data.get("page_uri")

        use_case = data.get("use_case")

        return RegistrationArguments(
            email=email,
            is_crs_user=is_crs_user,
            hubspot_token=hubspot_token,
            page_uri=page_uri,
            use_case=use_case,
        )


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


def _create_api_key() -> str:
    return uuid.uuid4().hex


def _get_api_key(email: str, is_crs_user: bool) -> Optional[str]:
    api_key = APIKeyRepo.get_api_key(email)

    if api_key:
        # Record if existing users are covid response simulator users as they
        # may have registered not in the CRS.  This ensures that anyone who
        # gets their api key through the CRS registration will be recorded as a CRS user.
        if is_crs_user:
            APIKeyRepo.record_covid_response_simulator_user(email)

        return api_key

    return None


def _create_new_user(args: RegistrationArguments) -> str:
    email = args.email

    api_key = _create_api_key()
    APIKeyRepo.add_api_key(email, api_key, args.is_crs_user)

    welcome_email = _build_welcome_email(email, api_key)
    if EmailRepo.send_email(welcome_email):
        APIKeyRepo.record_email_sent(email)
    else:
        _logger.error(f"Failed to send email to {email}")

    # attempt to add hubspot contact, but don't block reg on failure.
    try:
        registry.hubspot_client.submit_reg_form(
            email, hubspot_token=args.hubspot_token, page_uri=args.page_uri, use_case=args.use_case
        )
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


def _make_error_message(message: str, status: int = 403) -> Dict:
    error_message = {"error": message}

    # Format follows Response Object from
    # https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/lambda-generating-http-responses-in-requests.html
    return {
        "status": status,
        "body": json.dumps(error_message),
        "headers": {"content-type": [{"value": "application/json"}]},
        "bodyEncoding": "text",
        "statusDescription": message,
    }


def check_api_key_edge(event, context):
    request = event["Records"][0]["cf"]["request"]

    query_parameters = urllib.parse.parse_qs(request["querystring"])
    # parse query parameter by taking first api key in query string arg.
    api_key = None
    for api_key in query_parameters.get("apiKey") or []:
        break

    if not api_key:
        return _make_error_message(
            "API key required. Visit https://apidocs.covidactnow.org/#register to get an API key"
        )

    record = APIKeyRepo.get_record_for_api_key(api_key)
    if not record:
        return _make_error_message("Invalid API key.")

    if record["email"] in Config.Constants.EMAIL_BLOCKLIST:
        error_message = "Unauthorized. Please contact api@covidactnow.org to restore access."
        return _make_error_message(error_message)

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

    try:
        args = RegistrationArguments.make_from_json_body(data)
    except InvalidInputError as exn:
        return {"status": 400, "errorMessage": str(exn), "headers": headers}

    api_key = _get_api_key(args.email, args.is_crs_user)

    if api_key:
        new_user = False
    else:
        api_key = _create_new_user(args)
        new_user = True

    body = {"api_key": api_key, "email": args.email, "new_user": new_user}

    headers["content-type"] = [{"key": "Content-Type", "value": "application/json"}]
    response = {"status": 200, "body": json.dumps(body), "headers": headers}

    return response
