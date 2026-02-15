from typing import Dict, Optional
import uuid
import dataclasses
import urllib.parse
import datetime

import os
import re
import json
import pathlib
import logging

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

# Set to None to keep API running, or a date string (YYYY-MM-DD) to shut down
# after that date. When set, all API requests will receive a deprecation message
# instead of being forwarded to the S3 backend.
API_SHUTDOWN_DATE = "2026-03-11"

API_SHUTDOWN_MESSAGE = (
    "The Covid Act Now API has been permanently shut down. "
    "Data snapshots are no longer being updated and the API is no longer serving requests. "
    "If you need assistance, please reach out to api@covidactnow.org."
)

# Bulk data files that are no longer available for download due to excessive
# bandwidth costs from automated scraping. Users can contact us for a copy.
BLOCKED_BULK_FILES = ["counties.timeseries.json", "counties.timeseries.csv"]
BULK_FILE_MESSAGE = (
    "This bulk data file is no longer available for download. "
    "For a copy of this data, please contact api@covidactnow.org."
)

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
    # If a shutdown date is set and we've passed it, reject all requests immediately.
    if API_SHUTDOWN_DATE and datetime.date.today() >= datetime.date.fromisoformat(
        API_SHUTDOWN_DATE
    ):
        return _make_error_message(API_SHUTDOWN_MESSAGE)

    request = event["Records"][0]["cf"]["request"]

    # Block bulk file downloads before any further processing.
    if any(blocked in request["uri"] for blocked in BLOCKED_BULK_FILES):
        return _make_error_message(BULK_FILE_MESSAGE)

    query_parameters = urllib.parse.parse_qs(request["querystring"])
    # parse query parameter by taking first api key in query string arg.
    api_key = None
    for api_key in query_parameters.get("apiKey") or []:
        break

    if not api_key:
        return _make_error_message(
            "API key required. The Covid Act Now API is no longer accepting new registrations. "
            "If you need assistance, please reach out to api@covidactnow.org."
        )

    record = APIKeyRepo.get_record_for_api_key(api_key)
    if not record:
        return _make_error_message(
            "Invalid API key. The Covid Act Now API is no longer accepting new registrations. "
            "If you need assistance, please reach out to api@covidactnow.org."
        )

    if record.get("blocked") or record["email"] in Config.Constants.EMAIL_BLOCKLIST:
        return _make_error_message(
            "Access has been suspended. "
            "If you need assistance, please reach out to api@covidactnow.org."
        )

    _record_successful_request(request, record)

    # Return request, which forwards to S3 backend.
    return request


def register_edge(event, context):
    """API Registration function used in Lambda@Edge cloudfront distribution.

    Registration is permanently closed. All requests are rejected.
    """
    return _make_error_message(
        "The Covid Act Now API is no longer accepting new registrations. "
        "If you need assistance, please reach out to api@covidactnow.org."
    )
