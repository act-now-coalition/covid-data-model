import pathlib
import sentry_sdk

from awsauth import ses_client

WELCOME_EMAIL_PATH = pathlib.Path(__file__).parent / "welcome_email.html"


def build_welcome_email(to_email: str, api_key: str):
    welcome_email_html = WELCOME_EMAIL_PATH.read_text()
    welcome_email_html = welcome_email_html.format(api_key=api_key)

    return ses_client.EmailData(
        subject="Your Covid Act Now API Key",
        from_email="api@covidactnow.org",
        reply_to="api@covidactnow.org",
        to_email=to_email,
        html=welcome_email_html,
        configuration_set="api-welcome-emails",
    )


def send_registration_email(email: ses_client.EmailData):
    client = ses_client.SESClient()
    try:
        client.send_email(email)
    except Exception:
        sentry_sdk.capture_exception()
