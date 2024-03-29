from typing import List
import pydantic


class EnvConstants(pydantic.BaseSettings):
    """Environment constants.

    Constants should be set in .env file with keys matching variable names
    at root of project.
    """

    API_KEY_TABLE_NAME: str

    FIREHOSE_TABLE_NAME: str

    SENTRY_DSN: str

    SENTRY_ENVIRONMENT: str

    EMAILS_ENABLED: bool

    AWS_REGION: str = "us-east-1"

    HUBSPOT_API_KEY: str

    HUBSPOT_ENABLED: bool

    # Hubspot portal id differs between environments. We have a separate
    # hubspot portal for our dev environment.
    HUBSPOT_PORTAL_ID: str

    # GUID of hubspot form used for API signup
    HUBSPOT_REG_FORM_GUID: str

    EMAIL_BLOCKLIST: List[str]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class Config:

    Constants: EnvConstants = None

    @staticmethod
    def init():
        Config.Constants = EnvConstants()
