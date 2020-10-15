import sentry_sdk
from awsauth import ses_client
from awsauth.the_registry import registry


class EmailRepo:
    @staticmethod
    def send_email(email: ses_client.EmailData) -> bool:
        """Sends email.

        Args:
            email: Email to send.

        Returns: True on successful send, False otherwise
        """
        client = registry.ses_client
        try:
            client.send_email(email)
            return True
        except Exception:
            sentry_sdk.capture_exception()
            return False
