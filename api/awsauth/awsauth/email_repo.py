import sentry_sdk
from awsauth import ses_client


class EmailRepo:
    @staticmethod
    def send_email(email: ses_client.EmailData) -> bool:
        """Sends email.

        Args:
            email: Email to send.

        Returns: True on successful send, False otherwise
        """
        client = ses_client.SESClient()
        try:
            client.send_email(email)
            return True
        except Exception:
            sentry_sdk.capture_exception()
            return False
