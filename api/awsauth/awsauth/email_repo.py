import logging

from awsauth import ses_client
from awsauth.the_registry import registry

logger = logging.getLogger(__name__)


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
            logger.exception("Failed to send email")
            return False
