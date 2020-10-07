from typing import Optional
import dataclasses

import boto3


@dataclasses.dataclass
class EmailData:

    subject: str
    from_email: str
    reply_to: Optional[str]
    to_email: str
    html: str
    configuration_set: str
    group: Optional[str] = None


class SESClient:
    def __init__(self, client=None):
        self._client = client or boto3.client("ses")

    def send_email(self, email: EmailData):

        response = self._client.send_email(
            Destination={"ToAddresses": [email.to_email]},
            Message={
                "Body": {"Html": {"Charset": "UTF-8", "Data": email.html},},
                "Subject": {"Charset": "UTF-8", "Data": email.subject},
            },
            Source=email.from_email,
            ReplyToAddresses=[email.reply_to] if email.reply_to else None,
            ConfigurationSetName=email.configuration_set,
        )
