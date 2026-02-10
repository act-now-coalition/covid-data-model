#!/usr/bin/env python3
"""Send deprecation notice emails to all API users.

Usage:
    # Preview who will receive the email from a JSON export
    python tools/send_deprecation_email.py --dry-run --from-json /path/to/api_keys_prod.json

    # Send deprecation emails to all users from a JSON export
    python tools/send_deprecation_email.py --send --from-json /path/to/api_keys_prod.json

    # Send to a single email address for testing
    python tools/send_deprecation_email.py --send --test-email you@example.com

    # Send to all users by scanning DynamoDB directly (requires .env)
    python tools/send_deprecation_email.py --send

Requires:
    - AWS credentials configured (via environment or ~/.aws/credentials)
    - For --test-email / --from-json: only AWS credentials (no .env needed)
    - For DynamoDB scan mode: a .env file with the required config
"""
import argparse
import json
import logging
import pathlib
import sys
import time

import boto3

# Add parent directory to path so we can import awsauth
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Config import is deferred to avoid pydantic version issues when running
# in --test-email mode (which doesn't need Config at all).

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)

DEPRECATION_EMAIL_PATH = pathlib.Path(__file__).resolve().parent.parent / "awsauth" / "deprecation_email.html"
SENT_LOG_PATH = pathlib.Path(__file__).resolve().parent / "deprecation_emails_sent.json"

# SES has a sending rate limit (typically 14 emails/sec for production accounts).
# Add a small delay between sends to stay well under the limit.
SEND_DELAY_SECONDS = 0.1


def get_all_emails_from_dynamo(table_name: str, region: str = "us-east-1") -> list:
    """Scan the DynamoDB API keys table and return all email addresses."""
    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(table_name)

    emails = []
    scan_kwargs = {"ProjectionExpression": "email"}

    while True:
        response = table.scan(**scan_kwargs)
        for item in response.get("Items", []):
            email = item.get("email")
            if email:
                emails.append(email)

        # Handle pagination
        if "LastEvaluatedKey" in response:
            scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
        else:
            break

    return sorted(set(emails))


def get_all_emails_from_json(json_path: str) -> list:
    """Load email addresses from a DynamoDB JSON export file."""
    data = json.loads(pathlib.Path(json_path).read_text())
    emails = [record["email"] for record in data if record.get("email")]
    return sorted(set(emails))


def load_sent_log() -> set:
    """Load the set of emails we've already sent to (for safe re-runs)."""
    if SENT_LOG_PATH.exists():
        data = json.loads(SENT_LOG_PATH.read_text())
        return set(data)
    return set()


def save_sent_log(sent_emails: set):
    """Persist the set of emails we've sent to."""
    SENT_LOG_PATH.write_text(json.dumps(sorted(sent_emails), indent=2))


FROM_EMAIL = "Covid Act Now API <api@covidactnow.org>"
REPLY_TO = "api@covidactnow.org"
SUBJECT = "Important: Covid Act Now API Shutting Down"
CONFIGURATION_SET = "api-welcome-emails"


def send_email_via_ses(ses_client, to_email: str, html: str):
    """Send a single email directly via boto3 SES (no Config dependency)."""
    ses_client.send_email(
        Destination={"ToAddresses": [to_email]},
        Message={
            "Body": {"Html": {"Charset": "UTF-8", "Data": html}},
            "Subject": {"Charset": "UTF-8", "Data": SUBJECT},
        },
        Source=FROM_EMAIL,
        ReplyToAddresses=[REPLY_TO],
        ConfigurationSetName=CONFIGURATION_SET,
    )


def main():
    parser = argparse.ArgumentParser(description="Send API deprecation emails")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="List emails without sending")
    group.add_argument("--send", action="store_true", help="Actually send the emails")
    parser.add_argument(
        "--test-email",
        type=str,
        default=None,
        help="Send only to this email address (for testing)",
    )
    parser.add_argument(
        "--from-json",
        type=str,
        default=None,
        help="Path to a DynamoDB JSON export file (avoids needing .env or DynamoDB access)",
    )
    args = parser.parse_args()

    if args.test_email:
        # Test mode: skip Config.init() entirely so no .env is needed.
        emails = [args.test_email]
        _logger.info(f"Test mode: will only send to {args.test_email}")
    elif args.from_json:
        # JSON export mode: no .env or DynamoDB access needed.
        _logger.info(f"Loading emails from {args.from_json}...")
        emails = get_all_emails_from_json(args.from_json)
        _logger.info(f"Found {len(emails)} unique email addresses.")
    else:
        # DynamoDB scan mode: need .env for table name and config
        from awsauth.config import Config
        Config.init()
        table_name = Config.Constants.API_KEY_TABLE_NAME
        region = Config.Constants.AWS_REGION
        _logger.info(f"Scanning DynamoDB table '{table_name}' for user emails...")
        emails = get_all_emails_from_dynamo(table_name, region)
        _logger.info(f"Found {len(emails)} unique email addresses.")

    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN - {len(emails)} emails would be sent:")
        print(f"{'='*60}\n")
        for email in emails:
            print(f"  {email}")
        print(f"\n{'='*60}")
        print(f"Total: {len(emails)} emails")
        print(f"{'='*60}")
        return

    # Sending mode
    sent_log = load_sent_log()
    skipped = 0
    sent = 0
    failed = 0

    ses_client = boto3.client("ses", region_name="us-east-1")
    email_html = DEPRECATION_EMAIL_PATH.read_text()

    for i, email in enumerate(emails, 1):
        if email in sent_log:
            _logger.info(f"[{i}/{len(emails)}] Skipping {email} (already sent)")
            skipped += 1
            continue

        try:
            send_email_via_ses(ses_client, email, email_html)
            sent_log.add(email)
            sent += 1
            _logger.info(f"[{i}/{len(emails)}] Sent to {email}")

            # Save progress after each successful send for safe re-runs
            save_sent_log(sent_log)

            # Rate limiting
            time.sleep(SEND_DELAY_SECONDS)

        except Exception:
            _logger.exception(f"[{i}/{len(emails)}] Failed to send to {email}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Done! Sent: {sent}, Skipped (already sent): {skipped}, Failed: {failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
