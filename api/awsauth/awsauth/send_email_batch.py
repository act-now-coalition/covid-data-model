#!/usr/bin/env python

import pathlib
import csv

import click

from awsauth import auth_app
from awsauth import ses_client
from awsauth.email_repo import EmailRepo

RISK_LEVEL_UPDATE_PATH = pathlib.Path(__file__).parent / "risk_level_update_email.html"


def _load_emails(path: pathlib.Path):

    with path.open() as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            yield row["email"]


def _build_email(to_email: str) -> ses_client.EmailData:
    email_html = RISK_LEVEL_UPDATE_PATH.read_text()

    return ses_client.EmailData(
        subject="[TEST] Update to risk levels",
        from_email="Covid Act Now API <api@covidactnow.org>",
        reply_to="api@covidactnow.org",
        to_email=to_email,
        html=email_html,
        configuration_set="api-risk-level-update-emails",
    )


@click.command()
@click.argument("emails-path", type=pathlib.Path)
@click.option("--dry-run", is_flag=True)
def send_emails(emails_path: pathlib.Path, dry_run=False):
    auth_app.init()
    email = "michael@covidactnow.org"
    email = _build_email(email)
    EmailRepo.send_email(email)
    # for email in _load_emails(emails_path):
    #     print(email)
    #     break


if __name__ == "__main__":
    send_emails()
