#!/usr/bin/env python
"""
Entry point for covid-data-model CLI.

"""
import os
import logging
import click
import sentry_sdk
from pandarallel import pandarallel

from cli import run_top_counties_dataset
from cli import run_states_api
from cli import run_counties_api
from cli import compare_snapshots

from cli import api


@click.group()
def entry_point():
    """Entry point for covid-data-model CLI."""
    pass


# adding the QA command
entry_point.add_command(compare_snapshots.compare_snapshots)

entry_point.add_command(run_top_counties_dataset.deploy_top_counties)
entry_point.add_command(run_counties_api.deploy_counties_api)
entry_point.add_command(run_states_api.deploy_states_api)
entry_point.add_command(api.main)

if __name__ == "__main__":
    sentry_sdk.init(os.getenv("SENTRY_DSN"))
    logging.basicConfig(level=logging.INFO)
    pandarallel.initialize(progress_bar=False)
    try:
        entry_point()
    except Exception as e:
        # blanket catch exceptions at the entry point and send them to sentry
        sentry_sdk.capture_exception(e)
        raise e
