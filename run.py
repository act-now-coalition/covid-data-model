#!/usr/bin/env python
"""
Entry point for covid-data-model CLI.

"""
import os
import logging
import click
import sentry_sdk
from cli import run_data
from cli import run_model
from cli import run_dod_dataset
from cli import run_top_counties_dataset
from cli import run_states_api
from cli import run_counties_api

from cli import api


@click.group()
def entry_point():
    """Entry point for covid-data-model CLI."""
    pass


entry_point.add_command(run_data.main)
entry_point.add_command(run_model.main)
entry_point.add_command(run_dod_dataset.deploy_dod_projections)
entry_point.add_command(run_top_counties_dataset.deploy_top_counties)
entry_point.add_command(run_counties_api.deploy_counties_api)
entry_point.add_command(run_states_api.deploy_states_api)
entry_point.add_command(api.main)

if __name__ == "__main__":
    sentry_sdk.init(os.getenv("SENTRY_DSN"))
    logging.basicConfig(level=logging.INFO)
    try: 
        entry_point()
    except Exception as e: 
        # blanket catch excpetions at the entry point and send them to sentry
        sentry_sdk.capture_exception(e)
        raise e