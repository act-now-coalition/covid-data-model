#!/usr/bin/env python
"""
Entry point for covid-data-model CLI.

"""
import os
import logging
import click
import sentry_sdk
from pandarallel import pandarallel
import structlog
from structlog_sentry import SentryProcessor

from cli import api
from cli import run_top_counties_dataset
from cli import run_states_api
from cli import run_counties_api
from cli import compare_snapshots
from cli import utils
from libs.datasets import dataset_cache


@click.group()
@click.pass_context
# Disable pylint warning as suggested by https://stackoverflow.com/a/49680253
def entry_point(ctx):  # pylint: disable=no-value-for-parameter
    """Entry point for covid-data-model CLI."""
    with sentry_sdk.configure_scope() as scope:
        scope.set_tag("command", ctx.invoked_subcommand)
        # changes applied to scope remain after scope exits. See
        # https://github.com/getsentry/sentry-python/issues/184

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,  # required before SentryProcessor()
            # sentry_sdk creates events for level >= ERROR and keeps level >= INFO as breadcrumbs.
            SentryProcessor(level=logging.INFO),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]
    )


# adding the QA command
entry_point.add_command(compare_snapshots.compare_snapshots)

entry_point.add_command(run_top_counties_dataset.deploy_top_counties)
entry_point.add_command(run_counties_api.deploy_counties_api)
entry_point.add_command(run_states_api.deploy_states_api)
entry_point.add_command(api.main)
entry_point.add_command(utils.main)


# This code is executed when invoked as `python run.py ...` and will need to be changed if you
# want to add run.py to setup.py entry_points console_scripts. See
# https://github.com/pallets/click/issues/571#issuecomment-216261699
if __name__ == "__main__":
    sentry_sdk.init(os.getenv("SENTRY_DSN"))

    logging.basicConfig(level=logging.INFO)
    dataset_cache.set_pickle_cache_dir()
    pandarallel.initialize(progress_bar=False)
    try:
        entry_point()  # pylint: disable=no-value-for-parameter
    except Exception as e:
        # blanket catch exceptions at the entry point and send them to sentry
        sentry_sdk.capture_exception(e)
        raise e
