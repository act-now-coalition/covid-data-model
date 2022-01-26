#!/usr/bin/env python
"""
Entry point for covid-data-model CLI.

"""
import logging
import click
from datapublic import common_init
from pandarallel import pandarallel

from cli import api
from cli import data
from cli import utils


@click.group()
@click.pass_context
# Disable pylint warning as suggested by https://stackoverflow.com/a/49680253
def entry_point(ctx):  # pylint: disable=no-value-for-parameter
    """Entry point for covid-data-model CLI."""
    common_init.configure_logging(command=ctx.invoked_subcommand)

    pandarallel.initialize(progress_bar=False)


entry_point.add_command(api.main)
entry_point.add_command(utils.main)
entry_point.add_command(data.main)


# This code is executed when invoked as `python run.py ...` and will need to be changed if you
# want to add run.py to setup.py entry_points console_scripts. See
# https://github.com/pallets/click/issues/571#issuecomment-216261699
if __name__ == "__main__":
    try:
        entry_point()  # pylint: disable=no-value-for-parameter
    except Exception:
        # According to https://github.com/getsentry/sentry-python/issues/480 Sentry is expected
        # to create an event when this is called.
        logging.exception("Exception reached __main__")
        raise
