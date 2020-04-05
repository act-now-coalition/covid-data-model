#!/usr/bin/env python

import datetime
import logging
import click
from libs import build_params
from libs.datasets import data_version
import run

WEB_DEPLOY_PATH = "../covid-projections/public/data"

_logger = logging.getLogger(__name__)


@click.group()
def main():
    pass


@main.command("county")
@click.option("--state", "-s")
@click.option(
    "--deploy",
    is_flag=True,
    help="Output data files to public data directory in local covid-projections.",
)
@click.option(
    "--summary-only",
    is_flag=True,
    help="Only runs the county summary if true.",
)
@data_version.with_git_version_click_option
def run_county(version: data_version.DataVersion, state=None, deploy=False, summary_only=False):
    """Run county level model."""
    min_date = datetime.datetime(2020, 3, 7)
    max_date = datetime.datetime(2020, 7, 6)

    output_dir = build_params.OUTPUT_DIR
    if deploy:
        output_dir = WEB_DEPLOY_PATH

    if not summary_only:
        run.run_county_level_forecast(
            min_date, max_date, country="USA", state=state, output_dir=output_dir
        )
    run.build_county_summary(min_date, state=state, output_dir=output_dir)
    # only write the version if we saved everything
    if not state and not summary_only:
        version.write_file('counties', output_dir)
    else:
        _logger.info('Skip version file because this is not a full run')


@main.command("state")
@click.option("--state", "-s")
@click.option(
    "--deploy",
    is_flag=True,
    help="Output data files to public data directory in local covid-projections.",
)
@data_version.with_git_version_click_option
def run_state(version: data_version.DataVersion, state=None, deploy=False):
    """Run State level model."""
    min_date = datetime.datetime(2020, 3, 7)
    max_date = datetime.datetime(2020, 7, 6)

    output_dir = build_params.OUTPUT_DIR
    if deploy:
        output_dir = WEB_DEPLOY_PATH

    run.run_state_level_forecast(
        min_date, max_date, country="USA", state=state, output_dir=output_dir
    )
    _logger.info(f'Wrote output to {output_dir}')
    # only write the version if we saved everything
    if not state:
        version.write_file('states', output_dir)
    else:
        _logger.info('Skip version file because this is not a full run')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
