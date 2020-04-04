#!/usr/bin/env python

import datetime
import logging
import click
from libs import build_params
from libs.datasets.dataset_utils import data_version
import run

WEB_DEPLOY_PATH = "../covid-projections/public/data"


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
@click.option(
    '--git-hash',
    type=str,
    help='''
    | Git hash of the commit in covid-data-public to use.
    | If provided, covid-data-public must have no pending changes.
    | If omitted, the repository will be used as-is'''
)
def run_county(state=None, deploy=False, summary_only=False, git_hash=None):
    """Run county level model."""
    with data_version(git_hash) as version:
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
        if state is None and not summary_only:
            version.write_file('counties', output_dir)
        else:
            logging.info('Skip version file because this is not a full run')

@main.command("state")
@click.option("--state", "-s")
@click.option(
    "--deploy",
    is_flag=True,
    help="Output data files to public data directory in local covid-projections.",
)
@click.option(
    '--git-hash',
    type=str,
    help='''
    | Git hash of the commit in covid-data-public to use.
    | If provided, covid-data-public must have no pending changes.
    | If omitted, the repository will be used as-is'''
)
def run_state(state=None, deploy=False, git_hash=None):
    """Run State level model."""
    with data_version(git_hash) as version:
        min_date = datetime.datetime(2020, 3, 7)
        max_date = datetime.datetime(2020, 7, 6)

        output_dir = build_params.OUTPUT_DIR
        if deploy:
            output_dir = WEB_DEPLOY_PATH

        run.run_state_level_forecast(
            min_date, max_date, country="USA", state=state, output_dir=output_dir
        )
        logging.info(f'Wrote output to {output_dir}')
        # only write the version if we saved everything
        if state is None :
            version.write_file('states', output_dir)
        else:
            logging.info('Skip version file because this is not a full run')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
