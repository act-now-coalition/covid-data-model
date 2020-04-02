#!/usr/bin/env python

import datetime
import logging
import click
from libs import build_params
import run
print(run)
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
def run_county(state=None, deploy=False):
    """Run county level model."""
    min_date = datetime.datetime(2020, 3, 7)
    max_date = datetime.datetime(2020, 7, 6)

    output_dir = build_params.OUTPUT_DIR
    if deploy:
        output_dir = WEB_DEPLOY_PATH

    run.run_county_level_forecast(
        min_date, max_date, country="USA", state=state, output_dir=output_dir
    )
    run.build_county_summary(min_date, state=state, output_dir=output_dir)


@main.command("state")
@click.option("--state", "-s")
@click.option(
    "--deploy",
    is_flag=True,
    help="Output data files to public data directory in local covid-projections.",
)
def run_state(state=None, deploy=False):
    """Run State level model."""
    min_date = datetime.datetime(2020, 3, 7)
    max_date = datetime.datetime(2020, 7, 6)

    output_dir = build_params.OUTPUT_DIR
    if deploy:
        output_dir = WEB_DEPLOY_PATH

    run.run_state_level_forecast(
        min_date, max_date, country="USA", state=state, output_dir=output_dir
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
