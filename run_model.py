#!/usr/bin/env python
import pathlib
import datetime
import logging
import click
from libs.datasets import data_version
import run

_logger = logging.getLogger(__name__)


@click.group()
def main():
    pass


@main.command("county")
@click.option("--state", "-s")
@click.option(
    "--output",
    "-o",
    help="Output directory",
    type=pathlib.Path,
    default=pathlib.Path("results/county"),
)
@data_version.with_git_version_click_option
def run_county(
    version: data_version.DataVersion, output, state=None
):
    """Run county level model."""
    min_date = datetime.datetime(2020, 3, 7)
    max_date = datetime.datetime(2020, 7, 6)

    run.run_county_level_forecast(
        min_date, max_date, output, country="USA", state=state
    )
    if not state:
        version.write_file("county", output)
    else:
        _logger.info("Skip version file because this is not a full run")


@main.command("county-summary")
@click.option("--state", "-s")
@click.option(
    "--output",
    "-o",
    help="Output directory",
    type=pathlib.Path,
    default=pathlib.Path("results/county_summaries"),
)
@data_version.with_git_version_click_option
def run_county_summary(version: data_version.DataVersion, output, state=None):
    """Run county level model."""
    min_date = datetime.datetime(2020, 3, 7)
    run.build_county_summary(min_date, output, state=state)

    # only write the version if we saved everything
    if not state:
        version.write_file("county_summary", output)
    else:
        _logger.info("Skip version file because this is not a full run")


@main.command("state")
@click.option("--state", "-s")
@click.option(
    "--output",
    "-o",
    help="Output directory",
    type=pathlib.Path,
    default=pathlib.Path("results/state"),
)
@data_version.with_git_version_click_option
def run_state(version: data_version.DataVersion, output, state=None):
    """Run State level model."""
    min_date = datetime.datetime(2020, 3, 7)
    max_date = datetime.datetime(2020, 7, 6)

    run.run_state_level_forecast(min_date, max_date, output, country="USA", state=state)
    _logger.info(f"Wrote output to {output}")
    # only write the version if we saved everything
    if not state:
        version.write_file("states", output)
    else:
        _logger.info("Skip version file because this is not a full run")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
